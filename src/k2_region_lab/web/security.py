from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from k2_region_lab.http_security import SlidingWindowRateLimiter


SESSION_COOKIE = "__Host-k2lab-session"
CSRF_COOKIE = "k2lab-csrf"


@dataclass(frozen=True)
class ControlPlaneSecuritySettings:
    enabled: bool = False
    loopback_only: bool = False
    trusted_proxy_secret: str = ""
    allowed_subject: str = ""
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    require_mfa: bool = True
    session_ttl_seconds: int = 3600
    read_requests_per_minute: int = 600
    write_requests_per_minute: int = 120
    provisioning_requests_per_minute: int = 12
    max_request_bytes: int = 65 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.enabled and self.loopback_only:
            raise RuntimeError("Hosted authentication and loopback-only mode are mutually exclusive")
        if not self.enabled:
            return
        if len(self.trusted_proxy_secret) < 32:
            raise RuntimeError("K2LAB_TRUSTED_PROXY_SECRET must contain at least 32 characters")
        if not self.allowed_subject or len(self.allowed_subject) > 191:
            raise RuntimeError("K2LAB_AUTH_ALLOWED_SUBJECT is required for hosted mode")
        if not 300 <= self.session_ttl_seconds <= 86_400:
            raise RuntimeError("K2LAB_SESSION_TTL_SECONDS must be between 300 and 86400")
        for origin in self.allowed_origins:
            parsed = urlparse(origin)
            if parsed.scheme != "https" or not parsed.netloc or parsed.path not in {"", "/"}:
                raise RuntimeError("Hosted K2LAB_ALLOWED_ORIGINS must be HTTPS origins")

    @classmethod
    def from_environment(cls, *, production: bool) -> ControlPlaneSecuritySettings:
        if not production:
            return cls()
        origins = tuple(
            value.strip().rstrip("/")
            for value in os.environ.get("K2LAB_ALLOWED_ORIGINS", "").split(",")
            if value.strip()
        )
        if not origins:
            raise RuntimeError("K2LAB_ALLOWED_ORIGINS is required for the RunPod backend")
        return cls(
            enabled=True,
            trusted_proxy_secret=os.environ.get("K2LAB_TRUSTED_PROXY_SECRET", ""),
            allowed_subject=os.environ.get("K2LAB_AUTH_ALLOWED_SUBJECT", ""),
            allowed_origins=origins,
            require_mfa=os.environ.get("K2LAB_REQUIRE_MFA", "true").casefold()
            not in {"0", "false", "no"},
            session_ttl_seconds=int(os.environ.get("K2LAB_SESSION_TTL_SECONDS", "3600")),
        )

    @classmethod
    def local_single_user(cls, *, port: int) -> ControlPlaneSecuritySettings:
        return cls(
            loopback_only=True,
            allowed_origins=(
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
            ),
        )


class BrowserSession(BaseModel):
    authenticated: bool
    subject: str
    mfa_verified: bool
    expires_at: datetime


@dataclass
class _Session:
    subject: str
    csrf_token: str
    mfa_verified: bool
    created_at: datetime
    expires_at: datetime


class BrowserSessionManager:
    def __init__(self, settings: ControlPlaneSecuritySettings) -> None:
        self.settings = settings
        self._sessions: dict[str, _Session] = {}

    def open(self, request: Request, response: Response) -> BrowserSession:
        if not self.settings.enabled:
            return BrowserSession(
                authenticated=True,
                subject="local-development",
                mfa_verified=False,
                expires_at=datetime.now(UTC) + timedelta(days=1),
            )
        supplied_secret = request.headers.get("X-K2-Proxy-Secret", "")
        subject = request.headers.get("X-K2-Authenticated-User", "").strip()
        mfa_verified = request.headers.get("X-K2-Authenticated-MFA", "").casefold() in {
            "1",
            "true",
            "yes",
        }
        if not hmac.compare_digest(
            supplied_secret.encode(), self.settings.trusted_proxy_secret.encode()
        ):
            raise SessionSecurityError(
                "authentication_required", "The trusted authentication assertion is missing.", 401
            )
        if not hmac.compare_digest(subject.encode(), self.settings.allowed_subject.encode()):
            raise SessionSecurityError(
                "account_forbidden", "This account is not authorized for this control plane.", 403
            )
        if self.settings.require_mfa and not mfa_verified:
            raise SessionSecurityError(
                "mfa_required", "Complete multi-factor authentication before continuing.", 403
            )
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=self.settings.session_ttl_seconds)
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        self._sessions[session_id] = _Session(
            subject=subject,
            csrf_token=csrf_token,
            mfa_verified=mfa_verified,
            created_at=now,
            expires_at=expires_at,
        )
        self._expire(now)
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            max_age=self.settings.session_ttl_seconds,
            secure=True,
            httponly=True,
            samesite="strict",
            path="/",
        )
        response.set_cookie(
            CSRF_COOKIE,
            csrf_token,
            max_age=self.settings.session_ttl_seconds,
            secure=True,
            httponly=False,
            samesite="strict",
            path="/",
        )
        return BrowserSession(
            authenticated=True,
            subject=subject,
            mfa_verified=mfa_verified,
            expires_at=expires_at,
        )

    def authenticate(self, request: Request) -> _Session | None:
        session_id = request.cookies.get(SESSION_COOKIE, "")
        session = self._sessions.get(session_id)
        now = datetime.now(UTC)
        if session is None or session.expires_at <= now:
            if session_id:
                self._sessions.pop(session_id, None)
            return None
        return session

    def close(self, request: Request, response: Response) -> None:
        self._sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
        response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True)
        response.delete_cookie(CSRF_COOKIE, path="/", secure=True, httponly=False)

    def _expire(self, now: datetime) -> None:
        for session_id, session in list(self._sessions.items()):
            if session.expires_at <= now:
                self._sessions.pop(session_id, None)


class SessionSecurityError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ControlPlaneSecurityMiddleware(BaseHTTPMiddleware):
    _SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
    _PUBLIC_PATHS = frozenset({"/api/v1/health", "/api/v1/auth/session"})

    def __init__(
        self,
        app,
        *,
        settings: ControlPlaneSecuritySettings,
        sessions: BrowserSessionManager,
    ) -> None:
        super().__init__(app)
        self.settings = settings
        self.sessions = sessions
        self.limiter = SlidingWindowRateLimiter()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        error = self._loopback_error(request)
        if error is None:
            error = self._request_size_error(request)
        session: _Session | None = None
        if error is None and self.settings.enabled and request.url.path not in self._PUBLIC_PATHS:
            session = self.sessions.authenticate(request)
            if session is None:
                error = self._error(
                    "authentication_required", "Sign in before using the control plane.", 401
                )
            elif request.method not in self._SAFE_METHODS:
                error = self._validate_mutation(request, session)
        if (
            error is None
            and self.settings.loopback_only
            and request.method not in self._SAFE_METHODS
            and request.url.path not in self._PUBLIC_PATHS
        ):
            origin = request.headers.get("Origin", "").rstrip("/")
            if origin not in self.settings.allowed_origins:
                error = self._error(
                    "origin_forbidden",
                    "Local control-plane changes require the K2 Region Lab browser origin.",
                    403,
                )
        identity = session.subject if session else request.client.host if request.client else "unknown"
        if error is None:
            limit = self._rate_limit(request)
            allowed, retry_after = await self.limiter.allow(
                f"{identity}:{request.method}:{self._rate_class(request)}", limit=limit
            )
            if not allowed:
                error = self._error(
                    "rate_limit_exceeded",
                    "Too many requests; wait briefly before retrying.",
                    429,
                    retry_after=retry_after,
                )
        if error is not None:
            self._secure(error)
            return error
        request.state.auth_subject = session.subject if session else "local-development"
        response = await call_next(request)
        self._secure(response)
        return response

    def _loopback_error(self, request: Request) -> JSONResponse | None:
        if not self.settings.loopback_only:
            return None
        client_host = request.client.host if request.client else ""
        try:
            client_is_loopback = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            client_is_loopback = False
        request_host = (request.url.hostname or "").casefold()
        if client_is_loopback and request_host in {"127.0.0.1", "localhost", "::1"}:
            return None
        return self._error(
            "loopback_required",
            "The single-user control plane is available only from this computer.",
            403,
        )

    def _request_size_error(self, request: Request) -> JSONResponse | None:
        try:
            length = int(request.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error("request_size_invalid", "The request size is invalid.", 400)
        if length > self.settings.max_request_bytes:
            return self._error("request_too_large", "The request body is too large.", 413)
        return None

    def _validate_mutation(self, request: Request, session: _Session) -> JSONResponse | None:
        origin = request.headers.get("Origin", "").rstrip("/")
        if origin not in self.settings.allowed_origins:
            return self._error("origin_forbidden", "The request origin is not allowed.", 403)
        supplied = request.headers.get("X-CSRF-Token", "")
        cookie = request.cookies.get(CSRF_COOKIE, "")
        if not supplied or not cookie or not hmac.compare_digest(supplied, session.csrf_token):
            return self._error("csrf_failed", "The request security token is invalid.", 403)
        if not hmac.compare_digest(cookie, session.csrf_token):
            return self._error("csrf_failed", "The request security token is invalid.", 403)
        return None

    def _rate_limit(self, request: Request) -> int:
        rate_class = self._rate_class(request)
        if rate_class == "provisioning":
            return self.settings.provisioning_requests_per_minute
        if request.method in self._SAFE_METHODS:
            return self.settings.read_requests_per_minute
        return self.settings.write_requests_per_minute

    @staticmethod
    def _rate_class(request: Request) -> str:
        path = request.url.path
        if (
            path in {"/api/v1/workspaces", "/api/v1/workspace-plans"}
            or "/migrations" in path
            or path.endswith(("/start", "/stop", "/terminate"))
        ):
            return "provisioning"
        if "/uploads/" in path and "/chunks/" in path:
            return "upload-chunk"
        if "/downloads/" in path:
            return "download"
        if "/jobs" in path:
            return "job"
        return "api"

    @staticmethod
    def _error(
        code: str,
        message: str,
        status_code: int,
        *,
        retry_after: int | None = None,
    ) -> JSONResponse:
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
        return JSONResponse(
            status_code=status_code,
            content={"code": code, "message": message},
            headers=headers,
        )

    @staticmethod
    def _secure(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
