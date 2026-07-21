from __future__ import annotations

from datetime import datetime
from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken

from k2_region_lab.web.domain import CredentialStatus, WorkspaceError
from k2_region_lab.web.state_store import RunPodStateStore, StoredCredential


class CredentialVault(Protocol):
    async def store(
        self,
        credential_id: str,
        plaintext: str,
        *,
        key_hint: str | None = None,
        validated_at: datetime | None = None,
    ) -> None: ...

    async def retrieve(self, credential_id: str) -> str | None: ...

    async def delete(self, credential_id: str) -> None: ...

    async def status(self, credential_id: str) -> CredentialStatus: ...


class EncryptedMemoryCredentialVault:
    """Process-local encrypted vault used until the durable KMS store is implemented.

    The encryption key is supplied by the control-plane environment. Ciphertext is kept
    separately from application domain records, and plaintext only exists while a provider
    call is being prepared. This class deliberately provides no method that exposes its
    ciphertext collection to application callers.
    """

    def __init__(self, encryption_key: str | bytes) -> None:
        key = encryption_key.encode("ascii") if isinstance(encryption_key, str) else encryption_key
        try:
            self._cipher = Fernet(key)
        except (TypeError, ValueError) as error:
            raise ValueError("Credential encryption key must be a valid Fernet key") from error
        self._encrypted: dict[str, bytes] = {}
        self._statuses: dict[str, CredentialStatus] = {}

    async def store(
        self,
        credential_id: str,
        plaintext: str,
        *,
        key_hint: str | None = None,
        validated_at: datetime | None = None,
    ) -> None:
        self._encrypted[credential_id] = self._cipher.encrypt(plaintext.encode("utf-8"))
        self._statuses[credential_id] = CredentialStatus(
            configured=True,
            key_hint=key_hint,
            validated_at=validated_at,
        )

    async def retrieve(self, credential_id: str) -> str | None:
        ciphertext = self._encrypted.get(credential_id)
        if ciphertext is None:
            return None
        try:
            return self._cipher.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as error:
            raise WorkspaceError(
                "credential_decryption_failed",
                "The stored provider credential could not be decrypted.",
                status_code=500,
            ) from error

    async def delete(self, credential_id: str) -> None:
        self._encrypted.pop(credential_id, None)
        self._statuses.pop(credential_id, None)

    async def status(self, credential_id: str) -> CredentialStatus:
        return self._statuses.get(credential_id, CredentialStatus(configured=False))


class DatabaseCredentialVault:
    """Fernet envelope stored in the durable state repository.

    The wrapping key is external to the database and can be supplied by a KMS bootstrap
    integration. Only ciphertext and non-secret metadata are persisted.
    """

    def __init__(self, state_store: RunPodStateStore, encryption_key: str | bytes) -> None:
        key = encryption_key.encode("ascii") if isinstance(encryption_key, str) else encryption_key
        try:
            self._cipher = Fernet(key)
        except (TypeError, ValueError) as error:
            raise ValueError("Credential encryption key must be a valid Fernet key") from error
        self._state_store = state_store

    async def store(
        self,
        credential_id: str,
        plaintext: str,
        *,
        key_hint: str | None = None,
        validated_at: datetime | None = None,
    ) -> None:
        await self._state_store.save_credential(
            StoredCredential(
                credential_id=credential_id,
                ciphertext=self._cipher.encrypt(plaintext.encode("utf-8")),
                key_hint=key_hint,
                validated_at=validated_at,
            )
        )

    async def retrieve(self, credential_id: str) -> str | None:
        credential = await self._state_store.get_credential(credential_id)
        if credential is None:
            return None
        try:
            return self._cipher.decrypt(credential.ciphertext).decode("utf-8")
        except InvalidToken as error:
            raise WorkspaceError(
                "credential_decryption_failed",
                "The stored provider credential could not be decrypted.",
                status_code=500,
            ) from error

    async def delete(self, credential_id: str) -> None:
        await self._state_store.delete_credential(credential_id)

    async def status(self, credential_id: str) -> CredentialStatus:
        credential = await self._state_store.get_credential(credential_id)
        if credential is None:
            return CredentialStatus(configured=False)
        return CredentialStatus(
            configured=True,
            key_hint=credential.key_hint,
            validated_at=credential.validated_at,
        )
