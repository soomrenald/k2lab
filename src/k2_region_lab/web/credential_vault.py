from __future__ import annotations

from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken

from k2_region_lab.web.domain import WorkspaceError


class CredentialVault(Protocol):
    async def store(self, credential_id: str, plaintext: str) -> None: ...

    async def retrieve(self, credential_id: str) -> str | None: ...

    async def delete(self, credential_id: str) -> None: ...


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

    async def store(self, credential_id: str, plaintext: str) -> None:
        self._encrypted[credential_id] = self._cipher.encrypt(plaintext.encode("utf-8"))

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
