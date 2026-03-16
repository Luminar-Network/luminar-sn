"""
Shared authentication helpers used by both the validator and any tooling
that needs to authenticate against the Luminar backend.
"""

from __future__ import annotations

import bittensor as bt
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from luminar.common.config import settings
from luminar.common.logging import get_logger

log = get_logger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """
    Only retry on network-level errors.
    Never retry on 4xx — those are permanent auth failures (no permit, blacklisted, etc.)
    """
    if isinstance(exc, requests.HTTPError):  # noqa: SIM102
        if exc.response is not None and exc.response.status_code < 500:
            return False  # 4xx = permanent, don't retry
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, requests.HTTPError))


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def fetch_validator_nonce(hotkey_ss58: str) -> dict:
    """
    Call GET /v1/validator/nonce and return the full data payload.
    """
    url = f"{settings.backend_url}/v1/validator/nonce"
    log.debug("Fetching validator nonce from %s", url)

    r = requests.get(
        url,
        params={"hotkey": hotkey_ss58},
        timeout=settings.http_timeout,
    )

    if r.status_code == 403:
        raise requests.HTTPError(
            f"403 Forbidden — hotkey {hotkey_ss58} has no validator permit on this subnet.",
            response=r,
        )

    r.raise_for_status()
    return r.json()["data"]


def sign_message(wallet: bt.Wallet, message: str) -> str:
    """Sign *message* with the wallet's hotkey, return hex-encoded signature."""
    sig: bytes = wallet.hotkey.sign(message.encode())
    return sig.hex()
