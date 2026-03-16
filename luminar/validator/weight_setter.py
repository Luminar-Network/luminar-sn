"""
On-chain weight assignment for the Luminar subnet.
Supports normal best-agent mode and 100% burn mode (weights always to UID 0).
"""

from __future__ import annotations

import threading

import bittensor as bt
from tenacity import retry, stop_after_attempt, wait_exponential

from luminar.common.config import settings
from luminar.common.logging import get_logger

log = get_logger(__name__)

_SET_WEIGHTS_RETRY = {
    "stop": stop_after_attempt(5),
    "wait": wait_exponential(multiplier=2, min=4, max=60),
    "reraise": True,
}


class WeightMonitor(threading.Thread):
    """
    Background daemon thread that logs on-chain weights every
    LUMINAR_WEIGHT_MONITOR_INTERVAL seconds.
    """

    daemon = True

    def __init__(
        self,
        wallet: bt.Wallet,
        subtensor: bt.Subtensor,
        metagraph: bt.Metagraph,
        interval: int | None = None,
    ) -> None:
        super().__init__(name="WeightMonitor")
        self._wallet = wallet
        self._subtensor = subtensor
        self._metagraph = metagraph
        self._interval = interval or settings.weight_monitor_interval
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        log.info("WeightMonitor started (reporting every %ds)", self._interval)
        # Wait one full interval before first report — avoid noise at startup
        self._stop_event.wait(timeout=self._interval)

        while not self._stop_event.is_set():
            try:
                self._report()
            except Exception as exc:  # noqa: BLE001
                log.warning("WeightMonitor error: %s", exc)
            self._stop_event.wait(timeout=self._interval)

        log.info("WeightMonitor stopped.")

    def _report(self) -> None:
        """Fetch and log current on-chain weights set by this validator."""
        try:
            self._metagraph.sync(subtensor=self._subtensor)
        except Exception as exc:  # noqa: BLE001
            log.warning("WeightMonitor: metagraph sync failed: %s", exc)
            return

        hotkey = self._wallet.hotkey.ss58_address
        if hotkey not in self._metagraph.hotkeys:
            log.warning("WeightMonitor: validator hotkey not in metagraph.")
            return

        validator_uid = self._metagraph.hotkeys.index(hotkey)
        all_uids = list(self._metagraph.uids)
        weights_matrix = self._metagraph.weights

        if weights_matrix is None or len(weights_matrix) == 0:
            log.info(
                "── Weight Monitor (validator uid=%d) ──\n"
                "  No weights set on-chain yet — waiting for first weight-set event.",
                validator_uid,
            )
            return

        if validator_uid >= len(weights_matrix):
            log.info(
                "── Weight Monitor (validator uid=%d) ──\n"
                "  Weights matrix has %d rows — this validator has not set weights yet.",
                validator_uid,
                len(weights_matrix),
            )
            return

        weights = [float(w) for w in weights_matrix[validator_uid]]

        if len(weights) != len(all_uids):
            weights = (weights + [0.0] * len(all_uids))[: len(all_uids)]

        nonzero = [
            (int(uid), round(w, 4)) for uid, w in zip(all_uids, weights, strict=False) if w > 0.0
        ]
        max_w = max(weights) if weights else 0.0

        log.info(
            "── Weight Monitor (validator uid=%d) ──\n  Total UIDs : %d\n  Non-zero   : %s",
            validator_uid,
            len(all_uids),
            nonzero if nonzero else "none yet — weights not set yet",
        )
        log.debug("UIDs array:     %s", all_uids)
        log.debug("Weights array:  %s", weights)


class WeightSetter:
    """Manages on-chain weight setting. Supports normal best-agent mode and 100% BURN mode."""

    def __init__(
        self,
        wallet: bt.Wallet,
        subtensor: bt.Subtensor,
        metagraph: bt.Metagraph,
    ) -> None:
        self._wallet = wallet
        self._subtensor = subtensor
        self._metagraph = metagraph
        self._last_set_block: int = 0
        self._monitor = WeightMonitor(wallet, subtensor, metagraph)

    def start_monitor(self) -> None:
        self._monitor.start()

    def stop_monitor(self) -> None:
        self._monitor.stop()

    def maybe_set_weights(self, best_hotkey: str | None = None) -> bool:
        current_block = self._current_block()
        blocks_since = current_block - self._last_set_block

        if settings.burn_mode:
            log.info(
                "BURN MODE active — forcing weights to UID 0 (100%% burn) "
                "at block %d (interval check skipped)",
                current_block,
            )
            return self._set_weights_for_burn(current_block)

        # Normal mode: best-agent logic
        if not best_hotkey:
            best_meta = self._cache.best_meta if hasattr(self, "_cache") else None
            best_hotkey = best_meta.hotkey if best_meta else None

        if not best_hotkey:
            log.info("No best hotkey available — skipping normal weight set.")
            return False

        log.info(
            "Normal weight set check — best_hotkey=%s  current_block=%d  "
            "last_set_block=%d  blocks_since=%d  interval=%d",
            best_hotkey,
            current_block,
            self._last_set_block,
            blocks_since,
            settings.weight_interval_blocks,
        )

        if blocks_since < settings.weight_interval_blocks:
            log.info(
                "Weight set gated: %d/%d blocks elapsed — will set in ~%d more blocks.",
                blocks_since,
                settings.weight_interval_blocks,
                settings.weight_interval_blocks - blocks_since,
            )
            return False

        log.info("Weight set interval cleared — proceeding to set weights (normal mode).")
        return self._set_weights(best_hotkey, current_block)

    def _current_block(self) -> int:
        try:
            return self._subtensor.get_current_block()
        except Exception as exc:
            log.warning("Could not fetch current block: %s", exc)
            return self._last_set_block

    @retry(**_SET_WEIGHTS_RETRY)
    def _set_weights(self, best_hotkey: str, current_block: int) -> bool:
        try:
            self._metagraph.sync(subtensor=self._subtensor)
        except Exception as exc:
            log.warning("Metagraph sync failed: %s — proceeding with stale data.", exc)

        uid_map: dict[str, int] = {
            self._metagraph.hotkeys[i]: int(self._metagraph.uids[i])
            for i in range(len(self._metagraph.uids))
        }

        if best_hotkey not in uid_map:
            log.warning(
                "Best hotkey %s not found in metagraph — may have deregistered. "
                "Skipping weight set.",
                best_hotkey,
            )
            return False

        best_uid = uid_map[best_hotkey]
        all_uids = list(self._metagraph.uids)
        weights = [0.0] * len(all_uids)
        weights[all_uids.index(best_uid)] = 1.0

        lines = [
            f"  uid={uid}  weight={w:.1f}{'  ← BEST' if uid == best_uid else ''}"
            for uid, w in zip(all_uids, weights, strict=False)
        ]
        log.info(
            "Setting weights (normal mode) at block %d — best UID=%d hotkey=%s\n  Total UIDs: %d\n%s",
            current_block,
            best_uid,
            best_hotkey,
            len(all_uids),
            "\n".join(lines),
        )

        success, message = self._subtensor.set_weights(
            netuid=settings.netuid,
            wallet=self._wallet,
            uids=all_uids,
            weights=weights,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        if success:
            self._last_set_block = current_block
            log.info("Weights set successfully at block %d.", current_block)
        else:
            log.error("set_weights failed: %s", message)
            raise RuntimeError(f"set_weights failed: {message}")

        return success

    @retry(**_SET_WEIGHTS_RETRY)
    def _set_weights_for_burn(self, current_block: int) -> bool:
        """Force 1.0 weight to UID 0 (100% burn)."""
        try:
            self._metagraph.sync(subtensor=self._subtensor)
        except Exception as exc:
            log.warning("Metagraph sync failed during burn: %s", exc)

        all_uids = list(self._metagraph.uids)
        if not all_uids or 0 not in set(all_uids):
            log.warning("UID 0 not present in metagraph — cannot perform burn weight set.")
            return False

        weights = [0.0] * len(all_uids)
        weights[all_uids.index(0)] = 1.0

        log.info(
            "BURN MODE: Setting weights at block %d — UID 0 = 1.0  (all others 0.0)",
            current_block,
        )

        success, message = self._subtensor.set_weights(
            netuid=settings.netuid,
            wallet=self._wallet,
            uids=all_uids,
            weights=weights,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        if success:
            self._last_set_block = current_block
            log.info("Burn weights set successfully at block %d.", current_block)
        else:
            log.error("Burn set_weights failed: %s", message)
            raise RuntimeError(f"set_weights failed: {message}")

        return success
