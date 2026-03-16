"""
Entry-point for the Luminar validator node.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import bittensor as bt

# Ensure the project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from luminar.common.config import settings
from luminar.common.logging import get_logger
from luminar.validator.core import ValidatorCore

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Luminar subnet validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)

    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmark/traffic"),
        help="Path to the directory containing benchmark input files.",
    )

    return parser.parse_args()


def _get(args: argparse.Namespace, key: str, default: str = "default") -> str:
    """
    Read a value from the argparse Namespace by its raw string key.
    """
    return vars(args).get(key, default)


def main() -> None:
    args = _parse_args()

    # Validate benchmark directory
    benchmark_dir: Path = args.benchmark_dir.resolve()
    if not benchmark_dir.is_dir():
        log.error(
            "Benchmark directory not found: %s\nEnsure benchmark/classify/ is populated.",
            benchmark_dir,
        )
        sys.exit(1)

    # Pull wallet values using dot-key access
    wallet_name = _get(args, "wallet.name", "default")
    wallet_hotkey = _get(args, "wallet.hotkey", "default")
    wallet_path = _get(args, "wallet.path", "~/.bittensor/wallet")

    # Allow CLI --subtensor.network to override the env var
    subtensor_network = _get(args, "subtensor.network", None)
    if subtensor_network:
        os.environ["BITTENSOR_NETWORK"] = subtensor_network

    wallet_kwargs: dict = {"name": wallet_name, "hotkey": wallet_hotkey}
    if wallet_path:
        wallet_kwargs["path"] = str(Path(wallet_path).expanduser())

    try:
        wallet = bt.Wallet(**wallet_kwargs)
        _ = wallet.hotkey  # triggers keyfile load; raises if not found
    except Exception as exc:
        log.error(
            "Failed to load wallet '%s' / hotkey '%s': %s\n"
            "  Check your wallets with:  btcli wallet list",
            wallet_name,
            wallet_hotkey,
            exc,
        )
        sys.exit(1)

    log.info("Checking validator permit on netuid %d ...", settings.netuid)
    try:
        subtensor = bt.Subtensor(network=settings.network)
        metagraph = subtensor.metagraph(netuid=settings.netuid)
        hotkey = wallet.hotkey.ss58_address

        if hotkey not in metagraph.hotkeys:
            log.error(
                "  Hotkey %s is not registered on netuid %d.\n"
                "  Register first:  btcli subnet register --netuid %d",
                hotkey,
                settings.netuid,
                settings.netuid,
            )
            sys.exit(1)

        uid = metagraph.hotkeys.index(hotkey)
        has_permit = bool(metagraph.validator_permit[uid])

        if not has_permit:
            log.error(
                "  Hotkey %s (uid=%d) does not have a validator permit on netuid %d.\n"
                "  Please make sure you are using validator hotkey.\n",
                hotkey,
                uid,
                settings.netuid,
            )
            sys.exit(1)

        log.info("Validator permit confirmed — uid=%d hotkey=%s", uid, hotkey)

    except SystemExit:
        raise
    except Exception as exc:
        # Don't block startup if the chain is unreachable — just warn.
        log.warning(
            "Could not verify validator permit (chain unreachable?): %s\n"
            "  Proceeding anyway — backend will reject requests if permit is missing.",
            exc,
        )

    log.info(
        "Starting Luminar validator\n"
        "  network  : %s\n"
        "  netuid   : %d\n"
        "  hotkey   : %s\n"
        "  backend  : %s\n"
        "  benchmark: %s",
        settings.network,
        settings.netuid,
        wallet.hotkey.ss58_address,
        settings.backend_url,
        benchmark_dir,
    )

    core = ValidatorCore(wallet=wallet, benchmark_data_dir=benchmark_dir)
    core.start()  # blocks until KeyboardInterrupt


if __name__ == "__main__":
    main()
