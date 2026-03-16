"""
Orchestrates the two-phase Docker sandbox evaluation of a miner agent.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003
from typing import Any

import docker
import docker.errors
from docker.models.containers import Container  # noqa: TC002

from luminar.common.config import settings
from luminar.common.logging import get_logger

log = get_logger(__name__)


@dataclass
class SandboxResult:
    success: bool
    output_csv_bytes: bytes | None = None
    stderr: str = ""
    exit_code: int | None = None


class SandboxRunner:
    """
    Manages the full two-phase sandbox lifecycle for a single agent evaluation.
    """

    def __init__(self, benchmark_data_dir: Path) -> None:
        self._benchmark_dir = benchmark_data_dir
        self._client = docker.from_env()

    def run(self, agent_bytes: bytes) -> SandboxResult:
        """
        Execute both sandbox phases for the given agent source code.
        """
        run_id = uuid.uuid4().hex[:12]
        work_dir = settings.sandbox_workdir / run_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._run_in_workdir(agent_bytes, work_dir, run_id)
        finally:
            # Always clean up temp files
            shutil.rmtree(work_dir, ignore_errors=True)
            log.debug("Cleaned up sandbox workdir: %s", work_dir)

    # Private implementation

    def _run_in_workdir(
        self,
        agent_bytes: bytes,
        work_dir: Path,
        run_id: str,
    ) -> SandboxResult:
        # Prepare host directories
        agent_path = work_dir / "agent.py"
        cache_dir = work_dir / "cache"
        output_dir = work_dir / "output"

        agent_path.write_bytes(agent_bytes)
        cache_dir.mkdir()
        output_dir.mkdir()

        # Phase 1: Setup (network ON)
        log.info("[%s] Phase 1: Setup — downloading models", run_id)
        setup_result = self._run_container(
            run_id=run_id,
            phase="setup",
            agent_path=agent_path,
            cache_dir=cache_dir,
            output_dir=None,
            network_mode="host",
            timeout=settings.setup_timeout,
            cmd_flag="--setup",
        )

        if not setup_result["success"]:
            log.warning("[%s] Setup phase failed: %s", run_id, setup_result["stderr"][:500])
            return SandboxResult(
                success=False,
                stderr=setup_result["stderr"],
                exit_code=setup_result["exit_code"],
            )

        # Phase 2: Infer (network OFF)
        log.info("[%s] Phase 2: Infer — running inference", run_id)
        infer_result = self._run_container(
            run_id=run_id,
            phase="infer",
            agent_path=agent_path,
            cache_dir=cache_dir,
            output_dir=output_dir,
            network_mode="none",
            timeout=settings.infer_timeout,
            cmd_flag="--infer",
        )

        output_csv = output_dir / "output.csv"

        if not infer_result["success"]:
            log.warning("[%s] Infer phase failed: %s", run_id, infer_result["stderr"][:500])
            return SandboxResult(
                success=False,
                stderr=infer_result["stderr"],
                exit_code=infer_result["exit_code"],
            )

        if not output_csv.exists():
            log.warning("[%s] Infer phase succeeded but output.csv not found.", run_id)
            return SandboxResult(
                success=False,
                stderr="output.csv not produced",
                exit_code=infer_result["exit_code"],
            )

        # Read output bytes NOW — workdir is cleaned up in the finally block
        output_csv_bytes = output_csv.read_bytes()
        log.info("[%s] Sandbox evaluation completed successfully.", run_id)
        return SandboxResult(
            success=True,
            output_csv_bytes=output_csv_bytes,
            stderr=infer_result["stderr"],
            exit_code=infer_result["exit_code"],
        )

    def _run_container(
        self,
        *,
        run_id: str,
        phase: str,
        agent_path: Path,
        cache_dir: Path,
        output_dir: Path | None,
        network_mode: str,
        timeout: int,
        cmd_flag: str,
    ) -> dict[str, Any]:
        """
        Spin up one Docker container, wait for completion, return result dict.
        """
        volumes = {
            str(agent_path): {
                "bind": "/agent/agent.py",
                "mode": "ro",
            },
            str(cache_dir): {
                "bind": "/cache",
                "mode": "rw",
            },
        }

        if output_dir is not None:
            volumes[str(self._benchmark_dir)] = {"bind": "/data/input", "mode": "ro"}
            volumes[str(output_dir)] = {"bind": "/data/output", "mode": "rw"}

        container_name = f"luminar-{phase}-{run_id}"

        # Request GPU only if the host has nvidia-container-toolkit installed.
        # Validators without a GPU (or during testing) run CPU-only.
        device_requests = []
        try:
            self._client.info()
            runtimes = self._client.info().get("Runtimes", {})
            if "nvidia" in runtimes:
                device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        except Exception:
            pass

        container: Container | None = None
        try:
            container = self._client.containers.run(
                image=settings.sandbox_image,
                command=f"python /agent/agent.py {cmd_flag}",
                name=container_name,
                detach=True,
                remove=False,
                network_mode=network_mode,
                mem_limit=settings.memory_limit,
                device_requests=device_requests,
                volumes=volumes,
            )

            # Block until done or timeout
            try:
                result = container.wait(timeout=timeout)
                exit_code: int = result.get("StatusCode", -1)
            except Exception:
                log.warning(
                    "[%s] Container %s timed out after %ds", run_id, container_name, timeout
                )
                try:  # noqa: SIM105
                    container.kill()
                except Exception:
                    pass
                return {"success": False, "stderr": "Timed out", "exit_code": -1}

            logs = container.logs(stderr=True, stdout=True).decode(errors="replace")
            success = exit_code == 0

            if not success:
                log.debug("[%s] Container exited %d:\n%s", run_id, exit_code, logs[-2000:])

            return {"success": success, "stderr": logs, "exit_code": exit_code}

        except docker.errors.ImageNotFound:
            log.error(
                "Sandbox image '%s' not found.  Build it first: "
                "docker build -f Dockerfile.sandbox -t %s .",
                settings.sandbox_image,
                settings.sandbox_image,
            )
            return {"success": False, "stderr": "Image not found", "exit_code": -1}

        except docker.errors.APIError as exc:
            log.error("[%s] Docker API error: %s", run_id, exc)
            return {"success": False, "stderr": str(exc), "exit_code": -1}

        finally:
            if container is not None:
                try:  # noqa: SIM105
                    container.remove(force=True)
                except Exception:
                    pass
