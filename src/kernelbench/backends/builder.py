"""Hash-isolated Ascend C project build, private OPP install and binding build."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .artifact import artifact_metadata, write_metadata
from .errors import (
    AscendCCompileError,
    BindingCompileError,
    OPPInstallError,
)
from .generator import TEMPLATE_VERSION, GeneratedNames, generated_names, render_project
from .security import validate_candidate_source
from .spec import (
    AscendCSpec,
    AscendCTilingSpec,
    extract_literal_assignment,
    extract_literal_tiling,
)


_SECRET_ENV = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_WIRE_API",
    "ANTHROPIC_API_KEY",
    "DASHSCOPE_API_KEY",
    "PYRAMIDKERNEL_ENV_FILE",
}


@dataclass(frozen=True)
class AscendCArtifact:
    candidate_hash: str
    root: Path
    metadata: dict[str, Any]
    spec: AscendCSpec
    tiling: AscendCTilingSpec
    names: GeneratedNames
    opp_root: Path
    binding_path: Path | None = None

    @property
    def custom_vendor_root(self) -> Path:
        return self.opp_root / "vendors" / "customize"


def scrubbed_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    for name in _SECRET_ENV:
        env.pop(name, None)
    return env


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        handle.close()


class AscendCBuilder:
    def __init__(
        self,
        cache_root: str | os.PathLike = "cache/ascendc",
        target_soc: str = "Ascend910B3",
        cann_path: str | os.PathLike = "/usr/local/Ascend/cann-9.0.1",
        build_timeout: int = 900,
        build_jobs: int | None = None,
    ):
        self.cache_root = Path(cache_root).resolve()
        self.target_soc = target_soc
        self.cann_path = Path(cann_path).resolve()
        self.build_timeout = build_timeout
        self.build_jobs = build_jobs

    def _candidate_parts(
        self,
        candidate_source: str,
        *,
        level: int | None = None,
        problem_id: int | None = None,
    ) -> tuple[AscendCSpec, AscendCTilingSpec, str, dict[str, Any]]:
        spec = validate_candidate_source(candidate_source)
        if (level is None) != (problem_id is None):
            raise ValueError("level and problem_id must be provided together")
        if level is not None and problem_id is not None:
            from .ascendc_tasks import get_ascend_task_contract, validate_spec_for_task

            validate_spec_for_task(
                spec,
                get_ascend_task_contract(level, problem_id),
            )
        tiling = extract_literal_tiling(candidate_source, spec)
        kernel_source = str(extract_literal_assignment(candidate_source, "ASCENDC_KERNEL_SOURCE"))
        compiler_options = {
            "target_soc": self.target_soc,
            "cann_path": str(self.cann_path),
            "build_jobs": self.build_jobs,
        }
        metadata = artifact_metadata(
            candidate_source,
            spec.to_json(),
            self.target_soc,
            self.cache_root,
            template_version=TEMPLATE_VERSION,
            compiler_options=compiler_options,
        )
        return spec, tiling, kernel_source, metadata

    def prepare(
        self,
        candidate_source: str,
        *,
        level: int | None = None,
        problem_id: int | None = None,
    ) -> AscendCArtifact:
        spec, tiling, kernel_source, metadata = self._candidate_parts(
            candidate_source,
            level=level,
            problem_id=problem_id,
        )
        root = self.cache_root / metadata["candidate_hash"]
        names = generated_names(spec, metadata["candidate_hash"])
        for directory in ("source", "project", "opp", "binding", "logs"):
            (root / directory).mkdir(parents=True, exist_ok=True)

        (root / "source" / "candidate.py").write_text(candidate_source, encoding="utf-8")
        (root / "source" / "spec.json").write_text(spec.to_json() + "\n", encoding="utf-8")
        (root / "source" / "tiling.json").write_text(
            json.dumps(tiling.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "source" / "kernel.cpp").write_text(kernel_source, encoding="utf-8")
        metadata.update(
            {
                "op_type": names.op_type,
                "op_file": names.op_file,
                "kernel_symbol": names.kernel_symbol,
                "binding_module": names.binding_module,
                "status": "prepared",
            }
        )
        write_metadata(root / "artifact.json", metadata)
        return AscendCArtifact(
            metadata["candidate_hash"],
            root,
            metadata,
            spec,
            tiling,
            names,
            root / "opp",
        )

    def _run(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        error_type: type[Exception],
    ) -> float:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.build_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            log_path.write_text(output, encoding="utf-8")
            raise error_type(f"command timed out after {self.build_timeout}s: {' '.join(command)}") from exc
        elapsed = time.perf_counter() - started
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-40:])
            raise error_type(
                f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}"
            )
        return elapsed

    def _prepare_compile_opp(self, root: Path) -> Path:
        """Create a readable private OPP view for the CANN compiler."""
        system_opp = self.cann_path / "opp"
        compile_opp = root / "compile_opp"
        vendors = compile_opp / "vendors"
        vendors.mkdir(parents=True, exist_ok=True)
        (vendors / "config.ini").write_text("load_priority=customize\n", encoding="utf-8")
        for name in (
            "Ascend",
            "bin",
            "built-in",
            "include",
            "lib64",
            "script",
            "test-ops",
            "scene.info",
            "version.info",
        ):
            target = system_opp / name
            link = compile_opp / name
            if target.exists() and not link.exists():
                link.symlink_to(target, target_is_directory=target.is_dir())
        return compile_opp

    def _cached_artifact(self, artifact: AscendCArtifact) -> AscendCArtifact | None:
        success = artifact.root / "success.json"
        if not success.is_file():
            return None
        state = json.loads(success.read_text(encoding="utf-8"))
        binding = Path(state["binding_path"])
        vendor = artifact.custom_vendor_root
        if not binding.is_file() or not (vendor / "op_api" / "lib" / "libcust_opapi.so").is_file():
            return None
        metadata = dict(artifact.metadata)
        metadata.update(state)
        metadata["cache_hit"] = True
        return AscendCArtifact(
            artifact.candidate_hash,
            artifact.root,
            metadata,
            artifact.spec,
            artifact.tiling,
            artifact.names,
            artifact.opp_root,
            binding,
        )

    def build(
        self,
        candidate_source: str,
        *,
        level: int | None = None,
        problem_id: int | None = None,
    ) -> AscendCArtifact:
        artifact = self.prepare(
            candidate_source,
            level=level,
            problem_id=problem_id,
        )
        with _file_lock(artifact.root / ".build.lock"):
            cached = self._cached_artifact(artifact)
            if cached is not None:
                return cached

            kernel_source = (artifact.root / "source" / "kernel.cpp").read_text(encoding="utf-8")
            project = artifact.root / "project"
            render_project(
                artifact.root,
                artifact.spec,
                artifact.tiling,
                kernel_source,
                artifact.names,
                self.cann_path,
            )

            env = scrubbed_environment()
            env["ASCEND_HOME_PATH"] = str(self.cann_path)
            env["ASCEND_OPP_PATH"] = str(self._prepare_compile_opp(artifact.root))
            if self.build_jobs is not None:
                env["KERNELBENCH_BUILD_JOBS"] = str(self.build_jobs)
                env["MAX_JOBS"] = str(self.build_jobs)

            timings: dict[str, float] = {}
            timings["compile_seconds"] = self._run(
                ["bash", "./build.sh"],
                cwd=project,
                env=env,
                log_path=artifact.root / "logs" / "compile.log",
                error_type=AscendCCompileError,
            )
            package = project / "build_out" / "custom_opp_ubuntu_aarch64.run"
            if not package.is_file():
                raise AscendCCompileError(f"CANN package was not generated: {package}")

            if artifact.opp_root.exists():
                shutil.rmtree(artifact.opp_root)
            artifact.opp_root.mkdir(parents=True)
            timings["opp_install_seconds"] = self._run(
                [str(package), "--quiet", f"--install-path={artifact.opp_root}"],
                cwd=project,
                env=env,
                log_path=artifact.root / "logs" / "opp_install.log",
                error_type=OPPInstallError,
            )

            vendor = artifact.custom_vendor_root
            binding_env = dict(env)
            binding_env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
            binding_env["LD_LIBRARY_PATH"] = os.pathsep.join(
                [
                    str(vendor / "op_api" / "lib"),
                    str(self.cann_path / "lib64"),
                    env.get("LD_LIBRARY_PATH", ""),
                ]
            ).rstrip(os.pathsep)
            timings["binding_compile_seconds"] = self._run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=artifact.root / "binding",
                env=binding_env,
                log_path=artifact.root / "logs" / "binding_compile.log",
                error_type=BindingCompileError,
            )
            matches = sorted(
                (artifact.root / "binding").glob(f"{artifact.names.binding_module}*.so")
            )
            if len(matches) != 1:
                raise BindingCompileError(
                    f"expected one binding shared object, found {len(matches)}"
                )
            binding_path = matches[0].resolve()
            state = {
                "status": "built",
                "cache_hit": False,
                "binding_path": str(binding_path),
                "opp_root": str(artifact.opp_root),
                "custom_vendor_root": str(vendor),
                "timings": timings,
                "system_opp_modified": False,
                "candidate_source_sha256": hashlib.sha256(
                    candidate_source.encode("utf-8")
                ).hexdigest(),
            }
            (artifact.root / "success.json").write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            metadata = dict(artifact.metadata)
            metadata.update(state)
            write_metadata(artifact.root / "artifact.json", metadata)
            return AscendCArtifact(
                artifact.candidate_hash,
                artifact.root,
                metadata,
                artifact.spec,
                artifact.tiling,
                artifact.names,
                artifact.opp_root,
                binding_path,
            )
