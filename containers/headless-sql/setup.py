"""Explicit anonymous operator setup; query execution never invokes this program."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import cast

_BASE = "python@sha256:c845af9399020c7e562969a13689e929074a10fd057acd1b1fad06a2fb068e97"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one pinned local SQL worker installation.")
    parser.add_argument("--docker", required=True)
    parser.add_argument("--engine-socket", required=True)
    parser.add_argument("--artifact-cache")
    args = parser.parse_args()
    docker = cast("str", args.docker)
    socket = cast("str", args.engine_socket)
    if not Path(docker).is_absolute() or not Path(socket).is_absolute():
        raise RuntimeError("sql_setup_invalid")
    root = Path(__file__).resolve().parents[2]
    directory = Path(tempfile.mkdtemp(prefix="strata-sql-setup-"))
    config = directory / "docker-config"
    config.mkdir(mode=0o700)
    sources = root / "src/strata_forge/datasets"
    for name in ("_sql_worker.py", "_sql_wire.py"):
        raw = (sources / name).read_bytes()
        if len(raw) > 131072:
            raise RuntimeError("sql_setup_invalid")
        (directory / name).write_bytes(raw)
    descriptor = importlib.util.spec_from_file_location(
        "_strata_sql_setup_wire", directory / "_sql_wire.py"
    )
    if descriptor is None or descriptor.loader is None:
        raise RuntimeError("sql_setup_invalid")
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    bundle = cast(
        "str",
        module.source_bundle(
            {name: (directory / name).read_bytes() for name in ("_sql_worker.py", "_sql_wire.py")}
        ),
    )
    profile = json.loads((Path(__file__).parent / "profile.json").read_bytes())
    if profile["base_image"] != _BASE or profile["platform"] != "linux/arm64":
        raise RuntimeError("sql_setup_invalid")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cache = Path(args.artifact_cache) if args.artifact_cache else None
    for artifact in profile["artifacts"]:
        name, size, digest, url = (
            artifact["filename"],
            artifact["bytes"],
            artifact["sha256"],
            artifact["url"],
        )
        if (
            type(name) is not str
            or Path(name).name != name
            or type(size) is not int
            or not 0 < size <= 33554432
        ):
            raise RuntimeError("sql_setup_invalid")
        parts = urllib.parse.urlsplit(url)
        if (
            parts.scheme != "https"
            or parts.netloc != "files.pythonhosted.org"
            or parts.query
            or parts.fragment
        ):
            raise RuntimeError("sql_setup_invalid")
        candidate = cache / name if cache is not None else None
        if candidate is not None and candidate.is_file():
            with candidate.open("rb") as source:
                body = source.read(size + 1)
        else:
            with opener.open(url, timeout=30) as response:
                body = response.read(size + 1)
        if len(body) != size or hashlib.sha256(body).hexdigest() != digest:
            raise RuntimeError("sql_setup_artifact_mismatch")
        (directory / name).write_bytes(body)
    shutil.copyfile(Path(__file__).parent / "Dockerfile", directory / "Dockerfile")
    command = [
        docker,
        "--host",
        "unix://" + urllib.parse.quote(socket, safe="/"),
        "--config",
        str(config),
    ]
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin"}
    tag = "strata-sql-owned:" + uuid.uuid4().hex
    with (directory / "setup.log").open("wb") as output:
        subprocess.run(  # noqa: S603 — explicit trusted operator executable; no query input or shell.
            [*command, "pull", "--platform", "linux/arm64", _BASE],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
        subprocess.run(  # noqa: S603 — explicit trusted operator executable; no query input or shell.
            [
                *command,
                "build",
                "--pull=false",
                "--platform",
                "linux/arm64",
                "--network",
                "none",
                "--build-arg",
                "WORKER_BUNDLE=" + bundle,
                "--tag",
                tag,
                str(directory),
            ],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    inspected = subprocess.run(  # noqa: S603 — fixed setup inspection on the explicit local engine.
        [*command, "image", "inspect", tag, "--format", "{{.Id}}"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=10,
    )
    image = inspected.stdout.decode("ascii").strip()
    if (
        len(image) != 71
        or not image.startswith("sha256:")
        or any(c not in "0123456789abcdef" for c in image[7:])
    ):
        raise RuntimeError("sql_setup_invalid")
    installation = {
        "version": 1,
        "profile": "sql_local_arm64_v1",
        "docker_executable": docker,
        "engine_socket": socket,
        "image_id": image,
        "worker_bundle_sha256": bundle,
    }
    (directory / "installation.json").write_text(json.dumps(installation, indent=2) + "\n")
    (directory / "setup-manifest.json").write_text(
        json.dumps(
            {
                "base_image": _BASE,
                "image_id": image,
                "bundle_sha256": bundle,
                "profile": profile,
                "source_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("_sql_worker.py", "_sql_wire.py", "Dockerfile")
                },
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "installation.json").chmod(0o600)
    print(directory / "installation.json")


if __name__ == "__main__":
    main()
