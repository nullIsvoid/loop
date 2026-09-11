#!/usr/bin/env python3
"""Download cross-seed metadata + x3 previews (+ first/last png)."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

REMOTE = "/root/latent-loop/artifacts/mode_b_cross_seed"
LOCAL = Path(__file__).resolve().parents[1] / "artifacts" / "mode_b_cross_seed"


def _keep(name: str) -> bool:
    lower = name.lower()
    if lower.endswith("_x3.mp4"):
        return True
    if lower.endswith((".json", ".png", ".md")):
        return True
    if lower == "run.log":
        return True
    return False


def main() -> None:
    password = os.environ.get("CLOUD_SSH_PASS")
    if not password:
        raise SystemExit("CLOUD_SSH_PASS is required")
    LOCAL.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ.get("CLOUD_SSH_HOST", "117.50.214.119"),
        port=int(os.environ.get("CLOUD_SSH_PORT", "23")),
        username=os.environ.get("CLOUD_SSH_USER", "root"),
        password=password,
        timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    sftp = client.open_sftp()
    for name in sorted(sftp.listdir(REMOTE)):
        if not _keep(name):
            print(f"skip {name}")
            continue
        print(f"get {name}")
        sftp.get(f"{REMOTE}/{name}", str(LOCAL / name))
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
