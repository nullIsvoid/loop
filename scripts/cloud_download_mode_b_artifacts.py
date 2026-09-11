#!/usr/bin/env python3
"""Download cloud artifacts/mode_b_real into local loop repo."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

REMOTE_DIR = "/root/latent-loop/artifacts/mode_b_real"
LOCAL_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "mode_b_real"


def main() -> None:
    password = os.environ.get("CLOUD_SSH_PASS")
    if not password:
        raise SystemExit("CLOUD_SSH_PASS is required")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
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
    for name in sftp.listdir(REMOTE_DIR):
        remote = f"{REMOTE_DIR}/{name}"
        local = LOCAL_DIR / name
        print(f"get {remote} -> {local}")
        sftp.get(remote, str(local))
        print(f"  bytes={local.stat().st_size}")
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
