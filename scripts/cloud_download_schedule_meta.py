#!/usr/bin/env python3
"""Download schedule-compare metadata + PNG previews (no mp4)."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

REMOTE = "/root/latent-loop/artifacts/mode_b_schedules"
LOCAL = Path(__file__).resolve().parents[1] / "artifacts" / "mode_b_schedules"
KEEP_SUFFIXES = {".json", ".png", ".md"}


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
    for name in sftp.listdir(REMOTE):
        if Path(name).suffix.lower() not in KEEP_SUFFIXES:
            print(f"skip {name}")
            continue
        remote = f"{REMOTE}/{name}"
        local = LOCAL / name
        print(f"get {name}")
        sftp.get(remote, str(local))
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
