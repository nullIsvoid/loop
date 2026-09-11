#!/usr/bin/env python3
"""Download Mode B I2V artifacts (x3 + stills + run.json; skip full mp4 optional)."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

REMOTE = "/root/latent-loop/artifacts/mode_b_i2v"
LOCAL = Path(__file__).resolve().parents[1] / "artifacts" / "mode_b_i2v"


def _keep(name: str) -> bool:
    lower = name.lower()
    if lower in {"run.json", "result.md", "source.png"}:
        return True
    if lower.endswith(("_x3.mp4", ".png", ".json", ".md", ".jpg", ".jpeg")):
        return True
    # skip full single-loop mp4s by default (A_baseline / B_symmetric)
    if lower in {"a_baseline.mp4", "b_symmetric.mp4"}:
        return False
    if lower.endswith(".mp4"):
        return False
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

    def walk(remote: str, local: Path) -> None:
        local.mkdir(parents=True, exist_ok=True)
        for name in sorted(sftp.listdir(remote)):
            r = f"{remote}/{name}"
            try:
                sftp.listdir(r)
                is_dir = True
            except OSError:
                is_dir = False
            if is_dir:
                walk(r, local / name)
                continue
            if not _keep(name):
                print(f"skip {r}")
                continue
            print(f"get {r}")
            sftp.get(r, str(local / name))

    walk(REMOTE, LOCAL)
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
