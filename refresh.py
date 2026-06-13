#!/usr/bin/env python3
"""
Kiro Refresh Token → Access Token
==================================
Input: refresh_token
Output: access_token, expires_at, profile_arn, new refresh_token (jika dirotasi)
"""

import hashlib
import getpass
import json
import os
import socket
import sys

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    sys.exit(1)

REGION = os.environ.get("KIRO_REGION", "us-east-1")
REFRESH_URL = f"https://prod.{REGION}.auth.desktop.kiro.dev/refreshToken"

_FINGERPRINT = hashlib.sha256(
    f"{socket.gethostname()}-{getpass.getuser()}-kiro-gateway".encode()
).hexdigest()


def refresh(token: str) -> dict:
    resp = requests.post(
        REFRESH_URL,
        json={"refreshToken": token},
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"KiroIDE-2.0.0-{_FINGERPRINT}",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


if name == "__main__":
    rt = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KIRO_REFRESH_TOKEN")
    if not rt:
        rt = getpass.getpass("Refresh token: ").strip()
    if not rt:
        print("No token provided.", file=sys.stderr)
        sys.exit(1)

    data = refresh(rt)
    print(json.dumps({
        "accessToken": data.get("accessToken"),
        "expiresAt": data.get("expiresAt"),
        "refreshToken": data.get("refreshToken"),
        "profileArn": data.get("profileArn"),
    }, indent=2))