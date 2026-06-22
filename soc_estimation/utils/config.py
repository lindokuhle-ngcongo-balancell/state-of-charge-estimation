"""
Load run-time configuration from environment variables or a .env file.

Priority order (highest to lowest):
  1. Explicit CLI flags (handled in cli.py — they override these values)
  2. Environment variables (set in shell or in a .env file)
  3. Hard-coded defaults below
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)  # .env values don't override existing env vars
except ImportError:
    pass  # python-dotenv not installed; rely on os.environ only


# ── Athena ────────────────────────────────────────────────────────────────────


@dataclass
class AthenaConfig:
    """AWS Athena connection settings."""

    aws_profile: str | None
    region: str
    database: str
    table: str
    workgroup: str
    s3_output: str | None  # optional — workgroup default is used when None


def get_athena_config() -> AthenaConfig:
    """Read Athena settings from environment variables."""
    return AthenaConfig(
        aws_profile=os.getenv("AWS_PROFILE") or None,
        region=os.getenv("AWS_REGION", "eu-west-1"),
        database=os.getenv("ATHENA_DATABASE", ""),
        table=os.getenv("ATHENA_TABLE", ""),
        workgroup=os.getenv("ATHENA_WORKGROUP", "data_consumers"),
        s3_output=os.getenv("ATHENA_S3_OUTPUT") or None,
    )
