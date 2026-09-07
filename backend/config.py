"""Loads database connection settings from the .env file.

Why this module exists as its own thing rather than having db.py read
os.environ directly: keeping all environment-variable access in one place
means the rest of the codebase (db.py, tests, scripts) never has to know
*how* configuration is supplied. If this project later moved from a local
.env file to, say, a secrets manager, only this file would change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project root and copies its
# key=value pairs into os.environ. It's called once, at import time, so
# every other module that imports `config` gets the same loaded environment
# without needing to call this again.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of the Postgres connection parameters.

    Grouping these into one object (instead of passing five loose strings
    around) means every function that needs a DB connection takes one
    `Settings` argument, and tests can construct a fake `Settings` directly
    instead of monkeypatching environment variables.
    """

    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str


def load_settings() -> Settings:
    """Reads the required DB_* environment variables and returns a Settings.

    Raises a clear KeyError-derived message if a variable is missing, rather
    than silently defaulting to something like "localhost" — a silent
    default here could point the whole simulation at the wrong database
    without any error, which is worse than failing loudly at startup.
    """
    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Check your .env file against .env.example."
        )

    return Settings(
        db_host=os.environ["DB_HOST"],
        db_port=int(os.environ["DB_PORT"]),
        db_name=os.environ["DB_NAME"],
        db_user=os.environ["DB_USER"],
        db_password=os.environ["DB_PASSWORD"],
    )
