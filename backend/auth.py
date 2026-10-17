"""Local accounts and revocable sessions. No external identity provider."""
from contextlib import closing
import hashlib
import re
import secrets
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

SESSION_SECONDS = 8 * 60 * 60


class LoginData(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().casefold()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Укажите корректный email")
        return value


class RegisterData(LoginData):
    name: str = Field(min_length=2, max_length=80)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Укажите имя, минимум 2 символа")
        return value


def password_hash(password: str, salt: str) -> str:
    return hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt), n=2**15, r=8, p=3,
        maxmem=64 * 1024 * 1024, dklen=64,
    ).hex()


class AuthStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
            """)
        path.chmod(0o600)
        self.dummy_salt = secrets.token_hex(16)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def register(self, data: RegisterData) -> dict:
        salt = secrets.token_hex(16)
        user = {"id": uuid4().hex, "email": data.email, "name": data.name}
        digest = password_hash(data.password, salt)
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                (user["id"], user["email"], user["name"], salt, digest),
            )
        return user

    def authenticate(self, data: LoginData) -> dict | None:
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM users WHERE email=?", (data.email,)).fetchone()
        # Unknown accounts still run the same password derivation.
        digest = password_hash(data.password, row["salt"] if row else self.dummy_salt)
        if row and secrets.compare_digest(digest, row["password_hash"]):
            return {key: row[key] for key in ("id", "email", "name")}
        return None

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM sessions WHERE expires_at<=?", (int(time.time()),))
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (hashlib.sha256(token.encode()).hexdigest(), user_id, int(time.time()) + SESSION_SECONDS),
            )
        return token

    def user_for_session(self, token: str | None) -> dict | None:
        if not token or len(token) > 128:
            return None
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT u.id, u.email, u.name FROM sessions s JOIN users u "
                "ON u.id=s.user_id WHERE s.token_hash=? AND s.expires_at>?",
                (hashlib.sha256(token.encode()).hexdigest(), int(time.time())),
            ).fetchone()
        return dict(row) if row else None

    def revoke(self, token: str | None):
        if token:
            with closing(self.connect()) as db, db:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
