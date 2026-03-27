"""Database setup with sqlite-vec extension and WAL mode."""

import sqlite_vec
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


def create_memory_engine(db_path: str, **kwargs):
    """Create a SQLAlchemy engine with sqlite-vec and WAL mode.

    Args:
        db_path: Path to the SQLite database file (e.g. "memory.db").
        **kwargs: Extra args passed to create_engine.
    """
    url = f"sqlite:///{db_path}" if not db_path.startswith("sqlite") else db_path
    engine = create_engine(url, connect_args={"check_same_thread": False}, **kwargs)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
        try:
            dbapi_connection.enable_load_extension(True)
            sqlite_vec.load(dbapi_connection)
            dbapi_connection.enable_load_extension(False)
        except AttributeError:
            # macOS system Python disables enable_load_extension for security.
            # Vector search won't work, but core blocks and keyword search still do.
            import logging
            logging.getLogger("amarin_memory.database").warning(
                "sqlite3.enable_load_extension unavailable (macOS system Python?). "
                "Vector search disabled. Use a non-system Python for full functionality."
            )

    return engine


def create_session(engine) -> sessionmaker:
    """Create a sessionmaker bound to the given engine."""
    return sessionmaker(bind=engine)


def get_db(session_factory: sessionmaker):
    """Yield a database session (for use as a dependency)."""
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
