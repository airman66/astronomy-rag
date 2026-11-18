"""Command entry point for python -m app.storage."""

from app.catalog.storage import rebuild_database

if __name__ == "__main__":
    rebuild_database()
