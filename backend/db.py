import glob
import sqlite3

from flask import g

from .config import Config


def configure_connection(db):
    """Apply SQLite settings that must be enabled for every connection."""
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 5000")
    return db


def get_db():
    """Open or reuse the SQLite database connection for the current request."""
    if "db" not in g:
        g.db = configure_connection(
            sqlite3.connect(Config.DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        )
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(error=None):
    """Close the SQLite database connection when the request ends."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def list_feedback_tables(directory_pattern: str = "*.db") -> dict:
    """List feedback-related tables found in each local *.db file.

    Merged from the former standalone check_db.py utility script.
    """
    report = {}
    for path in glob.glob(directory_pattern):
        con = sqlite3.connect(path)
        try:
            names = [
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE '%feedback%'"
                )
            ]
        finally:
            con.close()
        report[path] = names
    return report


if __name__ == "__main__":
    for db_path, tables in list_feedback_tables().items():
        print(db_path, tables)


def init_db():
    """Create the app tables needed for users, chats, and feedback."""
    db = configure_connection(sqlite3.connect(Config.DATABASE_PATH))
    db.execute("PRAGMA journal_mode = WAL")
    cursor = db.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id              TEXT PRIMARY KEY,
            google_id       TEXT UNIQUE NOT NULL,
            email           TEXT UNIQUE NOT NULL,
            name            TEXT,
            picture_url     TEXT,
            created_at      TEXT NOT NULL
        )
        """
    )

    # FR7: chat_sessions stores one unique conversation container per chat.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            title       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # FR7/FR11: messages are scoped by session_id and store each interaction's
    # user/model text, timestamp, topic label, and response timing metadata.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            topic_label TEXT,
            created_at  TEXT NOT NULL,
            request_started_at  TEXT,
            response_received_at TEXT,
            response_time_ms     INTEGER,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id              TEXT PRIMARY KEY,
            message_id      TEXT UNIQUE,
            session_id      TEXT NOT NULL,
            user_id         TEXT NOT NULL,
            rating          INTEGER,
            correctness     TEXT,
            length_type     TEXT,
            comment         TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (message_id) REFERENCES messages(id),
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    # FR22: analytics_summaries stores one permanent record per chat session.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics_summaries (
            session_id       TEXT PRIMARY KEY,
            user_id          TEXT NOT NULL,
            overall_accuracy REAL,
            avg_rating       REAL,
            correct_count    INTEGER,
            partial_count    INTEGER,
            incorrect_count  INTEGER,
            total_messages   INTEGER,
            topic_breakdown  TEXT NOT NULL DEFAULT '{}',
            phase_breakdown  TEXT NOT NULL DEFAULT '{}',
            created_at       TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )

    db.commit()

    # Remove deprecated feedback stores from existing local databases.
    cursor.execute("DROP TABLE IF EXISTS logout_feedback")
    cursor.execute("DROP TABLE IF EXISTS general_feedback")
    cursor.execute("DROP TABLE IF EXISTS general_feedback_starts")

    # Migrate the old FR22 table, whose random summary IDs and ALL-user
    # aggregates were not tied to an actual chat session.
    summary_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(analytics_summaries)").fetchall()
    }
    summary_fks = {
        (row[2], row[3])
        for row in cursor.execute("PRAGMA foreign_key_list(analytics_summaries)").fetchall()
    }
    required_summary_columns = {"topic_breakdown", "phase_breakdown"}
    required_summary_fks = {("chat_sessions", "session_id"), ("users", "user_id")}
    if (
        not required_summary_columns.issubset(summary_columns)
        or not required_summary_fks.issubset(summary_fks)
        or "id" in summary_columns
    ):
        cursor.execute("ALTER TABLE analytics_summaries RENAME TO analytics_summaries_old")
        cursor.execute(
            """
            CREATE TABLE analytics_summaries (
                session_id       TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                overall_accuracy REAL,
                avg_rating       REAL,
                correct_count    INTEGER,
                partial_count    INTEGER,
                incorrect_count  INTEGER,
                total_messages   INTEGER,
                topic_breakdown  TEXT NOT NULL DEFAULT '{}',
                phase_breakdown  TEXT NOT NULL DEFAULT '{}',
                created_at       TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO analytics_summaries
            (session_id, user_id, overall_accuracy, avg_rating, correct_count,
             partial_count, incorrect_count, total_messages, topic_breakdown,
             phase_breakdown, created_at)
            SELECT old.session_id, old.user_id, old.overall_accuracy, old.avg_rating,
                   old.correct_count, old.partial_count, old.incorrect_count,
                   old.total_messages, '{}', '{}', old.created_at
            FROM analytics_summaries_old old
            JOIN chat_sessions cs ON cs.id = old.session_id
            JOIN users u ON u.id = old.user_id
            """
        )
        cursor.execute("DROP TABLE analytics_summaries_old")
    db.commit()

    # A failed rebuild (e.g. the server was killed mid-migration) can leave a
    # feedback_old table behind, which blocks the ALTER TABLE rename below.
    leftover = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('feedback_old')"
    ).fetchone()
    if leftover:
        cursor.execute("DROP TABLE feedback_old")

    # Existing databases need a table rebuild because SQLite cannot drop a
    # NOT NULL constraint or add a foreign key with ALTER TABLE.
    feedback_fks = {
        (row[2], row[3])
        for row in cursor.execute("PRAGMA foreign_key_list(feedback)").fetchall()
    }
    required_fks = {
        ("message_id", "messages"),
        ("session_id", "chat_sessions"),
        ("user_id", "users"),
    }
    # Rebuild when the FK set or the nullability is wrong for the current
    # schema: message_id/session_id must be nullable (general bot feedback has
    # neither), and message_id must stay unique per message.
    feedback_cols = {
        row[1]: row for row in cursor.execute("PRAGMA table_info(feedback)").fetchall()
    }
    message_col = feedback_cols.get("message_id")
    session_col = feedback_cols.get("session_id")
    nullable_ok = (
        message_col and not message_col[3]
        and session_col and not session_col[3]
    )
    if not required_fks.issubset(feedback_fks) or not nullable_ok:
        cursor.execute("ALTER TABLE feedback RENAME TO feedback_old")
        cursor.execute(
            """
            CREATE TABLE feedback (
                id              TEXT PRIMARY KEY,
                message_id      TEXT UNIQUE,
                session_id      TEXT,
                user_id         TEXT NOT NULL,
                rating          INTEGER,
                correctness     TEXT,
                length_type     TEXT,
                comment         TEXT,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(id),
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO feedback
            (id, message_id, session_id, user_id, rating, correctness,
             length_type, comment, created_at)
            SELECT id, message_id, session_id, user_id, rating, correctness,
                   length_type, comment, created_at
            FROM feedback_old
            """
        )
        cursor.execute("DROP TABLE feedback_old")
    db.commit()

    # Keep old local databases compatible with the current feedback form.
    feedback_columns = [row[1] for row in cursor.execute("PRAGMA table_info(feedback)").fetchall()]
    if "comment" not in feedback_columns:
        cursor.execute("ALTER TABLE feedback ADD COLUMN comment TEXT")
        db.commit()

    # Enforce one feedback row per assistant message at the database level.
    # If a legacy database contains duplicates, retain the newest row first.
    # NULL message_id rows (general bot feedback) are exempt from the dedupe.
    cursor.execute(
        """
        DELETE FROM feedback
        WHERE message_id IS NOT NULL
          AND rowid NOT IN (
            SELECT MAX(rowid)
            FROM feedback
            WHERE message_id IS NOT NULL
            GROUP BY message_id
        )
        """
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_message_id_unique "
        "ON feedback(message_id)"
    )
    db.commit()

    # FR10: keep old message tables compatible with response-time tracking.
    message_columns = [row[1] for row in cursor.execute("PRAGMA table_info(messages)").fetchall()]
    if "request_started_at" not in message_columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN request_started_at TEXT")
    if "response_received_at" not in message_columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN response_received_at TEXT")
    if "response_time_ms" not in message_columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN response_time_ms INTEGER")
    # FR11: topic_label lets every stored user/model interaction be grouped by topic.
    if "topic_label" not in message_columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN topic_label TEXT")
    db.commit()

    db.close()
