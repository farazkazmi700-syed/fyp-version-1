import uuid
from datetime import datetime

from flask import Blueprint, jsonify, redirect, url_for

from .db import get_db
from .utils import current_user, current_user_id, json_payload, now_iso, require_login

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/feedback")
@require_login
def feedback_page():
    """Redirect to chat since the feedback page has been removed."""
    return redirect(url_for("chat.chat"))


@feedback_bp.route("/api/feedback", methods=["POST"])
@require_login
def submit_feedback():
    """FR12: save required response feedback before the user can continue chatting."""
    data = json_payload()
    message_id = data.get("message_id")
    rating = data.get("rating")
    correctness = data.get("correctness")
    length_type = data.get("length_type") or data.get("length_rating")
    comment = data.get("comment")

    db = get_db()

    if message_id:
        # FR12: message feedback is mandatory and must include all required fields.
        if rating not in (1, 2, 3, 4):
            return jsonify({"error": "Rating from 1 to 4 is required."}), 400
        if correctness not in ("Correct", "Partial", "Incorrect"):
            return jsonify({"error": "Correctness must be Correct, Partial, or Incorrect."}), 400
        if length_type not in ("Short", "To the Point", "Lengthy"):
            return jsonify({"error": "Length type must be Short, To the Point, or Lengthy."}), 400

        # session_id comes from the message itself; feedback requires only the message.
        message = db.execute(
            """
            SELECT id, session_id FROM messages
            WHERE id = ? AND user_id = ? AND role = 'assistant'
            """,
            (message_id, current_user_id()),
        ).fetchone()

        if not message:
            return jsonify({"error": "Assistant message not found."}), 404

        session_id = message["session_id"]
    else:
        # General bot feedback: no message required; something must be given.
        if rating not in (1, 2, 3, 4) and not (comment or "").strip():
            return jsonify({"error": "Select a rating or write a comment."}), 400
        if correctness is not None and correctness not in ("Correct", "Partial", "Incorrect"):
            return jsonify({"error": "Correctness must be Correct, Partial, or Incorrect."}), 400
        if length_type is not None and length_type not in ("Short", "To the Point", "Lengthy"):
            return jsonify({"error": "Length type must be Short, To the Point, or Lengthy."}), 400
        message_id = None
        session_id = None

    # One feedback per message: the UNIQUE(message_id) constraint and this UPSERT make
    # duplicate submissions safe even when two requests arrive concurrently. General
    # feedback (message_id = NULL) always inserts a new row because UNIQUE allows many
    # NULLs and ON CONFLICT(message_id) does not match NULL keys.
    db.execute(
        """
        INSERT INTO feedback
        (id, message_id, session_id, user_id, rating, correctness, length_type, comment, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(message_id) DO UPDATE SET
            rating = excluded.rating,
            correctness = excluded.correctness,
            length_type = excluded.length_type,
            comment = excluded.comment
        """,
        (str(uuid.uuid4()), message_id, session_id, current_user_id(), rating,
         correctness, length_type, comment, now_iso()),
    )

    db.commit()
    return jsonify({"success": True})


@feedback_bp.route("/feedback/submit", methods=["POST"])
@require_login
def submit_feedback_frontend():
    """Alias endpoint for frontend feedback submission."""
    return submit_feedback()


