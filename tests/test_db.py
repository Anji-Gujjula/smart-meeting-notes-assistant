import os

# Must be set before importing db module so tests do not touch a developer DB.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from services import db


def test_save_and_get_meeting():
    db.init_db()
    analysis = {
        "summary": ["a", "b", "c"],
        "action_items": [],
        "decisions": ["ship"],
        "sentiment": "positive",
    }
    meeting_id = db.save_meeting(
        title="Test",
        source_type="txt",
        source_name="meeting.txt",
        transcript="test transcript",
        analysis=analysis,
        chunks=["test transcript"],
        embeddings=[[1.0, 0.0]],
    )
    result = db.get_meeting(meeting_id)
    assert result is not None
    assert result["title"] == "Test"
    assert result["source_name"] == "meeting.txt"
    assert result["analysis"]["decisions"] == ["ship"]
