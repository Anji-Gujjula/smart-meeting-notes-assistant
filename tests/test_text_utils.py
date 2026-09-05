from services.text_utils import chunk_text, normalize_transcript


def test_normalize_transcript():
    value = "Hello   team\r\n\r\n\r\nNext line"
    assert normalize_transcript(value) == "Hello team\n\nNext line"


def test_chunk_text_splits_long_text():
    text = "\n\n".join(["A" * 1000, "B" * 1000, "C" * 1000])
    chunks = chunk_text(text, chunk_size=1500, overlap=100)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
