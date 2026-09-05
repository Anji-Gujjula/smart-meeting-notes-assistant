from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud exposes root-level secrets through st.secrets.
# Copy them into os.environ before importing services that initialize clients/DBs.
try:
    for _key in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "DATABASE_URL",
    ):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    # No secrets.toml locally is fine; python-dotenv can load .env instead.
    pass

from services.db import (
    add_chat_message,
    delete_meeting,
    get_chat_messages,
    get_chunks,
    get_meeting,
    init_db,
    list_meetings,
    save_meeting,
)
from services.llm_service import MeetingAI, cosine_similarity
from services.text_utils import chunk_text, normalize_transcript

st.set_page_config(
    page_title="Smart Meeting Notes Assistant",
    page_icon="📝",
    layout="wide",
)

init_db()


def get_ai() -> MeetingAI:
    return MeetingAI()


def format_dt(value: datetime | None) -> str:
    if not value:
        return ""
    return value.astimezone().strftime("%d %b %Y, %H:%M")


def show_analysis(analysis: dict) -> None:
    st.subheader("Meeting intelligence")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Summary")
        for item in analysis.get("summary", []):
            st.markdown(f"- {item}")
    with col2:
        st.markdown("#### Sentiment / tone")
        st.info(analysis.get("sentiment", "Not available"))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Action items")
        actions = analysis.get("action_items", [])
        if actions:
            st.dataframe(actions, use_container_width=True, hide_index=True)
        else:
            st.caption("No explicit action items found.")
    with right:
        st.markdown("#### Key decisions")
        decisions = analysis.get("decisions", [])
        if decisions:
            for item in decisions:
                st.markdown(f"- {item}")
        else:
            st.caption("No explicit decisions found.")


def retrieve_for_question(meeting_id: str, question: str, top_k: int = 5):
    chunks = get_chunks(meeting_id)
    if not chunks:
        meeting = get_meeting(meeting_id)
        if not meeting:
            return []
        return [(0, meeting["transcript"], 1.0)]

    ai = get_ai()
    query_embedding = ai.embed([question])[0]
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((chunk["chunk_index"], chunk["text"], score))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


if "selected_meeting_id" not in st.session_state:
    st.session_state.selected_meeting_id = None

with st.sidebar:
    st.title("📝 Meeting Notes")
    st.caption("Analyze, save, revisit, and chat with meeting transcripts.")

    if st.button("＋ New meeting", use_container_width=True):
        st.session_state.selected_meeting_id = None
        st.rerun()

    meetings = list_meetings()
    if meetings:
        st.markdown("### Saved meetings")
        for item in meetings:
            label = f"{item['title']}\n{format_dt(item['created_at'])}"
            if st.button(label, key=f"meeting_{item['id']}", use_container_width=True):
                st.session_state.selected_meeting_id = item["id"]
                st.rerun()
    else:
        st.caption("No saved meetings yet.")

st.title("Smart Meeting Notes Assistant")
st.caption(
    "Generate a concise summary, action items, decisions, sentiment, and ask grounded follow-up questions."
)

selected_id = st.session_state.selected_meeting_id

if not selected_id:
    st.subheader("Add a meeting transcript")

    meeting_title = st.text_input("Meeting title", placeholder="e.g. Product Planning - 5 Sep")
    source = st.radio("Transcript input", ["Paste text", "Upload .txt"], horizontal=True)

    transcript = ""
    source_type = "paste"
    source_name = "Pasted transcript"
    if source == "Paste text":
        transcript = st.text_area(
            "Transcript",
            height=330,
            placeholder="Paste the meeting transcript here...",
        )
    else:
        source_type = "txt"
        uploaded = st.file_uploader(
            "Upload transcript (.txt only)",
            type=["txt"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            source_name = Path(uploaded.name).name
            transcript = uploaded.getvalue().decode("utf-8", errors="replace")
            st.text_area("Preview", value=transcript, height=260, disabled=True)

    with st.expander("Try with sample transcript"):
        st.code(
            """Priya: We need to finalize the beta launch plan.
Sam: I can prepare the rollout checklist by Friday.
Priya: Great. We decided to launch to 10% of users first.
Lee: I will validate the monitoring dashboard before launch.
Sam: The API latency risk is still open; no owner yet.
Priya: Overall, let's proceed with the staged launch.""",
            language="text",
        )

    if st.button("Analyze & save meeting", type="primary", use_container_width=True):
        clean = normalize_transcript(transcript)
        if not meeting_title.strip():
            st.error("Please enter a meeting title.")
        elif not clean:
            st.error("Please provide a meeting transcript.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not configured. Add it to your local .env or Streamlit secrets.")
        else:
            try:
                with st.status("Analyzing meeting...", expanded=True) as status:
                    ai = get_ai()
                    st.write("Extracting summary, action items, decisions, and sentiment")
                    analysis = ai.analyze(clean)

                    st.write("Preparing transcript chunks for follow-up Q&A")
                    chunks = chunk_text(clean)
                    embeddings = ai.embed(chunks)

                    st.write("Saving meeting")
                    meeting_id = save_meeting(
                        title=meeting_title.strip(),
                        source_type=source_type,
                        source_name=source_name,
                        transcript=clean,
                        analysis=analysis,
                        chunks=chunks,
                        embeddings=embeddings,
                    )
                    status.update(label="Meeting saved", state="complete")

                st.session_state.selected_meeting_id = meeting_id
                st.rerun()
            except Exception as exc:
                st.error(f"Could not analyze the meeting: {exc}")

else:
    meeting = get_meeting(selected_id)
    if not meeting:
        st.session_state.selected_meeting_id = None
        st.rerun()

    header_col, action_col = st.columns([5, 1])
    with header_col:
        st.subheader(meeting["title"])
        st.caption(f"Saved {format_dt(meeting['created_at'])}")
    with action_col:
        if st.button("Delete", use_container_width=True):
            delete_meeting(selected_id)
            st.session_state.selected_meeting_id = None
            st.rerun()

    show_analysis(meeting["analysis"])

    with st.expander("View transcript"):
        st.text(meeting["transcript"])

    st.divider()
    st.subheader("Ask this meeting")
    st.caption("Answers are retrieved from this transcript; unsupported information is not guessed.")

    messages = get_chat_messages(selected_id)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about decisions, owners, deadlines, risks, or discussion points...")
    if question:
        question = question.strip()
        if not question:
            st.stop()
        add_chat_message(selected_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching this meeting..."):
                    retrieved = retrieve_for_question(selected_id, question)
                    history = get_chat_messages(selected_id)
                    answer = get_ai().answer_question(
                        meeting_title=meeting["title"],
                        question=question,
                        retrieved_chunks=retrieved,
                        chat_history=history,
                    )
                    st.markdown(answer)
                    # Keep citations user-friendly: show only the original filename.
                    if meeting.get("source_type") == "txt" and meeting.get("source_name"):
                        st.caption(f"Source: {meeting['source_name']}")
            add_chat_message(selected_id, "assistant", answer)
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")
