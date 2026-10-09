"""
Multi-agent newsletter summarizer using LangGraph + Gmail.

Fetches today's newsletters from Gmail, summarizes them with a
3-agent pipeline (Parser → Summarizer → Editor), and sends the
digest back to your inbox.
"""

import os
import subprocess
import sys
import base64
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import TypedDict

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()


def log(msg: str):
    """Print a message prefixed with a local timestamp (for cron.log)."""
    ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{ts}] {msg}", flush=True)


# Gmail API scopes — read and send
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# Add the sender addresses of newsletters you want to summarize
NEWSLETTER_SENDERS = [
    "techbrew@morningbrew.com",
    "crew@morningbrew.com",
    "dan@tldrnewsletter.com"
    # "newsletter@example.com",
    # "digest@techcrunch.com",
]


# -- Gmail helpers --

def notify_token_expired():
    """Send a macOS notification that Gmail token needs re-auth."""
    subprocess.run([
        "osascript", "-e",
        'display notification "Run newsletter_summarizer.py manually to re-authorize Gmail." '
        'with title "Newsletter Digest" subtitle "Gmail token expired"'
    ])


def get_gmail_service():
    """Authenticate and return a Gmail API service instance."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        # Try a silent refresh first if we have a refresh token.
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                log(f"[Auth] Token refresh failed: {e}")
                creds = None  # fall through to interactive re-authorization

        # Refresh failed or no usable token: re-authorize interactively.
        if not creds or not creds.valid:
            if not sys.stdin.isatty():
                notify_token_expired()
                log("[Auth] Could not refresh token and no interactive terminal "
                    "available. Run this script manually in a terminal to re-authorize.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def fetch_todays_newsletters(service) -> list[dict]:
    """Fetch today's newsletter emails from specified senders."""
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    sender_query = " OR ".join(f"from:{addr}" for addr in NEWSLETTER_SENDERS)
    query = f"({sender_query}) after:{today}"

    results = service.users().messages().list(userId="me", q=query).execute()
    messages = results.get("messages", [])

    newsletters = []
    for msg in messages:
        full = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
        headers = {h["name"]: h["value"] for h in full["payload"]["headers"]}
        body = extract_body(full["payload"])
        newsletters.append({
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", "unknown"),
            "body": body,
        })

    return newsletters


def extract_body(payload) -> str:
    """Extract plain text from a Gmail message payload."""
    if payload.get("body", {}).get("data"):
        raw = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        if payload.get("mimeType") == "text/html":
            return BeautifulSoup(raw, "html.parser").get_text(separator="\n", strip=True)
        return raw

    parts = payload.get("parts", [])
    # Prefer plain text, fall back to HTML
    for preferred in ["text/plain", "text/html"]:
        for part in parts:
            if part.get("mimeType") == preferred and part.get("body", {}).get("data"):
                raw = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                if preferred == "text/html":
                    return BeautifulSoup(raw, "html.parser").get_text(separator="\n", strip=True)
                return raw
            # Handle nested multipart
            if part.get("parts"):
                result = extract_body(part)
                if result:
                    return result

    return "(could not extract body)"


def send_digest_email(service, digest: str):
    """Send the digest as an email to yourself."""
    profile = service.users().getProfile(userId="me").execute()
    my_email = profile["emailAddress"]

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    message = MIMEText(digest)
    message["to"] = my_email
    message["subject"] = f"Newsletter Digest - {today}"

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    log(f"[Gmail] Digest sent to {my_email}")


# -- LangGraph state and agents --

class NewsletterState(TypedDict):
    newsletters: list[dict]  # Raw newsletter data from Gmail
    sections: list[dict]     # Parsed sections with source info
    summaries: list[dict]    # Per-section summaries
    final_output: str        # Polished digest


llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    default_headers={"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]},
)


def parser_agent(state: NewsletterState) -> dict:
    """Extracts key sections from each newsletter."""
    all_sections = []

    for nl in state["newsletters"]:
        # Truncate very long newsletters to avoid token limits
        body = nl["body"][:8000]
        response = llm.invoke(
            f"""You are a newsletter parser. Extract the distinct topics or articles
from this newsletter. Return each as a separate block separated by '---SECTION---'.
Include only the content, no commentary.

Subject: {nl['subject']}
From: {nl['from']}

Body:
{body}"""
        )
        sections = [s.strip() for s in response.content.split("---SECTION---") if s.strip()]
        for section in sections:
            all_sections.append({"source": nl["subject"], "content": section})
        log(f"[Parser] {nl['subject']}: {len(sections)} sections")

    return {"sections": all_sections}


def summarizer_agent(state: NewsletterState) -> dict:
    """Summarizes each section concisely."""
    summaries = []
    total = len(state["sections"])

    for i, section in enumerate(state["sections"]):
        response = llm.invoke(
            f"""Summarize this newsletter section in 2-3 sentences.
Capture the key point and why it matters.

Source: {section['source']}

Content:
{section['content']}"""
        )
        summaries.append({
            "source": section["source"],
            "summary": response.content,
        })
        log(f"[Summarizer] {i + 1}/{total}")

    return {"summaries": summaries}


def editor_agent(state: NewsletterState) -> dict:
    """Creates a polished, scannable digest from all summaries."""
    combined = "\n\n".join(
        f"From: {s['source']}\n{s['summary']}"
        for s in state["summaries"]
    )
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    response = llm.invoke(
        f"""You are a professional editor. Create a polished daily newsletter digest
for {today}. Group related items, use clear headings and bullet points.
Start with a brief intro, then the summaries, organized by theme.
Keep it concise and scannable.

Source summaries:
{combined}"""
    )
    log("[Editor] Digest ready")
    return {"final_output": response.content}


# -- Build the graph --

def build_graph():
    graph = StateGraph(NewsletterState)

    graph.add_node("parser", parser_agent)
    graph.add_node("summarizer", summarizer_agent)
    graph.add_node("editor", editor_agent)

    graph.add_edge(START, "parser")
    graph.add_edge("parser", "summarizer")
    graph.add_edge("summarizer", "editor")
    graph.add_edge("editor", END)

    return graph.compile()


# -- Main --

if __name__ == "__main__":
    if not NEWSLETTER_SENDERS:
        print("No newsletter senders configured!")
        print("Edit NEWSLETTER_SENDERS in newsletter_summarizer.py")
        print('Example: NEWSLETTER_SENDERS = ["newsletter@example.com"]')
        raise SystemExit(1)

    log("=== Newsletter digest run started ===")

    log("Connecting to Gmail...")
    service = get_gmail_service()

    log("Fetching today's newsletters...")
    newsletters = fetch_todays_newsletters(service)

    if not newsletters:
        log("No newsletters found for today. Nothing to send.")
        raise SystemExit(0)

    log(f"Found {len(newsletters)} newsletters. Processing...")

    app = build_graph()
    result = app.invoke({"newsletters": newsletters})

    print("\n" + "=" * 60)
    print(result["final_output"])
    print("=" * 60)

    send_digest_email(service, result["final_output"])
    log("=== Run complete. Check your inbox. ===")
