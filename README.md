# Multi-agent newsletter summarizer

A three-agent pipeline that reads the day's newsletters from Gmail and emails
you one clean digest. Built with [LangGraph](https://langchain-ai.github.io/langgraph/)
and Claude.

```
Gmail inbox ──▶ Parser ──▶ Summarizer ──▶ Editor ──▶ Gmail (digest)
                clean       condense       stitch into
                text        each section   one digest
```

Each agent does one job, so every step is easy to inspect and tune on its own —
a small, honest example of treating an LLM as a component in a system rather
than a chat box.

## How it works

- **Parser** pulls today's newsletters (by sender) via the Gmail API, strips
  HTML to clean text, and splits each into topic sections.
- **Summarizer** condenses each section to its substance in 2–3 sentences.
- **Editor** stitches the summaries into one themed, scannable digest.
- The digest is emailed back to your own inbox. A cron job runs the whole thing
  on a schedule.

`newsletter_summarizer.py` is the entire program; LangGraph wires the three
nodes into a typed `StateGraph`.

## Setup

1. **Install deps** (Python 3.11+):
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Anthropic:** copy `.env.example` to `.env` and add your key.
3. **Gmail API:** in Google Cloud, create an OAuth *Desktop app* credential,
   download it as `credentials.json` (see `credentials.example.json` for the
   shape), and enable the Gmail API. First run opens a browser to authorize and
   writes a local `token.json`.
4. **Pick your newsletters:** edit `NEWSLETTER_SENDERS` at the top of
   `newsletter_summarizer.py`.
5. **Run:**
   ```bash
   python newsletter_summarizer.py
   ```

## Scheduling

Run it daily with cron, e.g. 7am:

```cron
0 7 * * * cd /path/to/this/repo && ./venv/bin/python newsletter_summarizer.py >> cron.log 2>&1
```

If the Gmail token can't refresh during a non-interactive run, the script sends
a macOS notification and exits so you can re-authorize by running it manually.

## Security

`.env`, `credentials.json`, and `token.json` hold secrets and are gitignored —
never commit them. Use `.env.example` and `credentials.example.json` as
templates.
