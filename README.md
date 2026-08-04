# chatmem

Turn an exported chat archive into a queryable store of structured facts about one
conversation participant.

chatmem parses a chat export, normalizes it into sessions, uses a local LLM to extract
statements the chosen participant made about themselves, and lets you query the result
from the command line with citations back to dates in the archive.

**Everything runs on your machine.** Inference goes to
[LM Studio](https://lmstudio.ai/)'s OpenAI-compatible server on `localhost`; storage is a
single SQLite file. No cloud services, no telemetry, nothing is uploaded anywhere.

## Responsibility

You are responsible for having the right to process the conversation you feed to this
tool. Chat logs contain other people's words. Check the law that applies to you and the
terms of the service the archive came from, and get consent where it is required.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- LM Studio running its local server, with a chat model and an embedding model loaded

## Setup

```bash
uv sync --group dev
uv run pre-commit install   # one-time; enables the gitleaks pre-commit hook
cp config.example.yaml config.yaml
```

Then edit `config.yaml` and set `llm.chat_model` / `llm.embedding_model` to model ids
your LM Studio instance actually serves. There are no defaults for model names on
purpose — chatmem never hardcodes one.

`config.yaml` is gitignored. `config.example.yaml` is the committed template.

## Usage

```bash
chatmem ingest path/to/messages/inbox        # parse, normalize, sessionize
chatmem identities                           # see who was found, and how they mapped
chatmem stats                                # counts per thread and per person
chatmem sessions                             # session ranges
chatmem extract                              # LLM pass: sessions -> statements
chatmem query "where has Alex lived?"        # semantic search over those statements
chatmem statements                           # list what was extracted
chatmem export --format csv                  # dump statements as JSON or CSV
```

`ingest` accepts a single thread directory, an inbox root containing many, or a WhatsApp
`.txt` export.

### extract

`extract` is the slow, expensive step: one chat call per session, plus one batched
embedding call. It is resumable — each session is committed as it finishes, and a re-run
skips sessions already processed, so an interrupted run picks up where it stopped. Use
`--force` to redo them anyway (required after changing `llm.embedding_model`, since
vectors from different models are not comparable).

Statements that restate something already extracted are dropped; tune or disable that
with `extraction.dedup_threshold` in `config.yaml`.

### query

`query` embeds your question and ranks the target's statements by cosine similarity,
printing each match with its score, its date range, and the original messages it came
from:

```
[1] [0.782] Moved to Berlin in the spring.
    thread_alpha  2023-04-02T18:20:01.000000Z .. 2023-04-02T18:24:55.000000Z
    > Alex Rivera: finally made the move to berlin last week
```

Narrow the search with `--thread`, `--since` / `--until` (`YYYY-MM-DD`), `--min-score`,
and `--limit`. By default the command returns ranked statements, not prose; pass
`--answer` to also have the chat model compose an answer from the top results, citing
them by the same numbers.

## Identities: one person, several accounts

The same person may appear under different display names — a second account, a renamed
profile, a different thread. chatmem resolves every raw sender to a `person_id`, and all
downstream work keys on that, so facts from several accounts merge into one picture.

Declare the mapping in `config.yaml`:

```yaml
identities:
  - id: target
    display_name: Participant A
    aliases: ["Participant A", "participant.a", "Participant A (new)"]
  - id: other
    display_name: Participant B
    aliases: ["Participant B"]

target: target
```

Senders you have not declared are auto-created as their own person, and `ingest` prints a
warning listing them with a ready-to-paste `identities:` block. Run `chatmem identities`
at any time to see the current mapping.

`config.yaml` is the only source of truth. After editing it, `chatmem relink` re-resolves
existing rows in place — no re-parsing of the archive.

Aliases match on a normalized display name (case-folded, whitespace-collapsed) across all
threads. If two genuinely different people share a display name in your archive, that
name cannot be split apart.

## Supported archives

Instagram's JSON export (`messages/inbox/<thread>/message_N.json`) and WhatsApp's
"Export chat" `.txt` file. Parsers sit behind a `Parser` protocol, so other sources can
be added without touching the rest of the pipeline. Pass `--source instagram|whatsapp` if
auto-detection picks wrong.

The Instagram parser handles the export's quirks explicitly: text stored as UTF-8 bytes
written out as latin-1, threads split across numbered files with messages newest-first
inside each one, reactions and call logs and unsend tombstones, and media messages that
carry no text.

The WhatsApp parser handles both the iOS and Android line layouts, messages that span
several lines, media placeholders, and system notices. WhatsApp records no time zone and
no date order, so the day/month order is inferred per file and times are read as UTC —
consistent within a thread, which is what sessionizing depends on.

## Storage

One SQLite file, `data/chatmem.db` by default (`--db` overrides). Re-running `ingest` on
the same archive replaces rows rather than duplicating them, and statements already
extracted for a session survive the rebuild as long as that session's messages haven't
changed — so re-ingesting doesn't throw away LLM work. If a session's messages did
change, its statements are dropped and it is queued for re-extraction.

The schema migrates itself forward on connect. Back up `data/chatmem.db` before the first
run after an upgrade.

## License

MIT. See [LICENSE](LICENSE).
