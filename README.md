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
```

`ingest` accepts either a single thread directory or an inbox root containing many.

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

Instagram's JSON export (`messages/inbox/<thread>/message_N.json`) today. Parsers sit
behind a `Parser` protocol, so other sources can be added without touching the rest of
the pipeline.

The Instagram parser handles the export's quirks explicitly: text stored as UTF-8 bytes
written out as latin-1, threads split across numbered files with messages newest-first
inside each one, reactions and call logs and unsend tombstones, and media messages that
carry no text.

## Storage

One SQLite file, `data/chatmem.db` by default (`--db` overrides). Re-running `ingest` on
the same archive replaces rows rather than duplicating them.

## License

MIT. See [LICENSE](LICENSE).
