# chatmem

Turn an exported chat archive into a queryable store of structured facts about one
conversation participant.

chatmem parses a chat export, normalizes it into sessions, uses a local LLM to extract
statements the chosen participant made about themselves, and lets you query the result
from the command line with citations back to dates in the archive.

**Everything runs where you point it, and by default that is your machine.** Inference
goes to an OpenAI-compatible server at `llm.base_url` — [LM Studio](https://lmstudio.ai/)
on `localhost` out of the box; storage is a single SQLite file. No cloud services, no
telemetry, nothing is uploaded anywhere. Changing `base_url` to a host you do not control
changes that, deliberately and only when you say so — see
[Using a remote GPU](#using-a-remote-gpu).

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
chatmem digest                               # everything, as one markdown document
chatmem reembed                              # re-embed with a new embedding model
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
`--force` to redo them anyway — after changing `llm.chat_model` or the extraction prompt.
To change only `llm.embedding_model`, use [`reembed`](#reembed) instead; it skips the
chat pass entirely.

Statements that restate something already extracted are dropped; tune or disable that
with `extraction.dedup_threshold` in `config.yaml`. Sessions the run is about to rebuild
are excluded from that comparison, so `--force` rebuilds a session rather than deduping
its new statements against the copies it is replacing.

### reembed

Embedding models improve. `reembed` swaps yours without paying for extraction
again: it rewrites vectors in place, leaving statement rows, their ids, and their
topics untouched, so `MEMORIES.md` citations stay valid.

```bash
chatmem reembed          # after changing llm.embedding_model in config.yaml
```

`extract --force` would also re-embed, but it redoes the chat pass to produce the
same statements -- minutes per hundred sessions against seconds. Use `extract
--force` only when you change `llm.chat_model` or the extraction prompt.

It is whole-database on purpose. Vectors from different models are not
comparable, so re-embedding part of the store would leave `query` ranking
against a mix it cannot interpret.

### digest

`query` answers a question you already thought to ask. `digest` is for the ones you
didn't: it writes every statement to a single markdown document, `MEMORIES.md` by
default, grouped under a fixed set of topics.

```markdown
## Places

_Where they live, have lived, have moved to or from, and places they have travelled to
or want to._

- **2023-04-02** — Moved to Berlin in the spring. `#17`
- **2023-08-11 – 2023-08-14** — Was travelling around the Baltic coast. `#43`
```

Its contract is completeness, not relevance: every statement appears exactly once,
verbatim, with its date and the same id that `chatmem statements` and `chatmem export`
use. Nothing is summarized or paraphrased. Pass `--quotes` to nest the original messages
under each line, and `--out` to write somewhere other than `MEMORIES.md`.

The topics are a closed list, defined in `chatmem/topics.py`, so the document stays
diffable as the archive grows — a memory added next month lands under the heading it
would have landed under last month. Classification is one batched chat call per
`--batch-size` (40) statements, stored on the statement, so re-running after a new
`extract` only costs the new rows and re-rendering costs nothing. Editing the taxonomy
means re-filing everything: `chatmem digest --reclassify`.

Statements the model fails to classify are left untagged rather than forced into a
bucket. They still render, under an `Unclassified` heading, and the next run retries
them.

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

## Updating later

Every step is incremental, so a later export costs roughly what actually changed
rather than what the archive contains. Export again over the same thread
directories and run the same three commands:

```bash
chatmem ingest path/to/messages/inbox --source instagram
chatmem extract
chatmem digest
```

- `ingest` rebuilds each thread, but message ids are derived from the message
  rather than its position, and sessions are matched on their time bounds, so
  statements for sessions whose contents did not change are carried over.
- `extract` skips sessions marked extracted, so it processes only the new
  conversation — and the session at the join, if the new export merged into it.
- `digest` classifies only statements with no topic yet, and re-renders the
  whole document from what is stored.

The exception is a change to `llm.chat_model` or the extraction prompt, which
invalidates the statements themselves rather than the sessions: that needs
`chatmem extract --force`. A change to `llm.embedding_model` alone needs only
[`reembed`](#reembed).

Pass `--source` explicitly when the export folder contains media subdirectories,
or parser auto-detection may find the directory ambiguous and stop rather than
guess.

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

## Using a remote GPU

`extract` is the slow step, and a laptop is a poor place to run it. Any
OpenAI-compatible server will do, including [Ollama](https://ollama.com/) on a remote
machine reached through an SSH tunnel:

```bash
ssh -N -L 11434:127.0.0.1:11434 you@gpu-host
```

with `llm.base_url: http://localhost:11434/v1`. Extraction is resumable and commits per
session, so a tunnel that drops mid-run costs one session; add
`ServerAliveInterval 30` to your SSH config anyway.

Understand what this does and does not protect before pointing it at a machine you do not
administer. It does protect the data in transit, and it keeps the archive off the remote
host — only prompt text crosses the tunnel, while the export and `chatmem.db` stay local.
Never copy the archive to the remote machine; there is no reason to.

It does **not** make you anonymous to that host. Anyone with root there can read your
process memory and the model's GPU memory, and can turn on request logging. Prompt text
can reach disk through swap. Your logins are recorded in the host's auth logs. Concretely,
if you go ahead:

- Run your own server rather than sharing a system one, bound to loopback with debug
  logging off: `OLLAMA_HOST=127.0.0.1:11434 OLLAMA_DEBUG=0 ollama serve`. A server bound
  to `0.0.0.0` publishes an unauthenticated LLM API to the whole network.
- Use the API only. `ollama run` writes an interactive history file to disk; the server
  path does not persist prompts or responses.
- `ollama stop <model>` when you are done, so the context does not sit in VRAM.

If the machine belongs to an employer, its acceptable-use policy probably asserts a right
to inspect anything on it — and the other person in your chat logs is not a party to that
policy. That is a consent question the tunnel does not answer.

Vectors from different embedding models are not comparable, and the same nominal model can
differ between runtimes. Switching servers means `chatmem extract --force`.

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

This holds for a *fuller* re-export too, not just an identical one. Instagram splits a
thread so that `message_2.json` holds older messages than `message_1.json`, so exporting
again after the conversation has grown prepends history and shifts every message's
position in the thread. Message ids are derived from the message itself rather than its
position, and sessions are matched on their time bounds rather than their offsets, so
only the sessions whose contents actually changed — typically just the one at the join —
are re-extracted. Everything already done is kept.

The schema migrates itself forward on connect. Back up `data/chatmem.db` before the first
run after an upgrade.

## License

MIT. See [LICENSE](LICENSE).
