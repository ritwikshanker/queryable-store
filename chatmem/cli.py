"""chatmem CLI: ingest, identities, relink, stats, sessions, extract, reembed, digest, query."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from chatmem import digest as digest_render
from chatmem import store
from chatmem.config import Config, load_config
from chatmem.extract import dedup_statements, extract_session
from chatmem.identity import IdentityResolver
from chatmem.llm import LLMClient, LLMResponseError
from chatmem.models import Message
from chatmem.parsers import select_parser
from chatmem.query import query as run_query
from chatmem.sessionize import sessionize

app = typer.Typer(
    name="chatmem",
    help="Extract structured facts about a participant from a chat archive, locally.",
    no_args_is_help=True,
)


def _load_cfg(
    config_path: Path, db: Optional[Path] = None, target: Optional[str] = None
) -> Config:
    cfg = load_config(config_path)
    return cfg.with_overrides(target=target, db_path=db)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _ms_to_iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    # Explicit microsecond formatting (not dt.isoformat()) so every
    # timestamp has the same length and sorts correctly as plain text,
    # regardless of whether the microsecond component happens to be zero.
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _parse_when(value: Optional[str], *, end: bool) -> Optional[str]:
    """Normalize a --since/--until value to the stored timestamp format.

    Stored timestamps are fixed-width ISO-8601 UTC precisely so they compare
    correctly as plain strings, so filtering is a string comparison once the
    bound is padded out. A bare date expands to the start of that day, or to
    its last microsecond when it's an upper bound.
    """
    if value is None:
        return None
    text = value.strip()
    try:
        if len(text) == 10:
            day = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dt = day.replace(hour=23, minute=59, second=59, microsecond=999999) if end else day
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"could not read {value!r} as a date (use YYYY-MM-DD): {e}") from e
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _message_id(
    thread_id: str,
    timestamp_ms: int,
    sender: str,
    text: str | None,
    media_type: str | None,
    ordinal: int,
) -> str:
    payload = "|".join(
        [thread_id, str(timestamp_ms), sender, text or "", media_type or "", str(ordinal)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="A thread directory or an inbox root."),
    source: Optional[str] = typer.Option(
        None, "--source", help="Force a parser by name instead of auto-detecting."
    ),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    target: Optional[str] = typer.Option(
        None, "--target", help="Override config.yaml's target for this run."
    ),
    db: Optional[Path] = typer.Option(None, "--db", help="Override config.yaml's db_path."),
) -> None:
    """Parse an export, normalize it, resolve identities, and sessionize."""
    cfg = _load_cfg(config, db=db, target=target)
    conn = store.connect(cfg.db_path)
    resolver = IdentityResolver(cfg.identities)
    parser = select_parser(path, name=source)

    ingested_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    n_threads = 0
    n_kept = 0
    n_sessions = 0
    n_preserved = 0
    dropped_by_reason: dict[str, int] = defaultdict(int)

    for thread in parser.parse(path):
        n_threads += 1
        store.upsert_thread(
            conn, thread.thread_id, thread.title, thread.participants, thread.source, ingested_at
        )

        # The id must not change when a later export prepends older messages,
        # or every statement citing it is discarded on re-ingest. So the
        # disambiguator is the message's index among rows identical to it in
        # this thread -- stable under insertion -- not its position in the
        # thread, which shifts.
        duplicate_index: dict[tuple, int] = defaultdict(int)

        normalized: list[Message] = []
        for seq, raw in enumerate(thread.messages):
            person_id = resolver.resolve(raw.sender)
            identity = (raw.timestamp_ms, raw.sender, raw.text, raw.media_type)
            ordinal = duplicate_index[identity]
            duplicate_index[identity] += 1
            normalized.append(
                Message(
                    id=_message_id(
                        thread.thread_id, raw.timestamp_ms, raw.sender, raw.text, raw.media_type, ordinal
                    ),
                    thread_id=thread.thread_id,
                    sender=raw.sender,
                    person_id=person_id,
                    timestamp_utc=_ms_to_iso(raw.timestamp_ms),
                    timestamp_ms=raw.timestamp_ms,
                    text=raw.text,
                    media_type=raw.media_type,
                    seq=seq,
                )
            )
        # People referenced by these messages (including any just
        # auto-created by resolver.resolve() above) must exist before the
        # messages that reference them, since messages.person_id is a
        # foreign key. Committing is deferred so the whole thread lands as
        # one transaction -- a crash mid-thread rolls back cleanly.
        store.save_identities(conn, resolver, commit=False)
        n_kept += len(normalized)
        for reason, count in thread.dropped.items():
            dropped_by_reason[reason] += count

        ranges = sessionize(
            normalized,
            idle_gap_minutes=cfg.sessionize.idle_gap_minutes,
            max_messages=cfg.sessionize.max_messages,
        )
        messages_by_seq = {m.seq: m for m in normalized}
        saved, preserved = store.replace_thread_content(
            conn, thread.thread_id, normalized, ranges, messages_by_seq
        )
        n_sessions += len(saved)
        n_preserved += preserved
        conn.commit()

    store.save_identities(conn, resolver)
    conn.commit()

    typer.echo(f"threads:  {n_threads}")
    typer.echo(f"messages: {n_kept} kept")
    if dropped_by_reason:
        breakdown = ", ".join(f"{reason}={count}" for reason, count in sorted(dropped_by_reason.items()))
        typer.echo(f"dropped:  {sum(dropped_by_reason.values())} ({breakdown})")
    else:
        typer.echo("dropped:  0")
    typer.echo(f"sessions: {n_sessions}")
    if n_preserved:
        typer.echo(f"statements preserved: {n_preserved}")

    new_senders = resolver.new_senders()
    if new_senders:
        names = ", ".join(sorted({ns.raw_name for ns in new_senders}))
        typer.echo(
            f"\nWarning: {len(new_senders)} sender(s) not declared in config.yaml's "
            f"identities: were auto-created: {names}"
        )
        typer.echo("Paste-ready block:\n")
        typer.echo("identities:")
        typer.echo(resolver.new_senders_yaml())

    effective_target = cfg.target
    if effective_target:
        resolved = resolver.resolve_target(effective_target)
        if resolved is None:
            typer.echo(
                f"\nWarning: target {effective_target!r} did not match any sender seen "
                "in this ingest (it may appear in another thread)."
            )
        else:
            typer.echo(f"\ntarget resolved to person_id={resolved!r}")


@app.command()
def identities(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """List observed senders, their resolved person, and origin."""
    cfg = _load_cfg(config, db=db)
    conn = store.connect(cfg.db_path)

    people = store.all_people(conn)
    alias_rows = conn.execute(
        "SELECT alias_norm, raw_name, person_id FROM aliases ORDER BY person_id"
    ).fetchall()
    aliases_by_pid: dict[str, list[str]] = defaultdict(list)
    for row in alias_rows:
        aliases_by_pid[row["person_id"]].append(row["raw_name"])

    if not people:
        typer.echo("No people found. Run `chatmem ingest` first.")
        raise typer.Exit(code=0)

    for p in sorted(people, key=lambda p: (p.origin, p.person_id)):
        names = sorted(set(aliases_by_pid.get(p.person_id, [])))
        typer.echo(f"{p.person_id:24} [{p.origin:6}] {p.display_name!r:30} aliases={names}")

    auto = [p for p in people if p.origin == "auto"]
    if auto:
        typer.echo("\n# Paste into config.yaml to declare these explicitly:")
        typer.echo("identities:")
        for p in auto:
            names = sorted(set(aliases_by_pid.get(p.person_id, [])))
            typer.echo(f"  - id: {p.person_id}")
            typer.echo(f"    display_name: {p.display_name!r}")
            typer.echo(f"    aliases: {names!r}")


@app.command()
def relink(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Re-resolve every message's person_id from the current config.yaml."""
    cfg = _load_cfg(config, db=db)
    conn = store.connect(cfg.db_path)
    resolver = IdentityResolver(cfg.identities)
    result = store.relink_messages(conn, resolver)
    typer.echo(f"relinked {result.messages} messages")
    if result.statements:
        typer.echo(f"re-attributed {result.statements} statements")
    if result.ambiguous_session_ids:
        ids = ", ".join(f"#{i}" for i in result.ambiguous_session_ids)
        typer.echo(
            f"\nWarning: one person was split into several, so statements in these "
            f"sessions could not be re-attributed: {ids}\n"
            "Re-extract them with `chatmem extract --force`."
        )

    new_senders = resolver.new_senders()
    if new_senders:
        names = ", ".join(sorted({ns.raw_name for ns in new_senders}))
        typer.echo(f"Warning: {len(new_senders)} sender(s) still unmapped: {names}")


@app.command()
def stats(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Counts per thread and per person, with date ranges. No LLM involved."""
    cfg = _load_cfg(config, db=db)
    conn = store.connect(cfg.db_path)

    typer.echo("Threads:")
    for row in store.thread_stats(conn):
        typer.echo(
            f"  {row['thread_id']:30} messages={row['message_count']:<6} "
            f"{row['first_ts'] or '-'} .. {row['last_ts'] or '-'}"
        )

    typer.echo("\nPeople:")
    for row in store.person_stats(conn):
        typer.echo(
            f"  {row['person_id']:24} [{row['origin']:6}] {row['display_name']!r:30} "
            f"messages={row['message_count']}"
        )


@app.command()
def sessions(
    thread: Optional[str] = typer.Option(None, "--thread"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """List session ranges and timestamps. No LLM involved."""
    cfg = _load_cfg(config, db=db)
    conn = store.connect(cfg.db_path)

    for s in store.list_sessions(conn, thread_id=thread, limit=limit):
        typer.echo(
            f"  #{s.id:<5} {s.thread_id:24} seq[{s.start_seq}:{s.end_seq}] "
            f"n={s.message_count:<4} {s.start_ts} .. {s.end_ts}"
        )


def _resolve_person_or_exit(conn, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    person_id = store.resolve_person(conn, value)
    if person_id is None:
        typer.echo(f"Error: person {value!r} was not found.", err=True)
        raise typer.Exit(code=1)
    return person_id


@app.command()
def statements(
    person: Optional[str] = typer.Option(None, "--person", help="A person_id or alias."),
    thread: Optional[str] = typer.Option(None, "--thread"),
    session: Optional[int] = typer.Option(None, "--session", help="A session id from `sessions`."),
    limit: Optional[int] = typer.Option(None, "--limit"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """List extracted statements. No LLM involved."""
    cfg = _load_cfg(config, db=db)
    conn = store.connect(cfg.db_path)

    person_id = _resolve_person_or_exit(conn, person)
    rows = store.list_statements(
        conn, person_id=person_id, thread_id=thread, session_id=session, limit=limit
    )
    if not rows:
        typer.echo("No statements found. Run `chatmem extract` first.")
        return

    for s in rows:
        typer.echo(
            f"#{s.id:<5} [{s.person_id:12}] {s.start_ts[:10]}..{s.end_ts[:10]} "
            f"(session #{s.session_id}, {s.thread_id})"
        )
        typer.echo(f"      {s.text}")


@app.command()
def export(
    format: str = typer.Option("json", "--format", help="json or csv."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write to a file instead of stdout."),
    person: Optional[str] = typer.Option(None, "--person", help="A person_id or alias."),
    thread: Optional[str] = typer.Option(None, "--thread"),
    session: Optional[int] = typer.Option(None, "--session"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Export statements as JSON or CSV. Embeddings are not included."""
    fmt = format.lower()
    if fmt not in {"json", "csv"}:
        typer.echo(f"Error: unknown --format {format!r} (use json or csv).", err=True)
        raise typer.Exit(code=1)

    cfg = _load_cfg(config, db=db)
    conn = store.connect(cfg.db_path)

    person_id = _resolve_person_or_exit(conn, person)
    rows = store.list_statements(
        conn, person_id=person_id, thread_id=thread, session_id=session
    )
    names = {p.person_id: p.display_name for p in store.all_people(conn)}
    records = [
        {
            "id": s.id,
            "person_id": s.person_id,
            "display_name": names.get(s.person_id, s.person_id),
            "thread_id": s.thread_id,
            "session_id": s.session_id,
            "start_ts": s.start_ts,
            "end_ts": s.end_ts,
            "created_at": s.created_at,
            "topic": s.topic,
            "text": s.text,
            "source_message_ids": s.source_message_ids,
        }
        for s in rows
    ]

    buffer = io.StringIO()
    if fmt == "json":
        json.dump(records, buffer, indent=2, ensure_ascii=False)
        buffer.write("\n")
    else:
        writer = csv.DictWriter(buffer, fieldnames=list(_EXPORT_FIELDS))
        writer.writeheader()
        for record in records:
            # One cell can't hold a list; the ids are space-separated so the
            # column stays greppable.
            writer.writerow({**record, "source_message_ids": " ".join(record["source_message_ids"])})
    payload = buffer.getvalue()

    if out is None:
        typer.echo(payload, nl=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        typer.echo(f"wrote {len(records)} statements to {out}")


_EXPORT_FIELDS = (
    "id",
    "person_id",
    "display_name",
    "thread_id",
    "session_id",
    "start_ts",
    "end_ts",
    "created_at",
    "topic",
    "text",
    "source_message_ids",
)


@app.command()
def extract(
    thread: Optional[str] = typer.Option(None, "--thread"),
    force: bool = typer.Option(
        False, "--force", help="Re-extract sessions that were already processed."
    ),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    target: Optional[str] = typer.Option(
        None, "--target", help="Override config.yaml's target for this run."
    ),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Run each session the target participated in through the LLM, storing statements.

    Resumable: sessions already processed are skipped unless --force is given,
    and each session is committed as it completes, so an interrupted run keeps
    the work it finished.
    """
    cfg = _load_cfg(config, db=db, target=target)
    if not cfg.target:
        typer.echo(
            "Error: no target set. Pass --target or set 'target' in config.yaml.", err=True
        )
        raise typer.Exit(code=1)
    if not cfg.llm.chat_model:
        typer.echo("Error: llm.chat_model is not set in config.yaml.", err=True)
        raise typer.Exit(code=1)
    if not cfg.llm.embedding_model:
        typer.echo(
            "Error: llm.embedding_model is not set in config.yaml -- required to embed "
            "extracted statements for `chatmem query`.",
            err=True,
        )
        raise typer.Exit(code=1)

    llm = LLMClient(cfg.llm)
    conn = store.connect(cfg.db_path)

    target_person_id = store.resolve_person(conn, cfg.target)
    if target_person_id is None:
        typer.echo(
            f"Error: target {cfg.target!r} was not found. Run `chatmem ingest` first.", err=True
        )
        raise typer.Exit(code=1)

    # Vectors from different embedding models aren't comparable, so a whole
    # DB is embedded with one model. Changing models means re-embedding
    # everything, which is what --force (across all threads) does.
    stored_model = store.get_meta(conn, "embedding_model")
    if stored_model is not None and stored_model != cfg.llm.embedding_model:
        if not force or thread is not None:
            typer.echo(
                f"Error: stored statements were embedded with {stored_model!r}, but "
                f"config.yaml says {cfg.llm.embedding_model!r}. Re-embed everything with "
                "`chatmem extract --force` (without --thread).",
                err=True,
            )
            raise typer.Exit(code=1)
        store.set_meta(conn, "embedding_model", cfg.llm.embedding_model)
        store.set_meta(conn, "embedding_dim", "")
        conn.commit()
        stored_model = cfg.llm.embedding_model
        # Old vectors are about to be replaced; don't dedup against them.
        drop_existing_embeddings = True
    else:
        drop_existing_embeddings = False

    display_name_by_person = {p.person_id: p.display_name for p in store.all_people(conn)}
    target_name = display_name_by_person.get(target_person_id, target_person_id)

    participants_by_session = store.participants_by_session(conn)
    # Dedup compares against what the target already has stored; kept
    # statements are appended as sessions complete.
    seen_embeddings = (
        []
        if drop_existing_embeddings
        else [
            s.embedding
            for s in store.list_statements(conn, person_id=target_person_id)
            if s.embedding is not None
        ]
    )

    n_skipped = 0
    n_done = 0
    n_failed = 0
    n_extracted = 0
    n_processed = 0
    n_deduped = 0

    sessions = store.list_sessions(conn, thread_id=thread)
    total = len(sessions)
    interrupted = False

    try:
        for i, session in enumerate(sessions, start=1):
            if target_person_id not in participants_by_session.get(session.id, set()):
                n_skipped += 1
                continue
            if session.extracted_at is not None and not force:
                n_done += 1
                continue

            messages = store.session_messages(
                conn, session.thread_id, session.start_seq, session.end_seq
            )
            try:
                statements = extract_session(
                    session,
                    messages,
                    target_person_id,
                    target_name,
                    llm,
                    display_name_by_person,
                    validation_pass=cfg.extraction.validation_pass,
                )
            except LLMResponseError as e:
                typer.echo(f"Warning: session #{session.id} failed: {e}")
                n_failed += 1
                continue

            statements, dropped = dedup_statements(
                statements, seen_embeddings, cfg.extraction.dedup_threshold
            )
            seen_embeddings.extend(s.embedding for s in statements if s.embedding is not None)

            store.replace_session_statements(conn, session.id, statements)
            store.mark_session_extracted(conn, session.id, _now_iso())
            if statements and not store.get_meta(conn, "embedding_dim"):
                store.set_meta(conn, "embedding_model", cfg.llm.embedding_model)
                store.set_meta(conn, "embedding_dim", str(len(statements[0].embedding or [])))
            # Commit per session so an interrupted run resumes from here
            # instead of discarding everything.
            conn.commit()

            n_processed += 1
            n_extracted += len(statements)
            n_deduped += dropped
            typer.echo(
                f"[{i}/{total}] session #{session.id} {session.thread_id}: "
                f"{len(statements)} statements"
                + (f" ({dropped} deduped)" if dropped else "")
            )
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        typer.echo("\nInterrupted -- progress was saved; re-run `chatmem extract` to resume.")

    typer.echo(
        f"sessions:   {n_processed} processed, {n_skipped} skipped (target absent), "
        f"{n_done} already extracted, {n_failed} failed"
    )
    typer.echo(f"statements: {n_extracted} extracted, {n_deduped} deduped")

    if interrupted:
        raise typer.Exit(code=130)


@app.command()
def reembed(
    batch_size: int = typer.Option(64, "--batch-size", help="Statements per embedding call."),
    force: bool = typer.Option(
        False, "--force", help="Re-embed even if the stored model already matches config."
    ),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Re-embed stored statements with the configured embedding model.

    For swapping in a better embedding model without paying for extraction
    again: `extract --force` would redo the chat pass, which is the expensive
    part and produces the same statements. This rewrites vectors in place, so
    statement ids -- and anything citing them, like MEMORIES.md -- stay valid.

    Deliberately whole-database: vectors from different models are not
    comparable, so re-embedding only part of the store would leave `query`
    ranking against a mix it cannot interpret.
    """
    cfg = _load_cfg(config, db=db)
    if not cfg.llm.embedding_model:
        typer.echo("Error: llm.embedding_model is not set in config.yaml.", err=True)
        raise typer.Exit(code=1)
    if batch_size < 1:
        typer.echo("Error: --batch-size must be at least 1.", err=True)
        raise typer.Exit(code=1)

    conn = store.connect(cfg.db_path)
    rows = store.list_statements(conn)
    if not rows:
        typer.echo("No statements found. Run `chatmem extract` first.")
        raise typer.Exit(code=1)

    stored_model = store.get_meta(conn, "embedding_model")
    if stored_model == cfg.llm.embedding_model and not force:
        missing = [s for s in rows if s.embedding is None]
        if not missing:
            typer.echo(
                f"All {len(rows)} statements are already embedded with "
                f"{cfg.llm.embedding_model!r}. Pass --force to redo them anyway."
            )
            return
        rows = missing
        typer.echo(f"embedding {len(rows)} statements that have no vector yet")
    else:
        typer.echo(
            f"re-embedding {len(rows)} statements: "
            f"{stored_model or 'none'!r} -> {cfg.llm.embedding_model!r}"
        )

    llm = LLMClient(cfg.llm)
    total_batches = (len(rows) + batch_size - 1) // batch_size
    n_done = 0
    dim: int | None = None
    try:
        for b in range(total_batches):
            batch = rows[b * batch_size : (b + 1) * batch_size]
            try:
                vectors = llm.embed([s.text for s in batch])
            except LLMResponseError as e:
                # Partial progress is committed but the model marker is not,
                # so a resumed run re-embeds everything rather than leaving
                # the store split across two models.
                typer.echo(f"Error: embedding batch {b + 1} failed: {e}", err=True)
                raise typer.Exit(code=1) from e
            store.set_statement_embeddings(
                conn, {s.id: v for s, v in zip(batch, vectors) if s.id is not None}
            )
            conn.commit()
            dim = len(vectors[0]) if vectors else dim
            n_done += len(batch)
            typer.echo(f"[{b + 1}/{total_batches}] embedded {n_done}/{len(rows)}")
    except KeyboardInterrupt:
        typer.echo("\nInterrupted -- re-run `chatmem reembed` to finish.")
        raise typer.Exit(code=130) from None

    store.set_meta(conn, "embedding_model", cfg.llm.embedding_model)
    if dim is not None:
        store.set_meta(conn, "embedding_dim", str(dim))
    conn.commit()
    typer.echo(f"done: {n_done} statements embedded with {cfg.llm.embedding_model!r} (dim {dim})")


@app.command()
def digest(
    out: Path = typer.Option(Path("MEMORIES.md"), "--out", help="Markdown file to write."),
    quotes: bool = typer.Option(
        False, "--quotes", help="Include the source messages under each memory."
    ),
    reclassify: bool = typer.Option(
        False, "--reclassify", help="Re-file every statement, not just untagged ones."
    ),
    batch_size: int = typer.Option(40, "--batch-size", help="Statements per classify call."),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    target: Optional[str] = typer.Option(
        None, "--target", help="Override config.yaml's target for this run."
    ),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Write every extracted memory to one markdown document, grouped by topic.

    Files each statement under a fixed topic with one batched LLM call per
    `--batch-size` statements, then renders. Resumable and cheap to re-run:
    only statements not yet classified cost anything, so a digest after a new
    `extract` classifies just the new rows.
    """
    cfg = _load_cfg(config, db=db, target=target)
    if not cfg.target:
        typer.echo(
            "Error: no target set. Pass --target or set 'target' in config.yaml.", err=True
        )
        raise typer.Exit(code=1)
    if batch_size < 1:
        typer.echo("Error: --batch-size must be at least 1.", err=True)
        raise typer.Exit(code=1)

    conn = store.connect(cfg.db_path)
    target_person_id = store.resolve_person(conn, cfg.target)
    if target_person_id is None:
        typer.echo(
            f"Error: target {cfg.target!r} was not found. Run `chatmem ingest` first.", err=True
        )
        raise typer.Exit(code=1)

    if reclassify:
        cleared = store.clear_statement_topics(conn, person_id=target_person_id)
        conn.commit()
        typer.echo(f"cleared {cleared} existing topic assignments")

    pending = store.list_statements(conn, person_id=target_person_id, untagged=True)
    if pending:
        if not cfg.llm.chat_model:
            typer.echo(
                f"Error: {len(pending)} statements are not classified yet and "
                "llm.chat_model is not set in config.yaml.",
                err=True,
            )
            raise typer.Exit(code=1)
        llm = LLMClient(cfg.llm)
        n_failed = 0
        total_batches = (len(pending) + batch_size - 1) // batch_size
        for b in range(total_batches):
            batch = pending[b * batch_size : (b + 1) * batch_size]
            try:
                assigned = llm.classify_statements([s.text for s in batch])
            except LLMResponseError as e:
                # Left untagged rather than forced into a bucket: they still
                # render, under Unclassified, and the next run retries them.
                typer.echo(f"Warning: classify batch {b + 1} failed: {e}")
                n_failed += len(batch)
                continue
            store.set_statement_topics(
                conn, {s.id: topic for s, topic in zip(batch, assigned) if s.id is not None}
            )
            # Commit per batch so an interrupted run keeps what it classified.
            conn.commit()
            typer.echo(f"[{b + 1}/{total_batches}] classified {len(batch)} statements")
        if n_failed:
            typer.echo(f"{n_failed} statements left unclassified -- re-run to retry them")

    rows = store.list_statements(conn, person_id=target_person_id)
    if not rows:
        typer.echo("No statements found. Run `chatmem extract` first.")
        raise typer.Exit(code=1)

    names = {p.person_id: p.display_name for p in store.all_people(conn)}
    thread_ids = {s.thread_id for s in rows}
    quotes_by_statement = None
    if quotes:
        quotes_by_statement = {
            s.id: store.messages_by_ids(conn, s.source_message_ids)
            for s in rows
            if s.id is not None
        }

    meta = digest_render.DigestMeta(
        title=names.get(target_person_id, target_person_id),
        generated_at=_now_iso(),
        message_count=sum(r["message_count"] for r in store.thread_stats(conn)),
        session_count=len(store.list_sessions(conn)),
        thread_count=len(thread_ids),
    )
    markdown = digest_render.render(
        meta,
        rows,
        show_threads=len(thread_ids) > 1,
        quotes_by_statement=quotes_by_statement,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    buckets = digest_render.group_by_topic(rows)
    typer.echo(f"wrote {len(rows)} memories in {len(buckets)} sections to {out}")


@app.command()
def query(
    question: str = typer.Argument(..., help="A natural-language question about the target."),
    limit: int = typer.Option(5, "--limit"),
    thread: Optional[str] = typer.Option(None, "--thread", help="Only search one thread."),
    since: Optional[str] = typer.Option(
        None, "--since", help="Only statements at or after this date (YYYY-MM-DD)."
    ),
    until: Optional[str] = typer.Option(
        None, "--until", help="Only statements at or before this date (YYYY-MM-DD)."
    ),
    min_score: Optional[float] = typer.Option(
        None, "--min-score", help="Drop results scoring below this similarity."
    ),
    answer: bool = typer.Option(
        False, "--answer", help="Also compose an answer from the results, with citations."
    ),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    target: Optional[str] = typer.Option(
        None, "--target", help="Override config.yaml's target for this run."
    ),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Semantically search the target's extracted statements, with citations."""
    cfg = _load_cfg(config, db=db, target=target)
    if not cfg.target:
        typer.echo(
            "Error: no target set. Pass --target or set 'target' in config.yaml.", err=True
        )
        raise typer.Exit(code=1)
    if not cfg.llm.embedding_model:
        typer.echo("Error: llm.embedding_model is not set in config.yaml.", err=True)
        raise typer.Exit(code=1)
    if answer and not cfg.llm.chat_model:
        typer.echo(
            "Error: llm.chat_model is not set in config.yaml -- required for --answer.", err=True
        )
        raise typer.Exit(code=1)

    try:
        since_ts = _parse_when(since, end=False)
        until_ts = _parse_when(until, end=True)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    conn = store.connect(cfg.db_path)

    target_person_id = store.resolve_person(conn, cfg.target)
    if target_person_id is None:
        typer.echo(
            f"Error: target {cfg.target!r} was not found. Run `chatmem ingest` first.", err=True
        )
        raise typer.Exit(code=1)

    stored_model = store.get_meta(conn, "embedding_model")
    if stored_model is not None and stored_model != cfg.llm.embedding_model:
        typer.echo(
            f"Error: stored statements were embedded with {stored_model!r}, but config.yaml "
            f"says {cfg.llm.embedding_model!r}. Scores against a different model's vectors "
            "are meaningless -- re-embed with `chatmem extract --force`.",
            err=True,
        )
        raise typer.Exit(code=1)

    statements = store.list_statements(conn, person_id=target_person_id, thread_id=thread)
    embedded = [s for s in statements if s.embedding is not None]
    if not embedded:
        typer.echo("No embedded statements found. Run `chatmem extract` first.")
        raise typer.Exit(code=0)

    # A statement covers a span, so it matches a window if the two overlap.
    if since_ts is not None:
        embedded = [s for s in embedded if s.end_ts >= since_ts]
    if until_ts is not None:
        embedded = [s for s in embedded if s.start_ts <= until_ts]
    if not embedded:
        typer.echo("No statements fall in that range.")
        return

    # Cosine similarity zips the two vectors, so vectors of different lengths
    # would score against a truncated prefix instead of failing. That can only
    # happen if an `extract --force` after a model change was interrupted.
    dims = {len(s.embedding) for s in embedded}
    if len(dims) > 1:
        typer.echo(
            f"Error: stored statements have mixed embedding sizes {sorted(dims)}, so they "
            "were not all embedded with the same model. Re-embed with "
            "`chatmem extract --force`.",
            err=True,
        )
        raise typer.Exit(code=1)

    llm = LLMClient(cfg.llm)
    results = run_query(question, embedded, llm, limit=limit, min_score=min_score)

    if not results:
        typer.echo("No matching statements found.")
        return

    if answer:
        try:
            composed = llm.synthesize_answer(question, [s.text for s, _ in results])
        except LLMResponseError as e:
            typer.echo(f"Warning: could not compose an answer: {e}")
        else:
            typer.echo(composed)
            typer.echo("")

    for i, (statement, score) in enumerate(results, start=1):
        typer.echo(f"[{i}] [{score:.3f}] {statement.text}")
        typer.echo(f"    {statement.thread_id}  {statement.start_ts} .. {statement.end_ts}")
        for m in store.messages_by_ids(conn, statement.source_message_ids):
            typer.echo(f"    > {m.sender}: {m.text}")
        typer.echo("")


if __name__ == "__main__":
    app()
