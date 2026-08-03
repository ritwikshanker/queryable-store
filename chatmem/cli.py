"""chatmem CLI: ingest, identities, relink, stats, sessions, extract, query."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from chatmem import store
from chatmem.config import Config, load_config
from chatmem.extract import extract_session
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


def _ms_to_iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    # Explicit microsecond formatting (not dt.isoformat()) so every
    # timestamp has the same length and sorts correctly as plain text,
    # regardless of whether the microsecond component happens to be zero.
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


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
    dropped_by_reason: dict[str, int] = defaultdict(int)

    for thread in parser.parse(path):
        n_threads += 1
        store.upsert_thread(
            conn, thread.thread_id, thread.title, thread.participants, thread.source, ingested_at
        )

        normalized: list[Message] = []
        for seq, raw in enumerate(thread.messages):
            person_id = resolver.resolve(raw.sender)
            normalized.append(
                Message(
                    id=_message_id(
                        thread.thread_id, raw.timestamp_ms, raw.sender, raw.text, raw.media_type, raw.ordinal
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
        # foreign key.
        store.save_identities(conn, resolver)
        store.replace_thread_messages(conn, thread.thread_id, normalized)
        n_kept += len(normalized)
        for reason, count in thread.dropped.items():
            dropped_by_reason[reason] += count

        ranges = sessionize(
            normalized,
            idle_gap_minutes=cfg.sessionize.idle_gap_minutes,
            max_messages=cfg.sessionize.max_messages,
        )
        messages_by_seq = {m.seq: m for m in normalized}
        saved = store.replace_thread_sessions(conn, thread.thread_id, ranges, messages_by_seq)
        n_sessions += len(saved)

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
    n = store.relink_messages(conn, resolver)
    typer.echo(f"relinked {n} messages")

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


@app.command()
def extract(
    thread: Optional[str] = typer.Option(None, "--thread"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    target: Optional[str] = typer.Option(
        None, "--target", help="Override config.yaml's target for this run."
    ),
    db: Optional[Path] = typer.Option(None, "--db"),
) -> None:
    """Run each session the target participated in through the LLM, storing statements."""
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

    display_name_by_person = {p.person_id: p.display_name for p in store.all_people(conn)}
    target_name = display_name_by_person.get(target_person_id, target_person_id)

    n_skipped = 0
    n_failed = 0
    n_extracted = 0
    n_processed = 0

    for session in store.list_sessions(conn, thread_id=thread):
        participants = store.session_participant_ids(conn, session.id)
        if target_person_id not in participants:
            n_skipped += 1
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

        store.replace_session_statements(conn, session.id, statements)
        n_processed += 1
        n_extracted += len(statements)

    conn.commit()

    typer.echo(
        f"sessions:   {n_processed} processed, {n_skipped} skipped (target absent), "
        f"{n_failed} failed"
    )
    typer.echo(f"statements: {n_extracted} extracted")


@app.command()
def query(
    question: str = typer.Argument(..., help="A natural-language question about the target."),
    limit: int = typer.Option(5, "--limit"),
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

    conn = store.connect(cfg.db_path)

    target_person_id = store.resolve_person(conn, cfg.target)
    if target_person_id is None:
        typer.echo(
            f"Error: target {cfg.target!r} was not found. Run `chatmem ingest` first.", err=True
        )
        raise typer.Exit(code=1)

    statements = store.list_statements(conn, person_id=target_person_id)
    embedded = [s for s in statements if s.embedding is not None]
    if not embedded:
        typer.echo("No embedded statements found. Run `chatmem extract` first.")
        raise typer.Exit(code=0)

    llm = LLMClient(cfg.llm)
    results = run_query(question, embedded, llm, limit=limit)

    if not results:
        typer.echo("No matching statements found.")
        return

    for statement, score in results:
        typer.echo(f"[{score:.3f}] {statement.text}")
        typer.echo(f"    {statement.thread_id}  {statement.start_ts} .. {statement.end_ts}")
        for m in store.messages_by_ids(conn, statement.source_message_ids):
            typer.echo(f"    > {m.sender}: {m.text}")
        typer.echo("")


if __name__ == "__main__":
    app()
