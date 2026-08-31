import pytest

from chatmem import topics
from chatmem.digest import UNCLASSIFIED_KEY, DigestMeta, group_by_topic, render
from chatmem.models import Message, Statement

META = DigestMeta(
    title="Alex Rivera",
    generated_at="2024-05-02T10:00:00.000000Z",
    message_count=1268,
    session_count=51,
    thread_count=1,
)


def _stmt(
    sid: int,
    text: str,
    topic: str | None = "family",
    start: str = "2023-04-02T18:20:01.000000Z",
    end: str | None = None,
    thread_id: str = "thread_alpha",
) -> Statement:
    return Statement(
        id=sid,
        person_id="target",
        session_id=1,
        thread_id=thread_id,
        text=text,
        source_message_ids=["m1"],
        start_ts=start,
        end_ts=end or start,
        created_at="2024-05-01T00:00:00.000000Z",
        topic=topic,
    )


def test_group_by_topic_follows_taxonomy_order_not_insertion_order():
    """Document order is the taxonomy's, so the digest stays diffable across
    runs even as statements arrive in a different order."""
    statements = [_stmt(1, "a", "plans"), _stmt(2, "b", "identity"), _stmt(3, "c", "work")]
    assert list(group_by_topic(statements)) == ["identity", "work", "plans"]


def test_group_by_topic_omits_empty_topics():
    buckets = group_by_topic([_stmt(1, "a", "family")])
    assert list(buckets) == ["family"]


def test_group_by_topic_sorts_within_a_section_chronologically():
    statements = [
        _stmt(1, "later", "family", start="2023-06-01T00:00:00.000000Z"),
        _stmt(2, "earlier", "family", start="2023-01-01T00:00:00.000000Z"),
    ]
    assert [s.text for s in group_by_topic(statements)["family"]] == ["earlier", "later"]


def test_untagged_statements_are_kept_under_unclassified_and_ordered_last():
    """A partly-failed classify pass must not make statements disappear from
    the digest -- completeness is the whole contract."""
    buckets = group_by_topic([_stmt(1, "unfiled", None), _stmt(2, "filed", "family")])
    assert list(buckets) == ["family", UNCLASSIFIED_KEY]
    assert [s.text for s in buckets[UNCLASSIFIED_KEY]] == ["unfiled"]


def test_topic_key_retired_from_the_taxonomy_still_renders():
    """An older database classified under a since-removed key must not crash
    the renderer or lose the row."""
    buckets = group_by_topic([_stmt(1, "legacy", "some_old_key")])
    assert [s.text for s in buckets["some_old_key"]] == ["legacy"]
    assert "legacy" in render(META, [_stmt(1, "legacy", "some_old_key")])


def test_every_statement_appears_exactly_once():
    statements = [
        _stmt(i, f"memory {i}", topics.TOPIC_KEYS[i % len(topics.TOPIC_KEYS)])
        for i in range(30)
    ]
    bullets = [line for line in render(META, statements).splitlines() if line.startswith("- **")]
    assert len(bullets) == len(statements)
    assert len(set(bullets)) == len(statements)


def test_render_includes_heading_counts_and_anchors():
    out = render(META, [_stmt(1, "Has a sister.", "family")])
    assert "## Family" in out
    assert "- [Family](#family) — 1" in out


def test_anchor_slug_matches_github_for_headings_with_punctuation():
    """'Identity & background' anchors as 'identity--background' on GitHub:
    the ampersand is dropped but the spaces around it are not."""
    out = render(META, [_stmt(1, "Born in Pune.", "identity")])
    assert "[Identity & background](#identity--background)" in out


def test_single_day_statement_shows_one_date_not_a_range():
    out = render(META, [_stmt(1, "Moved to Berlin.", "places")])
    assert "**2023-04-02** — Moved to Berlin." in out
    assert "–" not in out


def test_multi_day_statement_shows_a_range():
    out = render(
        META,
        [_stmt(1, "Was travelling.", "places", start="2023-04-02T00:00:00.000000Z", end="2023-04-05T00:00:00.000000Z")],
    )
    assert "**2023-04-02 – 2023-04-05**" in out


def test_statement_id_is_printed_so_it_cross_references_other_commands():
    out = render(META, [_stmt(17, "Moved to Berlin.", "places")])
    assert "`#17`" in out


def test_thread_is_tagged_only_when_asked():
    statements = [_stmt(1, "a", "places", thread_id="thread_alpha")]
    assert "`thread_alpha`" not in render(META, statements)
    assert "`thread_alpha`" in render(META, statements, show_threads=True)


def test_header_reports_the_real_span_of_the_statements():
    statements = [
        _stmt(1, "a", "family", start="2023-01-05T00:00:00.000000Z"),
        _stmt(2, "b", "family", start="2023-09-30T00:00:00.000000Z"),
    ]
    out = render(META, statements)
    assert "covering 2023-01-05 to 2023-09-30" in out
    assert "**2 memories** from 1,268 messages" in out


def test_render_with_no_statements_says_so_instead_of_producing_empty_sections():
    out = render(META, [])
    assert "No statements have been extracted yet" in out
    assert "## Contents" not in out


def test_quotes_are_included_only_when_supplied():
    statement = _stmt(1, "Moved to Berlin.", "places")
    message = Message(
        id="m1",
        thread_id="thread_alpha",
        sender="Alex Rivera",
        person_id="target",
        timestamp_utc="2023-04-02T18:20:01.000000Z",
        timestamp_ms=1,
        text="finally made the move\nto berlin last week",
        media_type=None,
        seq=0,
    )
    assert "Alex Rivera:" not in render(META, [statement])
    out = render(META, [statement], quotes_by_statement={1: [message]})
    # Newlines inside a quoted message would break out of the blockquote.
    assert "  > Alex Rivera: finally made the move to berlin last week" in out


def test_media_only_message_quotes_as_a_placeholder_not_none():
    statement = _stmt(1, "Sent a photo of the flat.", "places")
    message = Message(
        id="m1",
        thread_id="thread_alpha",
        sender="Alex Rivera",
        person_id="target",
        timestamp_utc="2023-04-02T18:20:01.000000Z",
        timestamp_ms=1,
        text=None,
        media_type="photo",
        seq=0,
    )
    out = render(META, [statement], quotes_by_statement={1: [message]})
    assert "> Alex Rivera: [photo]" in out
    assert "None" not in out


@pytest.mark.parametrize("topic", topics.TOPICS)
def test_every_taxonomy_topic_renders_a_heading_and_description(topic):
    out = render(META, [_stmt(1, "something", topic.key)])
    assert f"## {topic.title}" in out
    assert topic.description in out


def test_taxonomy_keys_are_unique():
    assert len(set(topics.TOPIC_KEYS)) == len(topics.TOPIC_KEYS)


def test_fallback_key_is_part_of_the_taxonomy():
    assert topics.FALLBACK_KEY in topics.BY_KEY
