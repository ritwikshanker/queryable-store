"""The fixed taxonomy statements are filed under in `chatmem digest`.

Deliberately a closed list rather than topics discovered from the data: the
digest is meant to be re-read and diffed over time, so a statement added next
month should land under the same heading it would have last month. The keys
are stored in statements.topic and must stay stable -- renaming one orphans
every row already classified with it (run `chatmem digest --reclassify` after
any change here).

Order is document order: who they are, then the people around them, then the
outward circumstances of their life, then the inward ones, then the
relationship, then things anchored in time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    # Shown to the classifier verbatim, and used as the section's lead line in
    # the rendered markdown -- so it has to read as a definition, not a hint.
    description: str


TOPICS: tuple[Topic, ...] = (
    Topic(
        "identity",
        "Identity & background",
        "Who they are: age, birthday, where they are from, nationality, languages, "
        "how they describe themselves.",
    ),
    Topic(
        "family",
        "Family",
        "Parents, siblings, relatives, upbringing, and family circumstances.",
    ),
    Topic(
        "people",
        "People in their life",
        "Friends, colleagues, exes, pets, and other named people -- anyone who is "
        "not family and not the person they are talking to.",
    ),
    Topic(
        "places",
        "Places",
        "Where they live, have lived, have moved to or from, and places they have "
        "travelled to or want to.",
    ),
    Topic(
        "work",
        "Work & study",
        "Job, employer, career, studies, degrees, exams, money, and professional "
        "ambitions.",
    ),
    Topic(
        "health",
        "Health & wellbeing",
        "Physical health, illness, injury, sleep, diet as it bears on health, "
        "mental health, therapy, and medication.",
    ),
    Topic(
        "tastes",
        "Tastes & preferences",
        "What they like and dislike: food, music, films, books, games, clothes, "
        "aesthetics.",
    ),
    Topic(
        "routines",
        "Habits & routines",
        "Recurring behaviour: daily rhythms, hobbies practised regularly, sports, "
        "commutes, rituals.",
    ),
    Topic(
        "beliefs",
        "Beliefs & values",
        "Opinions, politics, religion, ethics, and how they think the world or "
        "other people work.",
    ),
    Topic(
        "feelings",
        "Feelings & inner life",
        "Emotional states they report about themselves: what they fear, want, "
        "regret, enjoy, or struggle with.",
    ),
    Topic(
        "relationship",
        "The relationship",
        "Statements about the relationship with the person they are talking to: "
        "how they feel about them, milestones, conflicts, and shared history.",
    ),
    Topic(
        "events",
        "Things that happened",
        "One-off episodes and incidents in their life, anchored to a point in "
        "time, that do not belong under a more specific heading.",
    ),
    Topic(
        "plans",
        "Plans & intentions",
        "What they intend, expect, or have arranged to do in the future.",
    ),
    Topic(
        "other",
        "Other",
        "Anything that genuinely does not fit any heading above. Prefer a real "
        "heading whenever one applies.",
    ),
)

TOPIC_KEYS: tuple[str, ...] = tuple(t.key for t in TOPICS)

BY_KEY: dict[str, Topic] = {t.key: t for t in TOPICS}

FALLBACK_KEY = "other"


def title_for(key: str | None) -> str:
    """Heading for a topic key, tolerating keys retired from the taxonomy so an
    older database still renders instead of raising."""
    if key is None:
        return "Unclassified"
    topic = BY_KEY.get(key)
    return topic.title if topic is not None else key


def catalog() -> str:
    """The taxonomy as the classifier prompt presents it."""
    return "\n".join(f"- {t.key}: {t.description}" for t in TOPICS)
