import pytest

from chatmem.config import ConfigError, IdentitySpec
from chatmem.identity import IdentityResolver


def test_resolves_multiple_aliases_to_one_person():
    resolver = IdentityResolver(
        [IdentitySpec(id="target", display_name="Alex Rivera", aliases=["Alex Rivera", "Alex R."])]
    )
    assert resolver.resolve("Alex Rivera") == "target"
    assert resolver.resolve("Alex R.") == "target"


def test_display_name_is_an_implicit_alias():
    resolver = IdentityResolver([IdentitySpec(id="target", display_name="Alex Rivera", aliases=[])])
    assert resolver.resolve("Alex Rivera") == "target"


def test_alias_matching_is_case_and_whitespace_insensitive():
    resolver = IdentityResolver(
        [IdentitySpec(id="target", display_name="Alex Rivera", aliases=["Alex Rivera"])]
    )
    assert resolver.resolve("  alex   rivera ") == "target"


def test_unmapped_sender_auto_creates_with_warning():
    resolver = IdentityResolver([])
    pid = resolver.resolve("Sam Chen")
    assert pid  # some stable id was generated
    new = resolver.new_senders()
    assert len(new) == 1
    assert new[0].raw_name == "Sam Chen"
    assert new[0].person_id == pid


def test_auto_created_person_is_stable_within_a_run():
    resolver = IdentityResolver([])
    first = resolver.resolve("Sam Chen")
    second = resolver.resolve("sam chen")  # same person, different casing
    assert first == second
    assert len(resolver.new_senders()) == 1


def test_duplicate_alias_across_two_ids_is_a_config_error():
    with pytest.raises(ConfigError):
        IdentityResolver(
            [
                IdentitySpec(id="a", display_name="A", aliases=["Shared Name"]),
                IdentitySpec(id="b", display_name="B", aliases=["Shared Name"]),
            ]
        )


def test_resolve_target_by_id_or_alias():
    resolver = IdentityResolver(
        [IdentitySpec(id="target", display_name="Alex Rivera", aliases=["Alex Rivera", "Alex R."])]
    )
    assert resolver.resolve_target("target") == "target"
    assert resolver.resolve_target("Alex R.") == "target"
    assert resolver.resolve_target("Alex Rivera") == "target"


def test_resolve_target_miss_returns_none_not_error():
    resolver = IdentityResolver([])
    assert resolver.resolve_target("Nobody Declared") is None


def test_new_senders_yaml_is_paste_ready():
    resolver = IdentityResolver([])
    resolver.resolve("Sam Chen")
    yaml_block = resolver.new_senders_yaml()
    assert "id:" in yaml_block
    assert "Sam Chen" in yaml_block
    assert "aliases:" in yaml_block
