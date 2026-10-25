"""Adversarial AABB identifier canonicalization regression."""

from enum import StrEnum

from physica.spatial import AABB, broad_phase_pairs, brute_force_pairs


def test_aabb_snapshots_builtin_string_from_overridden_subclass():
    class ReverseStr(str):
        def __lt__(self, other):
            return str(self) > str(other)

        def __gt__(self, other):
            return str(self) < str(other)

    # The input subclass reverses comparisons, but the stored ID must not.
    a = AABB(ReverseStr("a"), (0., 0.), (1., 1.))
    b = AABB(ReverseStr("b"), (0., 0.), (1., 1.))
    assert type(a.id) is type(b.id) is str
    assert (a.id, b.id) == ("a", "b")
    assert broad_phase_pairs((a, b)) == (("a", "b"),)
    assert brute_force_pairs((a, b)) == (("a", "b"),)


def test_strenum_identity_normalizes_without_changing_value_or_uniqueness():
    class Key(StrEnum):
        A = "a"
        B = "b"

    a = AABB(Key.A, (0., 0.), (1., 1.))
    b = AABB(Key.B, (0., 0.), (1., 1.))
    assert type(a.id) is type(b.id) is str
    assert (a.id, b.id) == ("a", "b")
    assert hash(a.id) == hash("a")
    assert hash(b.id) == hash("b")
    assert {a.id, b.id} == {"a", "b"}
    assert broad_phase_pairs((b, a)) == (("a", "b"),)


def test_custom_subclass_equality_and_hash_cannot_collapse_distinct_ids():
    class CollidingStr(str):
        def __hash__(self):
            return 0

        def __eq__(self, other):
            return isinstance(other, str)

    a = AABB(CollidingStr("a"), (0., 0.), (1., 1.))
    b = AABB(CollidingStr("b"), (0., 0.), (1., 1.))
    assert type(a.id) is type(b.id) is str
    assert {a.id, b.id} == {"a", "b"}
    assert broad_phase_pairs((a, b)) == (("a", "b"),)
