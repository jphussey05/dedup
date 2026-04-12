"""Tests for Union-Find and duplicate group formation."""

from dedup.comparator import UnionFind


def test_union_find_basic():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(1, 2)

    assert uf.find(0) == uf.find(3)
    assert uf.find(0) != uf.find(4)


def test_union_find_groups():
    uf = UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)

    groups = uf.groups()
    # Should have 3 groups: {0,1,2}, {3,4}, {5}
    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    assert len(multi_groups) == 2


def test_union_find_singleton():
    uf = UnionFind(3)
    groups = uf.groups()
    assert len(groups) == 3
    for members in groups.values():
        assert len(members) == 1
