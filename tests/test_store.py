"""Tests for PixeltableStore — CRUD, namespaces, filters, edge cases."""

from __future__ import annotations

import uuid

import pixeltable as pxt
import pytest

from langgraph.store.pixeltable import PixeltableStore


def _unique_table() -> str:
    uid = uuid.uuid4().hex[:8]
    return f"test_lg_{uid}.items"


@pytest.fixture()
def store():
    name = _unique_table()
    s = PixeltableStore(table_name=name)
    s.setup()
    yield s
    pxt.drop_table(name, if_not_exists="ignore")
    pxt.drop_dir(name.rsplit(".", 1)[0], if_not_exists="ignore")


# ------------------------------------------------------------------
# CRUD basics
# ------------------------------------------------------------------


class TestCRUD:
    def test_put_and_get(self, store: PixeltableStore):
        store.put(("users",), "u1", {"name": "Alice", "score": 42})
        item = store.get(("users",), "u1")
        assert item is not None
        assert item.value == {"name": "Alice", "score": 42}
        assert item.key == "u1"
        assert item.namespace == ("users",)

    def test_get_missing_returns_none(self, store: PixeltableStore):
        assert store.get(("x",), "nope") is None

    def test_put_overwrite(self, store: PixeltableStore):
        store.put(("a",), "k", {"v": 1})
        store.put(("a",), "k", {"v": 2})
        item = store.get(("a",), "k")
        assert item is not None
        assert item.value == {"v": 2}

    def test_delete(self, store: PixeltableStore):
        store.put(("a",), "k", {"v": 1})
        assert store.get(("a",), "k") is not None
        store.delete(("a",), "k")
        assert store.get(("a",), "k") is None

    def test_delete_nonexistent(self, store: PixeltableStore):
        store.delete(("x",), "nope")

    def test_put_multiple_items(self, store: PixeltableStore):
        store.put(("a",), "k1", {"v": 1})
        store.put(("a",), "k2", {"v": 2})
        store.put(("b",), "k3", {"v": 3})
        assert store.get(("a",), "k1") is not None
        assert store.get(("a",), "k2") is not None
        assert store.get(("b",), "k3") is not None

    def test_timestamps(self, store: PixeltableStore):
        store.put(("a",), "k", {"v": 1})
        item = store.get(("a",), "k")
        assert item is not None
        assert item.created_at is not None
        assert item.updated_at is not None


# ------------------------------------------------------------------
# Search and filtering
# ------------------------------------------------------------------


class TestSearch:
    def test_search_all(self, store: PixeltableStore):
        store.put(("docs",), "d1", {"type": "article"})
        store.put(("docs",), "d2", {"type": "paper"})
        results = store.search(("docs",))
        assert len(results) == 2

    def test_search_namespace_prefix(self, store: PixeltableStore):
        store.put(("docs", "a"), "d1", {"v": 1})
        store.put(("docs", "b"), "d2", {"v": 2})
        store.put(("other",), "d3", {"v": 3})
        results = store.search(("docs",))
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"d1", "d2"}

    def test_search_exact_namespace(self, store: PixeltableStore):
        store.put(("docs",), "d1", {"v": 1})
        store.put(("docs", "sub"), "d2", {"v": 2})
        results = store.search(("docs",))
        assert len(results) == 2

    def test_search_filter_eq(self, store: PixeltableStore):
        store.put(("docs",), "d1", {"type": "article", "status": "published"})
        store.put(("docs",), "d2", {"type": "paper", "status": "draft"})
        results = store.search(("docs",), filter={"type": "article"})
        assert len(results) == 1
        assert results[0].key == "d1"

    def test_search_filter_operators(self, store: PixeltableStore):
        store.put(("scores",), "s1", {"score": 10})
        store.put(("scores",), "s2", {"score": 20})
        store.put(("scores",), "s3", {"score": 30})

        results = store.search(("scores",), filter={"score": {"$gt": 15}})
        keys = {r.key for r in results}
        assert "s2" in keys
        assert "s3" in keys
        assert "s1" not in keys

    def test_search_filter_ne(self, store: PixeltableStore):
        store.put(("docs",), "d1", {"type": "article"})
        store.put(("docs",), "d2", {"type": "paper"})
        results = store.search(("docs",), filter={"type": {"$ne": "paper"}})
        assert len(results) == 1
        assert results[0].key == "d1"

    def test_search_limit_offset(self, store: PixeltableStore):
        for i in range(5):
            store.put(("items",), f"i{i}", {"v": i})
        all_results = store.search(("items",), limit=100)
        assert len(all_results) == 5

        page1 = store.search(("items",), limit=2, offset=0)
        assert len(page1) == 2

        page2 = store.search(("items",), limit=2, offset=2)
        assert len(page2) == 2

    def test_search_empty_namespace(self, store: PixeltableStore):
        store.put(("a",), "k1", {"v": 1})
        store.put(("b",), "k2", {"v": 2})
        results = store.search(())
        assert len(results) == 2


# ------------------------------------------------------------------
# Namespaces
# ------------------------------------------------------------------


class TestNamespaces:
    def test_list_namespaces(self, store: PixeltableStore):
        store.put(("a", "b"), "k1", {"v": 1})
        store.put(("a", "c"), "k2", {"v": 2})
        store.put(("d",), "k3", {"v": 3})
        ns = store.list_namespaces()
        assert ("a", "b") in ns
        assert ("a", "c") in ns
        assert ("d",) in ns

    def test_list_namespaces_prefix(self, store: PixeltableStore):
        store.put(("a", "b"), "k1", {"v": 1})
        store.put(("a", "c"), "k2", {"v": 2})
        store.put(("d",), "k3", {"v": 3})
        ns = store.list_namespaces(prefix=("a",))
        assert ("a", "b") in ns
        assert ("a", "c") in ns
        assert ("d",) not in ns

    def test_list_namespaces_suffix(self, store: PixeltableStore):
        store.put(("a", "b"), "k1", {"v": 1})
        store.put(("x", "b"), "k2", {"v": 2})
        store.put(("c",), "k3", {"v": 3})
        ns = store.list_namespaces(suffix=("b",))
        assert ("a", "b") in ns
        assert ("x", "b") in ns
        assert ("c",) not in ns

    def test_list_namespaces_max_depth(self, store: PixeltableStore):
        store.put(("a", "b", "c"), "k1", {"v": 1})
        store.put(("a", "b", "d"), "k2", {"v": 2})
        ns = store.list_namespaces(max_depth=2)
        assert all(len(n) <= 2 for n in ns)
        assert ("a", "b") in ns

    def test_list_namespaces_limit_offset(self, store: PixeltableStore):
        for i in range(5):
            store.put((f"ns{i}",), f"k{i}", {"v": i})
        all_ns = store.list_namespaces()
        assert len(all_ns) == 5
        page = store.list_namespaces(limit=2, offset=1)
        assert len(page) == 2


# ------------------------------------------------------------------
# Table property / escape hatch
# ------------------------------------------------------------------


class TestTableProperty:
    def test_table_access(self, store: PixeltableStore):
        t = store.table
        assert t is not None

    def test_cross_check_via_pxt(self, store: PixeltableStore):
        store.put(("x",), "k", {"v": 1})
        t = store.table
        rows = t.select(t.namespace, t.key, t.value).collect()
        assert len(rows) == 1
        assert rows[0]["value"] == {"v": 1}


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_value_dict(self, store: PixeltableStore):
        store.put(("a",), "k", {})
        item = store.get(("a",), "k")
        assert item is not None
        assert item.value == {}

    def test_nested_value(self, store: PixeltableStore):
        val = {"metadata": {"tags": ["ai", "ml"], "nested": {"deep": True}}}
        store.put(("a",), "k", val)
        item = store.get(("a",), "k")
        assert item is not None
        assert item.value == val

    def test_single_component_namespace(self, store: PixeltableStore):
        store.put(("ns",), "k", {"v": 1})
        item = store.get(("ns",), "k")
        assert item is not None

    def test_deep_namespace(self, store: PixeltableStore):
        ns = ("a", "b", "c", "d", "e")
        store.put(ns, "k", {"v": 1})
        item = store.get(ns, "k")
        assert item is not None
        assert item.namespace == ns
