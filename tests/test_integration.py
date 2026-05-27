"""Live integration tests — semantic search, agent usage, lineage.

Requires: sentence-transformers, langgraph
"""

from __future__ import annotations

import uuid

import pixeltable as pxt
import pytest
from sentence_transformers import SentenceTransformer

from langgraph.store.pixeltable import PixeltableStore


def _unique_table() -> str:
    uid = uuid.uuid4().hex[:8]
    return f'test_lg_int_{uid}.items'


_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def _embed_fn(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts).tolist()


@pytest.fixture()
def semantic_store():
    name = _unique_table()
    s = PixeltableStore(
        table_name=name,
        index={
            'dims': 384,
            'embed': _embed_fn,
            'fields': ['text'],
        },
    )
    s.setup()
    yield s
    pxt.drop_table(name, if_not_exists='ignore')
    pxt.drop_dir(name.rsplit('.', 1)[0], if_not_exists='ignore')


@pytest.fixture()
def plain_store():
    name = _unique_table()
    s = PixeltableStore(table_name=name)
    s.setup()
    yield s
    pxt.drop_table(name, if_not_exists='ignore')
    pxt.drop_dir(name.rsplit('.', 1)[0], if_not_exists='ignore')


# ------------------------------------------------------------------
# 1. CRUD round-trip
# ------------------------------------------------------------------

class TestCRUDRoundTrip:
    def test_put_get_delete(self, plain_store: PixeltableStore):
        store = plain_store
        store.put(('users', 'alice'), 'prefs', {'color': 'blue', 'lang': 'Python'})
        item = store.get(('users', 'alice'), 'prefs')
        assert item is not None
        assert item.value['color'] == 'blue'
        assert item.namespace == ('users', 'alice')

        store.put(('users', 'alice'), 'prefs', {'color': 'green', 'lang': 'Rust'})
        item = store.get(('users', 'alice'), 'prefs')
        assert item.value['color'] == 'green'

        store.delete(('users', 'alice'), 'prefs')
        assert store.get(('users', 'alice'), 'prefs') is None


# ------------------------------------------------------------------
# 2. Namespace operations
# ------------------------------------------------------------------

class TestNamespaceLive:
    def test_list_prefix_suffix_depth(self, plain_store: PixeltableStore):
        store = plain_store
        store.put(('team', 'eng', 'backend'), 'k1', {'v': 1})
        store.put(('team', 'eng', 'frontend'), 'k2', {'v': 2})
        store.put(('team', 'design'), 'k3', {'v': 3})
        store.put(('personal',), 'k4', {'v': 4})

        all_ns = store.list_namespaces()
        assert len(all_ns) == 4

        team_ns = store.list_namespaces(prefix=('team',))
        assert len(team_ns) == 3
        assert ('personal',) not in team_ns

        eng_ns = store.list_namespaces(prefix=('team', 'eng'))
        assert len(eng_ns) == 2

        depth2 = store.list_namespaces(prefix=('team',), max_depth=2)
        assert all(len(ns) <= 2 for ns in depth2)
        assert ('team', 'eng') in depth2
        assert ('team', 'design') in depth2


# ------------------------------------------------------------------
# 3. Filter operators
# ------------------------------------------------------------------

class TestFilterLive:
    def test_search_with_filters(self, plain_store: PixeltableStore):
        store = plain_store
        store.put(('products',), 'p1', {'name': 'Widget', 'price': 10, 'category': 'tools'})
        store.put(('products',), 'p2', {'name': 'Gadget', 'price': 25, 'category': 'electronics'})
        store.put(('products',), 'p3', {'name': 'Doohickey', 'price': 5, 'category': 'tools'})

        # Exact match
        tools = store.search(('products',), filter={'category': 'tools'})
        assert len(tools) == 2

        # Greater-than
        expensive = store.search(('products',), filter={'price': {'$gt': 8}})
        assert len(expensive) == 2
        keys = {r.key for r in expensive}
        assert 'p1' in keys and 'p2' in keys

        # Less-than-or-equal
        cheap = store.search(('products',), filter={'price': {'$lte': 10}})
        assert len(cheap) == 2

        # Not-equal
        not_tools = store.search(('products',), filter={'category': {'$ne': 'tools'}})
        assert len(not_tools) == 1
        assert not_tools[0].key == 'p2'


# ------------------------------------------------------------------
# 4. Semantic search with local embeddings
# ------------------------------------------------------------------

class TestSemanticSearch:
    def test_basic_semantic_search(self, semantic_store: PixeltableStore):
        store = semantic_store
        docs = [
            ('d1', 'Pixeltable provides declarative data infrastructure for multimodal AI'),
            ('d2', 'LangGraph enables building stateful multi-actor agents'),
            ('d3', 'Python is a great programming language for data science'),
            ('d4', 'Vector databases store embeddings for similarity search'),
            ('d5', 'Computed columns in Pixeltable track data lineage automatically'),
        ]
        for key, text in docs:
            store.put(('knowledge',), key, {'text': text, 'source': 'test'})

        results = store.search(('knowledge',), query='multimodal data pipelines', limit=3)
        assert len(results) <= 3
        assert results[0].score > 0
        # Pixeltable-related docs should rank highest
        top_keys = {r.key for r in results[:2]}
        assert 'd1' in top_keys or 'd5' in top_keys

    def test_semantic_search_with_filter(self, semantic_store: PixeltableStore):
        store = semantic_store
        store.put(('docs',), 'd1', {'text': 'AI and machine learning', 'type': 'article'})
        store.put(('docs',), 'd2', {'text': 'AI in healthcare', 'type': 'paper'})
        store.put(('docs',), 'd3', {'text': 'Cooking recipes', 'type': 'article'})

        results = store.search(
            ('docs',), query='artificial intelligence', filter={'type': 'article'}, limit=5
        )
        keys = {r.key for r in results}
        assert 'd2' not in keys  # paper filtered out
        assert 'd1' in keys

    def test_search_without_query_returns_all(self, semantic_store: PixeltableStore):
        store = semantic_store
        store.put(('items',), 'i1', {'text': 'hello'})
        store.put(('items',), 'i2', {'text': 'world'})
        results = store.search(('items',))
        assert len(results) == 2


# ------------------------------------------------------------------
# 5. Cross-check: store data visible via pxt.get_table()
# ------------------------------------------------------------------

class TestCrossCheck:
    def test_data_visible_via_pixeltable(self, plain_store: PixeltableStore):
        store = plain_store
        store.put(('cross',), 'ck', {'msg': 'visible via pixeltable'})

        t = store.table
        rows = t.select(t.namespace, t.key, t.value).collect()
        assert len(rows) == 1
        assert rows[0]['value']['msg'] == 'visible via pixeltable'


# ------------------------------------------------------------------
# 6. .table escape hatch — computed columns and lineage
# ------------------------------------------------------------------

class TestTableLineage:
    def test_computed_column_lineage(self, plain_store: PixeltableStore):
        store = plain_store
        store.put(('docs',), 'd1', {'text': 'Hello World', 'lang': 'en'})
        store.put(('docs',), 'd2', {'text': 'Bonjour le monde', 'lang': 'fr'})

        t = store.table
        t.add_computed_column(ns_upper=t.namespace.upper(), if_exists='ignore')

        rows = t.select(t.key, t.namespace, t.ns_upper).collect()
        assert len(rows) == 2
        for row in rows:
            assert row['ns_upper'] == row['namespace'].upper()

        # Lineage is maintained: inserting new data auto-computes
        store.put(('docs',), 'd3', {'text': 'Hola Mundo', 'lang': 'es'})
        rows = t.where(t.key == 'd3').select(t.key, t.ns_upper).collect()
        assert len(rows) == 1
        assert rows[0]['ns_upper'] == 'DOCS'
