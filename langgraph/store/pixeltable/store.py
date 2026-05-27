"""PixeltableStore — LangGraph BaseStore backed by Pixeltable."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pixeltable as pxt

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from langgraph.store.base.embed import ensure_embeddings, get_text_at_path, tokenize_path

logger = logging.getLogger(__name__)

_NS_SEP = '.'


def _ns_to_str(ns: tuple[str, ...]) -> str:
    return _NS_SEP.join(ns)


def _str_to_ns(s: str) -> tuple[str, ...]:
    if not s:
        return ()
    return tuple(s.split(_NS_SEP))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


class PixeltableStore(BaseStore):
    """LangGraph BaseStore backed by Pixeltable.

    Provides persistent, versioned agent memory with optional semantic search
    and full access to Pixeltable's computed-column lineage via the ``.table``
    property.

    Args:
        table_name: Dot-separated Pixeltable table path (e.g. ``"agent_memory.items"``).
        index: Optional :class:`IndexConfig` for semantic search (dims, embed, fields).

    Example::

        from langgraph.store.pixeltable import PixeltableStore
        store = PixeltableStore(table_name="agent_memory.items")
        store.setup()
        store.put(("users",), "u1", {"name": "Alice", "bio": "Loves AI"})
        item = store.get(("users",), "u1")
    """

    def __init__(
        self,
        *,
        table_name: str = 'langgraph_store.items',
        index: IndexConfig | None = None,
    ) -> None:
        self._table_name = table_name
        self._local = threading.local()
        self.index_config = index
        self.embeddings = None
        self._tokenized_fields: list[tuple[str, Any]] | None = None

        if self.index_config:
            self.index_config = self.index_config.copy()
            self.embeddings = ensure_embeddings(self.index_config.get('embed'))
            self._tokenized_fields = [
                (p, tokenize_path(p)) if p != '$' else (p, p)
                for p in (self.index_config.get('fields') or ['$'])
            ]

    def setup(self) -> None:
        """Create the Pixeltable directory and table if they don't exist."""
        parts = self._table_name.rsplit('.', 1)
        if len(parts) == 2:
            dir_name = parts[0]
            pxt.create_dir(dir_name, if_exists='ignore')

        schema: dict[str, Any] = {
            'namespace': pxt.String,
            'key': pxt.String,
            'value': pxt.Json,
            'created_at': pxt.Timestamp,
            'updated_at': pxt.Timestamp,
        }
        if self.index_config:
            schema['search_text'] = pxt.String
            dims = self.index_config['dims']
            schema['embedding'] = pxt.Array[(dims,), pxt.Float]

        self._local.table = pxt.create_table(self._table_name, schema, if_exists='ignore')

    @property
    def table(self) -> pxt.Table:
        """Access the underlying Pixeltable table for computed columns, lineage, etc.

        Returns a thread-local handle so the store is safe to use from
        LangGraph's threaded tool execution.
        """
        t = getattr(self._local, 'table', None)
        if t is None:
            try:
                t = pxt.get_table(self._table_name)
                self._local.table = t
            except Exception:
                raise RuntimeError(
                    f'Table {self._table_name!r} not found. Call store.setup() first.'
                )
        return t

    # ------------------------------------------------------------------
    # BaseStore abstract interface
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        put_ops: dict[tuple[str, str], PutOp] = {}
        search_ops: dict[int, SearchOp] = {}

        for i, op in enumerate(ops):
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, PutOp):
                put_ops[(op.namespace, op.key)] = op
                results.append(None)
            elif isinstance(op, SearchOp):
                search_ops[i] = op
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._handle_list_namespaces(op))
            else:
                raise ValueError(f'Unknown operation type: {type(op)}')

        if put_ops:
            self._handle_put_batch(put_ops)

        if search_ops:
            for idx, op in search_ops.items():
                results[idx] = self._handle_search(op)

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return await loop.run_in_executor(pool, self.batch, ops)

    # ------------------------------------------------------------------
    # Op handlers
    # ------------------------------------------------------------------

    def _handle_get(self, op: GetOp) -> Item | None:
        t = self.table
        ns_str = _ns_to_str(op.namespace)
        rows = (
            t.where((t.namespace == ns_str) & (t.key == op.key))
            .select(t.namespace, t.key, t.value, t.created_at, t.updated_at)
            .collect()
        )
        if len(rows) == 0:
            return None
        row = rows[0]
        return Item(
            value=row['value'],
            key=row['key'],
            namespace=_str_to_ns(row['namespace']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    def _handle_put_batch(self, put_ops: dict[tuple[str, str], PutOp]) -> None:
        t = self.table
        now = datetime.now(timezone.utc)

        texts_to_embed: dict[str, list[tuple[tuple[str, ...], str, str]]] = defaultdict(list)
        if self.embeddings and self._tokenized_fields:
            for op in put_ops.values():
                if op.value is not None and op.index is not False:
                    paths = self._tokenized_fields if op.index is None else [(ix, tokenize_path(ix)) for ix in op.index]
                    for path, field in paths:
                        texts = get_text_at_path(op.value, field)
                        if texts:
                            if len(texts) > 1:
                                for ti, text in enumerate(texts):
                                    texts_to_embed[text].append((op.namespace, op.key, f'{path}.{ti}'))
                            else:
                                texts_to_embed[texts[0]].append((op.namespace, op.key, path))

        embedded: dict[str, list[float]] = {}
        if texts_to_embed:
            unique_texts = list(texts_to_embed.keys())
            vectors = self.embeddings.embed_documents(unique_texts)
            embedded = dict(zip(unique_texts, vectors))

        # Group embeddings by (namespace, key) -> aggregate vector (average)
        key_embeddings: dict[tuple[tuple[str, ...], str], list[list[float]]] = defaultdict(list)
        for text, entries in texts_to_embed.items():
            vec = embedded[text]
            for ns, key, _path in entries:
                key_embeddings[(ns, key)].append(vec)

        for (ns, key), op in put_ops.items():
            ns_str = _ns_to_str(ns)
            # Delete existing row (upsert pattern)
            try:
                t.delete(where=(t.namespace == ns_str) & (t.key == key))
            except Exception:
                pass

            if op.value is None:
                continue

            row: dict[str, Any] = {
                'namespace': ns_str,
                'key': key,
                'value': op.value,
                'created_at': now,
                'updated_at': now,
            }

            if self.index_config:
                search_parts: list[str] = []
                if op.index is not False and self._tokenized_fields:
                    paths = self._tokenized_fields if op.index is None else [(ix, tokenize_path(ix)) for ix in op.index]
                    for _path, field in paths:
                        search_parts.extend(get_text_at_path(op.value, field))
                row['search_text'] = ' '.join(search_parts) if search_parts else ''

                vecs = key_embeddings.get((ns, key))
                if vecs:
                    avg = np.mean(vecs, axis=0).tolist()
                    row['embedding'] = avg
                else:
                    row['embedding'] = [0.0] * self.index_config['dims']

            t.insert([row])

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        t = self.table
        ns_prefix = _ns_to_str(op.namespace_prefix)

        # Build where clause: namespace prefix
        if ns_prefix:
            prefix_str = ns_prefix + _NS_SEP
            condition = t.namespace.startswith(prefix_str) | (t.namespace == ns_prefix)
        else:
            condition = None

        # Apply filters on value JSON field
        if op.filter:
            for field_key, filter_value in op.filter.items():
                fc = self._build_filter_condition(t, field_key, filter_value)
                if fc is not None:
                    condition = fc if condition is None else (condition & fc)

        select_cols = [t.namespace, t.key, t.value, t.created_at, t.updated_at]
        has_embeddings = self.index_config and self.embeddings

        if has_embeddings:
            select_cols.append(t.embedding)

        query = t.where(condition) if condition is not None else t
        rows = query.select(*select_cols).collect()

        if op.query and has_embeddings:
            query_vec = self.embeddings.embed_query(op.query)
            scored: list[tuple[dict[str, Any], float]] = []
            for row in rows:
                emb = row.get('embedding')
                if emb is not None and any(v != 0.0 for v in emb):
                    score = _cosine_similarity(query_vec, emb)
                else:
                    score = 0.0
                scored.append((row, score))
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored = [(row, 0.0) for row in rows]

        # Apply offset and limit
        page = scored[op.offset: op.offset + op.limit]

        return [
            SearchItem(
                namespace=_str_to_ns(row['namespace']),
                key=row['key'],
                value=row['value'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                score=score,
            )
            for row, score in page
        ]

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        t = self.table
        rows = t.select(t.namespace).collect()
        all_ns = {_str_to_ns(row['namespace']) for row in rows}

        filtered = all_ns
        if op.match_conditions:
            filtered = {ns for ns in filtered if all(self._match(cond, ns) for cond in op.match_conditions)}

        if op.max_depth is not None:
            filtered = {ns[: op.max_depth] for ns in filtered}

        result = sorted(filtered)
        return result[op.offset: op.offset + op.limit]

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter_condition(t: pxt.Table, field_key: str, filter_value: Any) -> Any:
        """Translate LangGraph filter dict to a Pixeltable where expression."""
        col = t.value[field_key]

        if isinstance(filter_value, dict) and any(k.startswith('$') for k in filter_value):
            conditions = []
            for op_key, op_val in filter_value.items():
                if op_key == '$eq':
                    conditions.append(col == op_val)
                elif op_key == '$ne':
                    conditions.append(col != op_val)
                elif op_key == '$gt':
                    conditions.append(col > op_val)
                elif op_key == '$gte':
                    conditions.append(col >= op_val)
                elif op_key == '$lt':
                    conditions.append(col < op_val)
                elif op_key == '$lte':
                    conditions.append(col <= op_val)
                else:
                    raise ValueError(f'Unsupported filter operator: {op_key}')
            result = conditions[0]
            for c in conditions[1:]:
                result = result & c
            return result
        else:
            return col == filter_value

    @staticmethod
    def _match(condition: MatchCondition, ns: tuple[str, ...]) -> bool:
        if condition.match_type == 'prefix':
            return ns[: len(condition.path)] == condition.path
        elif condition.match_type == 'suffix':
            return ns[-len(condition.path):] == condition.path if len(ns) >= len(condition.path) else False
        return True
