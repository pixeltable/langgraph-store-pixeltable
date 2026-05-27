# langgraph-store-pixeltable

LangGraph [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) backend for [Pixeltable](https://github.com/pixeltable/pixeltable) — persistent, versioned, multimodal agent memory.

## Installation

```bash
pip install langgraph-store-pixeltable
```

## Quick Start

```python
from langgraph.store.pixeltable import PixeltableStore

store = PixeltableStore(table_name="agent_memory.items")
store.setup()

# Store a memory
store.put(("users", "alice"), "prefs", {"color": "blue", "language": "Python"})

# Retrieve it
item = store.get(("users", "alice"), "prefs")
print(item.value)  # {"color": "blue", "language": "Python"}

# List namespaces
namespaces = store.list_namespaces(prefix=("users",))
```

## Filtered Search

The `filter` parameter maps to Pixeltable's `.where()` clause — predicates are evaluated server-side, not in Python:

```python
store.put(("team", "eng"), "alice", {"role": "senior", "level": 5, "lang": "Python"})
store.put(("team", "eng"), "bob", {"role": "junior", "level": 2, "lang": "Go"})
store.put(("team", "eng"), "carol", {"role": "senior", "level": 4, "lang": "Python"})

# Exact match (shorthand)
results = store.search(("team",), filter={"role": "senior"})

# Comparison operators
results = store.search(("team",), filter={"level": {"$gt": 3}})
results = store.search(("team",), filter={"role": {"$ne": "junior"}})

# Multiple conditions (AND)
results = store.search(("team", "eng"), filter={"role": "senior", "lang": "Python"})
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`.

## Semantic Search

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

store = PixeltableStore(
    table_name="agent_memory.semantic",
    index={
        "dims": 384,
        "embed": lambda texts: model.encode(texts).tolist(),
        "fields": ["text"],
    },
)
store.setup()

store.put(("docs",), "d1", {"text": "Pixeltable handles multimodal data", "domain": "infra"})
store.put(("docs",), "d2", {"text": "LangGraph builds stateful agents", "domain": "framework"})
store.put(("docs",), "d3", {"text": "React is a frontend library", "domain": "frontend"})

# Semantic search
results = store.search(("docs",), query="multimodal AI pipelines", limit=2)
for r in results:
    print(f"{r.key}: {r.value['text']} (score={r.score:.3f})")

# Semantic search + filter (narrowed by domain)
results = store.search(
    ("docs",), query="data infrastructure", limit=3, filter={"domain": "infra"},
)
```

## With LangGraph Agents

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.pixeltable import PixeltableStore

store = PixeltableStore(table_name="agent_memory.items")
store.setup()

agent = create_react_agent(model, tools=tools, store=store)
```

## Access the Underlying Pixeltable Table

The `.table` property gives direct access to the Pixeltable table for operations beyond the Store interface — computed columns, lineage, version history, and arbitrary predicates:

```python
import pixeltable as pxt

t = store.table

# Inspect all data
t.select(t.key, t.value, t.created_at).collect()

# Add a computed column — auto-backfills all existing rows
t.add_computed_column(word_count=my_word_counter(t.value['text'].astype(pxt.String)))

# Add a classification UDF — runs on every new insert
t.add_computed_column(auto_domain=classify_domain(t.value['text'].astype(pxt.String)))

# WHERE on computed columns
results = t.where(t.auto_domain == 'infra').select(t.key, t.value).collect()

# Compound WHERE on multiple computed columns
results = (
    t.where((t.word_count > 5) & (t.auto_domain == 'infra'))
    .select(t.key, t.value, t.word_count, t.auto_domain)
    .collect()
)

# New inserts via the store auto-compute all lineage columns
store.put(("docs",), "d4", {"text": "Kubernetes orchestrates containers"})
# word_count and auto_domain are already computed for d4
```

## Why Pixeltable for LangGraph Memory?

- **Metadata filtering via `.where()`**: Filter on value fields with `$eq/$ne/$gt/$gte/$lt/$lte`, evaluated server-side
- **Computed column lineage**: Add derived columns that auto-backfill and auto-compute on new inserts
- **Persistent and versioned**: Data survives restarts; every update is tracked with full version history
- **Incremental**: Only new/changed rows get re-embedded
- **Multimodal native**: Images, video, audio, and documents alongside text via `.table`
- **Any embedding model**: Works with sentence-transformers, OpenAI, or any callable
- **No external services**: Embedded PostgreSQL, no Docker required

| Feature | PostgresStore | **PixeltableStore** |
|---------|---------------|---------------------|
| Persistence | Yes | Yes + versioned |
| Vector search | pgvector only | Any embedding model |
| Multimodal values | JSON only | Image, Video, Audio via `.table` |
| Incremental embedding | Manual | Automatic via computed columns |
| History | No | Full version history per row |
| Computed columns | No | Arbitrary UDFs with lineage |
| External service | Requires PostgreSQL + pgvector | Embedded, no Docker |

## Links

- [Pixeltable Docs](https://docs.pixeltable.com/)
- [LangGraph Integration Docs](https://docs.pixeltable.com/libraries/langgraph)
- [GitHub](https://github.com/pixeltable/pixeltable)
- [Discord](https://discord.gg/QPyqFYx2UN)
