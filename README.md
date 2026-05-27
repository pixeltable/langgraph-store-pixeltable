# langgraph-store-pixeltable

LangGraph [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) backend for [Pixeltable](https://github.com/pixeltable/pixeltable) — persistent, versioned, multimodal agent memory.

## Why Pixeltable for LangGraph Memory?

| Feature | PostgresStore | **PixeltableStore** |
|---------|---------------|---------------------|
| Persistence | Yes | Yes + versioned |
| Vector search | pgvector only | Any embedding model |
| Multimodal values | JSON only | Image, Video, Audio via `.table` |
| Incremental embedding | Manual | Automatic via computed columns |
| History | No | Full version history per row |
| Computed columns | No | Arbitrary UDFs with lineage |
| External service | Requires PostgreSQL + pgvector | Embedded, no Docker |

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

# Search within a namespace
results = store.search(("users",), filter={"color": "blue"})

# List namespaces
namespaces = store.list_namespaces(prefix=("users",))
```

### Semantic Search

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

store.put(("docs",), "d1", {"text": "Pixeltable handles multimodal data"})
store.put(("docs",), "d2", {"text": "LangGraph builds stateful agents"})
store.put(("docs",), "d3", {"text": "Python is great for AI"})

results = store.search(("docs",), query="multimodal AI pipelines", limit=2)
for r in results:
    print(f"{r.key}: {r.value['text']} (score={r.score:.3f})")
```

### With LangGraph Agents

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.pixeltable import PixeltableStore

store = PixeltableStore(table_name="agent_memory.items")
store.setup()

agent = create_react_agent(model, tools=tools, store=store)
```

### `.table` Escape Hatch

Access the raw Pixeltable table for computed columns, UDFs, and multimodal lineage:

```python
import pixeltable as pxt

t = store.table  # the underlying Pixeltable table
t.add_computed_column(value_length=t.value['text'].apply(len), if_exists='ignore')
rows = t.select(t.key, t.value, t.value_length).collect()
```

## License

Apache 2.0
