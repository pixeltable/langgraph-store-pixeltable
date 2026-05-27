"""LangGraph BaseStore backed by Pixeltable.

Install:
    pip install langgraph-store-pixeltable

Usage:
    from langgraph.store.pixeltable import PixeltableStore

    store = PixeltableStore(table_name="agent_memory.items")
    store.setup()
    store.put(("users",), "u1", {"name": "Alice"})
    item = store.get(("users",), "u1")
"""

from langgraph.store.pixeltable.store import PixeltableStore

__all__ = ["PixeltableStore"]
__version__ = "0.1.0"
