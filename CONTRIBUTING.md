# Contributing

## Development

```bash
git clone https://github.com/pixeltable/langgraph-store-pixeltable.git
cd langgraph-store-pixeltable
pip install -e ".[dev]"
pytest tests/ -v
ruff check . && ruff format --check .
```

## Releasing

Releases are published to PyPI automatically when a GitHub Release is created.

### One-time setup (trusted publishing)

1. Add a trusted publisher on [PyPI](https://pypi.org/manage/project/langgraph-store-pixeltable/settings/publishing/) (owner/repo/workflow/environment)
2. Create a GitHub environment called `pypi`

### Publishing a release

1. Bump version in `langgraph/store/pixeltable/__init__.py` and `pyproject.toml`
2. Commit and push to `main`
3. Create a GitHub Release with a `v*` tag (e.g. `v0.1.1`)
