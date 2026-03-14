# Contributing

## Workflow

1. Fork or create a branch from `main`
2. Make focused changes
3. Run local checks:

```bash
python -m compileall -q .
```

4. If tests exist/changed:

```bash
pytest -q
```

5. Open a pull request with:
- Problem statement
- What changed
- Screenshots (if UI changes)
- Any known limitations

## Commit Style

Use concise, imperative commit messages:
- `Add satellite tracking map panel`
- `Fix risk gauge threshold color`

## Scope Guidelines

- Keep runtime-generated artifacts out of commits (`data/processed`, `models`)
- Keep configuration in `config.py`
- Keep UI logic in `app.py`
- Keep ML logic in training/preprocess modules
