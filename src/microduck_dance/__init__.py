"""Music driven dance for Microduck, built on mjlab_microduck (third_party/microduck-playground).

The task cfgs and their registration are in microduck_dance.tasks. That package
is listed under the mjlab.tasks entry point in pyproject.toml, so `import mjlab`
(and with it `uv run train`, `uv run play`, `uv run list-envs`) registers the
dance tasks next to the upstream ones.
"""
