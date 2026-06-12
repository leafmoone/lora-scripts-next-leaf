# Tag-Edit-Leaf Static Page Maintenance

`tag-edit-leaf.html` is intentionally maintained as a standalone static page in
`frontend/dist/` until it is moved into a VuePress source page.

Maintenance rules:

- Keep behavior changes in `frontend/dist/tag-edit-leaf.html` small and reviewed.
- Keep backend request keys aligned with `mikazuki/app/tag_edit_leaf_api.py`.
- For new tagger CLI flags, update `mikazuki/utils/tagger_cmd.py` first, then wire
  the UI field to the shared config key.
- Do not add VuePress sidebar DOM injection for this page; expose it through direct
  URL or a non-sidebar portal link.
- Run at least `python -m py_compile mikazuki/app/tag_edit_leaf_api.py
  mikazuki/utils/tagger_cmd.py mikazuki/utils/tag_edit_leaf_helpers.py` after backend-facing edits.

Migration target:

- Move this page to a real frontend source directory when the project has a stable
  build path for standalone tool pages.
