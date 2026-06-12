"""Character alias lookup for Anima Train (JSON + optional SQLite)."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .parser import dedupe_tags, split_tag_like_text

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = PACKAGE_ROOT / "resources"
CHARACTER_ALIASES_PATH = RESOURCES_DIR / "danbooru_character_aliases.json"
CHARACTER_ALIAS_SAFETY_PATH = RESOURCES_DIR / "character_alias_safety.json"
DEFAULT_DB_PATH = RESOURCES_DIR / "character_aliases.sqlite"

GENERIC_DESCRIPTOR_ALIAS_WORDS = {
    "girl", "boy", "woman", "man", "character", "original", "unknown",
    "solo", "duo", "group", "female", "male",
}


def normalize_alias_key(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace("_", " ")
        .replace("-", " ")
    )


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text))


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_character_alias_safety() -> dict[str, Any]:
    default = {
        "blocked_canonical_tags": ["blobcat", "grape-flavored_blobcat"],
        "blocked_aliases": [
            "blobcat", "grape", "grape-flavored blobcat", "grape-flavored_blobcat",
            "danbooru", "danbooru (site)", "kemono friends", "indie virtual youtuber",
        ],
        "blocked_generic_noun_aliases": [
            "hood", "jewelry", "coat", "shirt", "dress", "skirt", "jacket", "blouse",
            "glasses", "headphones", "hairpin", "ribbon", "bracelet", "necklace",
            "earrings", "boots", "shoes", "hat",
        ],
        "suspicious_copyright_tags": ["original", "danbooru_(site)", "indie_virtual_youtuber"],
        "single_word_ascii_min_count": 50,
    }
    if not CHARACTER_ALIAS_SAFETY_PATH.is_file():
        return default
    try:
        loaded = load_json_file(CHARACTER_ALIAS_SAFETY_PATH)
    except Exception:
        return default
    merged = dict(default)
    merged.update(loaded or {})
    return merged


def is_single_ascii_word(text: str) -> bool:
    normalized = normalize_alias_key(text)
    return bool(normalized) and normalized.isascii() and len(normalized.split()) == 1


def is_generic_descriptor_alias(text: str) -> bool:
    normalized = normalize_alias_key(text)
    if not normalized:
        return False
    words = [item for item in normalized.split() if item]
    if not words or len(words) > 3:
        return False
    return all(word in GENERIC_DESCRIPTOR_ALIAS_WORDS for word in words)


def should_skip_character_alias(alias_clean: str, canonical_clean: str, metadata: dict[str, Any], mode: str) -> bool:
    alias_key = normalize_alias_key(alias_clean)
    canonical_key = normalize_alias_key(canonical_clean)
    if not alias_key or not canonical_key:
        return True

    safety = load_character_alias_safety()
    blocked_aliases = {normalize_alias_key(item) for item in safety.get("blocked_aliases", [])}
    blocked_generic = {normalize_alias_key(item) for item in safety.get("blocked_generic_noun_aliases", [])}
    blocked_canonicals = {normalize_alias_key(item) for item in safety.get("blocked_canonical_tags", [])}
    suspicious_copyrights = {normalize_alias_key(item) for item in safety.get("suspicious_copyright_tags", [])}

    if alias_key in blocked_aliases or canonical_key in blocked_canonicals:
        return True
    if alias_key in blocked_generic:
        return True

    copyright_tag_key = normalize_alias_key(metadata.get("copyright_tag", ""))
    copyright_name_key = normalize_alias_key(metadata.get("copyright_name_zh", ""))
    if alias_key and (alias_key == copyright_tag_key or alias_key == copyright_name_key):
        return True

    count = int(metadata.get("count") or 0)
    source_name = str(metadata.get("_source", "")).strip().lower()
    if source_name != "manual" and is_generic_descriptor_alias(alias_clean):
        return True
    if is_single_ascii_word(alias_clean):
        if mode == "tag_text" and source_name != "manual":
            return True
        if mode == "free_text" and count < int(safety.get("single_word_ascii_min_count", 50)):
            return True
        if copyright_tag_key in suspicious_copyrights and count < 1000:
            return True
    return False


class AliasIndex:
    """In-memory alias map with optional SQLite backing store."""

    def __init__(self, db_path: Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.db_path = db_path or DEFAULT_DB_PATH
        self._aliases: dict[str, dict[str, Any]] = {}
        if enabled:
            self._load()

    def _load(self) -> None:
        if self.db_path.is_file():
            self._load_from_sqlite()
        elif CHARACTER_ALIASES_PATH.is_file():
            self._load_from_json(CHARACTER_ALIASES_PATH, source_name="manual")

    def _load_from_json(self, path: Path, *, source_name: str) -> None:
        raw = load_json_file(path)
        for canonical_tag, alias_spec in (raw or {}).items():
            canonical_clean = str(canonical_tag).strip()
            if not canonical_clean:
                continue
            if isinstance(alias_spec, dict):
                alias_list = alias_spec.get("aliases", [])
                blocked_tags = alias_spec.get("blocked_tags", [])
                metadata = dict(alias_spec.get("metadata", {}) or {})
            else:
                alias_list = alias_spec
                blocked_tags = []
                metadata = {}
            metadata["_source"] = source_name
            if should_skip_character_alias(canonical_clean, canonical_clean, metadata, mode="load"):
                continue
            priority = 10**12 if source_name == "manual" else int(metadata.get("count") or 0)
            self._upsert_alias(canonical_clean, canonical_clean, blocked_tags, metadata, priority)
            for alias in alias_list:
                alias_clean = str(alias).strip()
                if not alias_clean:
                    continue
                if should_skip_character_alias(alias_clean, canonical_clean, metadata, mode="load"):
                    continue
                self._upsert_alias(alias_clean, canonical_clean, blocked_tags, metadata, priority)

    def _upsert_alias(
        self,
        matched_alias: str,
        canonical_tag: str,
        blocked_tags: list[str],
        metadata: dict[str, Any],
        priority: int,
    ) -> None:
        alias_key = normalize_alias_key(matched_alias)
        entry = {
            "canonical_tag": canonical_tag,
            "matched_alias": matched_alias,
            "blocked_tags": [str(tag).strip() for tag in blocked_tags if str(tag).strip()],
            "_priority": priority,
            "_metadata": dict(metadata or {}),
        }
        current = self._aliases.get(alias_key)
        if current is None or priority >= current.get("_priority", -1):
            self._aliases[alias_key] = entry

    def _load_from_sqlite(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT alias_key, matched_alias, canonical_tag, blocked_tags, metadata_json, priority "
                "FROM character_aliases"
            ).fetchall()
        finally:
            conn.close()
        for alias_key, matched_alias, canonical_tag, blocked_tags_json, metadata_json, priority in rows:
            try:
                blocked_tags = json.loads(blocked_tags_json or "[]")
            except Exception:
                blocked_tags = []
            try:
                metadata = json.loads(metadata_json or "{}")
            except Exception:
                metadata = {}
            self._aliases[str(alias_key)] = {
                "canonical_tag": str(canonical_tag),
                "matched_alias": str(matched_alias),
                "blocked_tags": blocked_tags,
                "_priority": int(priority or 0),
                "_metadata": metadata,
            }

    def detect_in_tag_list(self, raw_tags: str | list[str]) -> list[dict[str, Any]]:
        if not self.enabled or not self._aliases:
            return []
        tokens = split_tag_like_text(raw_tags) if isinstance(raw_tags, str) else list(raw_tags or [])
        hits: list[dict[str, Any]] = []
        seen_canonical: set[str] = set()
        for token in tokens:
            alias = self._aliases.get(normalize_alias_key(token))
            if not alias:
                continue
            if should_skip_character_alias(
                alias["matched_alias"],
                alias["canonical_tag"],
                alias.get("_metadata", {}),
                mode="tag_text",
            ):
                continue
            canonical = alias["canonical_tag"]
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            hits.append(
                {
                    "matched_alias": token,
                    "canonical_tag": canonical,
                    "blocked_tags": list(alias.get("blocked_tags", [])),
                }
            )
        return hits

    def preprocess_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(inputs or {})
        alias_hits = self.detect_in_tag_list(prepared.get("raw_tags", ""))
        prepared["resolved_character_tags"] = alias_hits
        prepared["resolved_character_tag_strings"] = [hit["canonical_tag"] for hit in alias_hits]
        return prepared


def build_sqlite_index(
    json_path: Path,
    db_path: Path,
    *,
    source_name: str = "manual",
) -> int:
    index = AliasIndex(enabled=False)
    index._load_from_json(json_path, source_name=source_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS character_aliases (
                alias_key TEXT PRIMARY KEY,
                matched_alias TEXT NOT NULL,
                canonical_tag TEXT NOT NULL,
                blocked_tags TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                priority INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("DELETE FROM character_aliases")
        rows = []
        for alias_key, entry in index._aliases.items():
            rows.append(
                (
                    alias_key,
                    entry["matched_alias"],
                    entry["canonical_tag"],
                    json.dumps(entry.get("blocked_tags", []), ensure_ascii=False),
                    json.dumps(entry.get("_metadata", {}), ensure_ascii=False),
                    int(entry.get("_priority", 0)),
                )
            )
        conn.executemany(
            "INSERT INTO character_aliases "
            "(alias_key, matched_alias, canonical_tag, blocked_tags, metadata_json, priority) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()
