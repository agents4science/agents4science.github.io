"""Minimal stdlib validator for the ARIA contract schemas.

Checks the subset of JSON Schema the contracts use: required, type, enum,
const, pattern, additionalProperties, allOf, oneOf, if/then, $ref (relative
file refs and $id refs), items, minLength, minItems, minimum. The point is
to validate adapter output against the canonical contracts without adding a
dependency; the ARIA repo's own validator is the authority.

The contracts directory is located from (in order): an explicit argument,
the ARIA_CONTRACTS environment variable, or ../aria-spec/contracts relative
to this package's repository checkout.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def find_contracts_dir(explicit: str | Path | None = None) -> Path:
    for candidate in (
        explicit,
        os.environ.get('ARIA_CONTRACTS'),
        Path(__file__).resolve().parents[2] / 'aria-spec' / 'contracts',
    ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError(
        'ARIA contracts not found: set ARIA_CONTRACTS or check out '
        'the ARIA repo next to academy-aria',
    )


class SchemaStore:
    def __init__(self, contracts_dir: str | Path | None = None) -> None:
        self.root = find_contracts_dir(contracts_dir)
        self._by_path: dict[Path, dict] = {}
        self._by_id: dict[str, tuple[dict, Path]] = {}
        for sub in ('schemas/common', 'schemas/events', 'companion/schemas'):
            d = self.root / sub
            if not d.is_dir():
                continue
            for p in d.glob('*.json'):
                schema = json.loads(p.read_text())
                self._by_path[p.resolve()] = schema
                sid = schema.get('$id')
                if isinstance(sid, str):
                    self._by_id[sid] = (schema, p.resolve())

    def load(self, rel: str) -> tuple[dict, Path]:
        p = (self.root / rel).resolve()
        return json.loads(p.read_text()), p

    def resolve_ref(self, ref: str, current: Path) -> tuple[dict, Path]:
        if ref in self._by_id:
            return self._by_id[ref]
        target = (current.parent / ref).resolve()
        if target in self._by_path:
            return self._by_path[target], target
        return json.loads(target.read_text()), target


_TYPES = {
    'object': dict, 'array': list, 'string': str,
    'integer': int, 'number': (int, float), 'boolean': bool,
    'null': type(None),
}


def validate(instance: Any, schema: dict, store: SchemaStore,
             path: Path, trail: str = '$') -> list[str]:
    errs: list[str] = []
    if '$ref' in schema:
        schema, path = store.resolve_ref(schema['$ref'], path)

    for sub in schema.get('allOf', []):
        errs += validate(instance, sub, store, path, trail)

    if 'oneOf' in schema:
        hits = 0
        branch_errs: list[str] = []
        for sub in schema['oneOf']:
            be = validate(instance, sub, store, path, trail)
            if not be:
                hits += 1
            branch_errs += be
        if hits != 1:
            errs.append(f'{trail}: {hits} oneOf branches matched (want 1)')

    if 'if' in schema and _condition_matches(instance, schema['if'], store, path):
        if 'then' in schema:
            errs += validate(instance, schema['then'], store, path, trail)

    t = schema.get('type')
    if t:
        expected = _TYPES.get(t)
        if expected and not isinstance(instance, expected) or (
                t in ('integer', 'number') and isinstance(instance, bool)):
            errs.append(f'{trail}: expected {t}, got {type(instance).__name__}')
            return errs

    if 'enum' in schema and instance not in schema['enum']:
        errs.append(f'{trail}: {instance!r} not in enum')
    if 'const' in schema and instance != schema['const']:
        errs.append(f'{trail}: const mismatch')

    if isinstance(instance, dict):
        for req in schema.get('required', []):
            if req not in instance:
                errs.append(f'{trail}: missing required {req!r}')
        props = schema.get('properties', {})
        for key, val in instance.items():
            if key in props:
                errs += validate(val, props[key], store, path, f'{trail}.{key}')
            elif schema.get('additionalProperties') is False:
                errs.append(f'{trail}: additional property {key!r} not allowed')
    elif isinstance(instance, list):
        if 'minItems' in schema and len(instance) < schema['minItems']:
            errs.append(f'{trail}: fewer than {schema["minItems"]} items')
        items = schema.get('items')
        if items:
            for i, el in enumerate(instance):
                errs += validate(el, items, store, path, f'{trail}[{i}]')
    elif isinstance(instance, str):
        if 'minLength' in schema and len(instance) < schema['minLength']:
            errs.append(f'{trail}: shorter than minLength')
        pat = schema.get('pattern')
        if pat and not re.fullmatch(pat, instance):
            errs.append(f'{trail}: does not match pattern {pat}')
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if 'minimum' in schema and instance < schema['minimum']:
            errs.append(f'{trail}: below minimum')
    return errs


def _condition_matches(instance: Any, cond: dict, store: SchemaStore, path: Path) -> bool:
    if '$ref' in cond:
        cond, path = store.resolve_ref(cond['$ref'], path)
    if not isinstance(instance, dict):
        return False
    for req in cond.get('required', []):
        if req not in instance:
            return False
    for key, sub in cond.get('properties', {}).items():
        if key in instance and 'const' in sub and instance[key] != sub['const']:
            return False
    return True


def validate_event(event: dict, store: SchemaStore) -> list[str]:
    """Validate one event against the GmpEvent envelope (closed oneOf)."""
    schema, path = store.load('schemas/events/event-envelope.schema.json')
    return validate(event, schema, store, path)
