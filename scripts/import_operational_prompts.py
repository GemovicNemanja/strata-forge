"""Prove unchanged composition of one hash-pinned, authorized operational Strata snapshot."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import cast

from strata_forge.prompts.composition import CompositionRegistry, compile_surface, render_build
from strata_forge.prompts.composition_errors import PromptCompositionError
from strata_forge.prompts.portable import PortableValue, canonical_bytes, parse_portable_json

FIXTURE_SHA256 = "91bb581829bfac2f42037526870f81f701935ecb8c439bcd0886025dc2334f09"
SOURCE_SHA256 = "232816d1cd753cac1cf6533119b2565b862ae498dae2e92f1a74d1ce144667ea"
SOURCE_COMMIT = "962d4da84ae4235c736dc2f9f7f72d0ee71530df"
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/prompts/operational-strata-source-v1.json"
)
_SUBSTITUTION = re.compile(r"\{\{(context|question)\}\}")


def _object(value: object) -> dict[str, PortableValue]:
    if type(value) is not dict:
        raise PromptCompositionError("operational_fixture_invalid")
    return cast("dict[str, PortableValue]", value)


def _array(value: object) -> list[PortableValue]:
    if type(value) is not list:
        raise PromptCompositionError("operational_fixture_invalid")
    return cast("list[PortableValue]", value)


def _string(value: object) -> str:
    if type(value) is not str:
        raise PromptCompositionError("operational_fixture_invalid")
    return value


def _parts(parts: object, *, fstring: bool) -> list[dict[str, PortableValue]]:
    result: list[dict[str, PortableValue]] = []
    for item in _array(parts):
        part = _object(item)
        if part.get("kind") == "module":
            result.append({"kind": "module", "revision": _string(part.get("revision"))})
        elif part.get("kind") == "text":
            literal = _string(part.get("text"))
            if not fstring:
                result.append({"kind": "text", "text": literal})
                continue
            if literal != "Selected module:\n{{context}}\n\nQuestion: {{question}}":
                raise PromptCompositionError("operational_dialect_drift")
            position = 0
            for match in _SUBSTITUTION.finditer(literal):
                result.append({"kind": "text", "text": literal[position : match.start()]})
                result.append({"kind": "variable", "name": match[1]})
                position = match.end()
            result.append({"kind": "text", "text": literal[position:]})
        else:
            raise PromptCompositionError("operational_fixture_invalid")
    return result


async def import_operational_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, PortableValue]:
    """Import only the audited archive snapshot and compare every surface to its baseline."""
    raw = await asyncio.to_thread(path.read_bytes)
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise PromptCompositionError("operational_fixture_drift")
    fixture = _object(parse_portable_json(raw))
    source = _object(fixture.get("provenance"))
    if source.get("commit") != SOURCE_COMMIT or source.get("fileSha256") != SOURCE_SHA256:
        raise PromptCompositionError("operational_source_drift")
    registry = CompositionRegistry()
    surfaces: list[str] = []
    for item in [*_array(fixture.get("modules")), *_array(fixture.get("surfaces"))]:
        old = _object(item)
        old_source = _object(old.get("source"))
        single = old.get("name") == "strata/single-shot"
        definition: dict[str, object] = {
            "kind": old.get("kind"),
            "name": old.get("name"),
            "revision": old.get("version"),
            "variables": [
                {"name": name, "type": kind} for name, kind in _object(old.get("variables")).items()
            ],
            "constraints": [
                {"key": key, "value": value}
                for key, value in _object(old.get("constraints", {})).items()
            ],
            "provenance": {
                "repository": old_source.get("repository"),
                "commit": old_source.get("commit"),
                "path": old_source.get("path"),
                "source_file_sha256": old_source.get("fileSha256"),
                "selector": old_source.get("selector"),
                "dialect": "strata-python-fstring-import-v1"
                if single
                else "strata-python-literal-import-v1",
            },
        }
        if old.get("kind") == "module":
            definition["parts"] = _parts(old.get("parts"), fstring=False)
        else:
            messages: list[dict[str, object]] = []
            for message in _array(old.get("messages")):
                original = _object(message)
                messages.append(
                    {
                        "role": original.get("role"),
                        "separator": original.get("separator"),
                        "parts": _parts(
                            original.get("parts"), fstring=single and original.get("role") == "user"
                        ),
                    }
                )
            definition["target"] = {"kind": "chat", "messages": messages}
        coordinate = await registry.put(definition)
        if old.get("kind") == "surface":
            surfaces.append(coordinate)
    expected = _object(fixture.get("expected"))
    expected_agents = _object(expected.get("agentSystems"))
    cases: list[PortableValue] = []
    context, question = "A😀\r\n{{question}}", "Keep {{context}} literal."
    values: dict[str, PortableValue] = {"context": context, "question": question}
    for coordinate in surfaces:
        build = await compile_surface(registry, coordinate)
        inputs = values if coordinate == "strata/single-shot@1" else {}
        rendered = await render_build(build, inputs)
        baseline: dict[str, object] = {
            "kind": "chat",
            "messages": (
                [
                    {"role": "system", "content": expected.get("singleShotSystem")},
                    {
                        "role": "user",
                        "content": f"Selected module:\n{context}\n\nQuestion: {question}",
                    },
                ]
                if inputs
                else [{"role": "system", "content": expected_agents.get(coordinate)}]
            ),
        }
        if canonical_bytes(rendered.wire_payload()["output"]) != canonical_bytes(baseline):
            raise PromptCompositionError("operational_semantics_changed")
        cases.append(
            {
                "name": coordinate,
                "values": inputs,
                "build_canonical_utf8_hex": build.canonical.hex(),
                "build_id": build.build_id,
                "render_canonical_utf8_hex": rendered.canonical.hex(),
                "render_id": rendered.render_id,
            }
        )
    return {
        "profile": "strata-prompt-operational-import-v1",
        "fixture_sha256": FIXTURE_SHA256,
        "source_file_sha256": SOURCE_SHA256,
        "source_commit": SOURCE_COMMIT,
        "modules": 6,
        "surfaces": len(surfaces),
        "cases": cases,
        "evidence": "archived operational source equality; no provider dispatch or research prompts",
    }


def main() -> None:
    """Write portable conformance evidence without printing prompt bodies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(import_operational_fixture(args.fixture))
    args.output.write_bytes(canonical_bytes(result) + b"\n")
    print(
        json.dumps(
            {
                "modules": result["modules"],
                "surfaces": result["surfaces"],
                "fixture_sha256": result["fixture_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
