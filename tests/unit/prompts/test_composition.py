"""Exact expansion, immutable identities, scalar rendering and structural refusal fixtures."""

from __future__ import annotations

import copy
from typing import cast

import pytest

from strata_forge.prompts import composition
from strata_forge.prompts.composition import (
    CompositionRegistry,
    PromptBuild,
    PromptRender,
    compile_surface,
    composition_capabilities,
    render_build,
)
from strata_forge.prompts.composition_errors import PromptCompositionError
from strata_forge.prompts.composition_types import PortableDocument, definition_from_wire
from strata_forge.prompts.portable import canonical_bytes


def obj(value: object) -> dict[str, object]:
    assert type(value) is dict
    return cast("dict[str, object]", value)


def seq(value: object) -> list[object]:
    assert type(value) is list
    return cast("list[object]", value)


def text(value: str) -> dict[str, object]:
    return {"kind": "text", "text": value}


def variable(name: str) -> dict[str, object]:
    return {"kind": "variable", "name": name}


def include(coordinate: str) -> dict[str, object]:
    return {"kind": "module", "revision": coordinate}


def module(name: str, parts: list[object], **changes: object) -> dict[str, object]:
    return {"kind": "module", "name": name, "revision": 1, "parts": parts, **changes}


def surface(parts: list[object], *, separator: str = "", **changes: object) -> dict[str, object]:
    return {
        "kind": "surface",
        "name": "surface",
        "revision": 1,
        "target": {
            "kind": "chat",
            "messages": [{"role": "system", "parts": parts, "separator": separator}],
        },
        **changes,
    }


async def registry(*definitions: object) -> CompositionRegistry:
    result = CompositionRegistry()
    for definition in definitions:
        await result.put(definition)
    return result


async def output_text(build: PromptBuild, values: object = None) -> str:
    rendered = await render_build(build, {} if values is None else values)
    messages = seq(obj(rendered.wire_payload()["output"])["messages"])
    value = obj(messages[0])["content"]
    assert isinstance(value, str)
    return value


async def test_nested_order_empty_repeated_leaves_and_exact_occurrences() -> None:
    store = await registry(
        module("A", [text(""), text("A😀")]),
        module("B", [text("é"), include("A@1"), text("\r\n")]),
        surface([include("B@1"), include("A@1"), text("!")], separator="|"),
    )
    build = await compile_surface(store, "surface@1")
    assert await output_text(build) == "é||A😀|\r\n||A😀|!"
    wire = build.wire_payload()
    locks = seq(wire["dependency_lock"])
    assert [obj(lock)["coordinate"] for lock in locks] == ["A@1", "B@1", "surface@1"]
    leaves = seq(obj(seq(wire["outputs"])[0])["leaves"])
    assert [obj(obj(leaf)["origin"])["occurrence_path"] for leaf in leaves] == [
        [0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2],
    ]
    assert len(seq(wire["diagnostics"])) == 1
    assert obj(seq(wire["diagnostics"])[0])["severity"] == "advisory"


async def test_exact_scalar_rendering_never_reparses_inserted_values() -> None:
    declarations = [
        {"name": "s", "type": "string"},
        {"name": "n", "type": "integer"},
        {"name": "b", "type": "boolean"},
    ]
    store = await registry(
        surface(
            [text("{{ literal }}"), variable("s"), variable("n"), variable("b")],
            separator="\n",
            variables=declarations,
        )
    )
    build = await compile_surface(store, "surface@1")
    assert await output_text(
        build, {"s": "{{ n }} {b} {% include 'file' %}", "n": -9007199254740991, "b": False}
    ) == ("{{ literal }}\n{{ n }} {b} {% include 'file' %}\n-9007199254740991\nfalse")


async def test_utf8_spans_cover_exact_emitted_leaves_and_compiler_separators() -> None:
    store = await registry(
        surface([text("A"), text("😀"), text("é"), text("\r\n"), text("")], separator="")
    )
    rendered = await render_build(await compile_surface(store, "surface@1"), {})
    wire = rendered.wire_payload()
    source = "A😀é\r\n".encode()
    spans = seq(wire["source_map"])
    leaves = [obj(span) for span in spans if obj(span)["kind"] == "text"]
    assert [(span["start"], span["end"]) for span in leaves] == [
        (0, 1),
        (1, 5),
        (5, 8),
        (8, 10),
        (10, 10),
    ]
    assert [
        source[cast("int", span["start"]) : cast("int", span["end"])].decode() for span in leaves
    ] == ["A", "😀", "é", "\r\n", ""]
    separators = [obj(span) for span in spans if obj(span)["kind"] == "separator"]
    assert len(separators) == 4
    assert all(obj(span["origin"])["coordinate"] == "$compiler" for span in separators)
    assert all(span["start"] == span["end"] for span in separators)


@pytest.mark.parametrize("purpose", ["system_instruction", "tool_description", "skill", "text"])
async def test_artifact_purpose_and_descriptor_never_turn_into_chat(purpose: str) -> None:
    descriptor = {
        "name": "fixture_tool",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    definition = surface(
        [],
        target={
            "kind": "artifact",
            "purpose": purpose,
            "parts": [text("Exact description")],
            "separator": "",
            "descriptor": descriptor,
        },
    )
    store = await registry(definition)
    build = await compile_surface(store, "surface@1")
    descriptor["name"] = "caller mutation"
    rendered = await render_build(build, {})
    output = obj(rendered.wire_payload()["output"])
    assert output == {
        "kind": "artifact",
        "purpose": purpose,
        "content": "Exact description",
        "descriptor": {**descriptor, "name": "fixture_tool"},
    }
    assert "messages" not in output


async def test_snapshot_build_and_render_views_are_independent_and_idempotent() -> None:
    definition = module("core", [text("original")])
    store = await registry(definition, surface([include("core@1")]))
    assert await store.put(copy.deepcopy(definition)) == "core@1"
    first = await compile_surface(store, "surface@1")
    definition["parts"] = [text("mutated")]
    view = first.wire_payload()
    view["outputs"] = []
    resolved = await store.get("core@1")
    assert resolved.wire_payload()["parts"] == [{"kind": "text", "text": "original"}]
    second = await compile_surface(store, "surface@1")
    assert first.canonical == second.canonical
    assert first.build_id == second.build_id
    one, two = await render_build(first, {}), await render_build(second, {})
    one.wire_payload()["output"] = "discarded caller view"
    assert one.canonical == two.canonical
    assert one.render_id == two.render_id
    assert await output_text(first) == "original"


async def test_registry_order_never_changes_identity_but_versions_and_literals_do() -> None:
    definitions = [
        module("core", [text("A")]),
        module("unused", [text("ignored")]),
        surface([include("core@1")]),
    ]
    a, b = await registry(*definitions), await registry(*reversed(definitions))
    first, second = await compile_surface(a, "surface@1"), await compile_surface(b, "surface@1")
    assert first.build_id == second.build_id
    await b.put(module("core", [text("A")], revision=2))
    await b.put(surface([include("core@2")], revision=2))
    assert (await compile_surface(b, "surface@2")).build_id != first.build_id
    assert await output_text(await compile_surface(b, "surface@2")) == "A"
    with pytest.raises(PromptCompositionError, match="revision_replaced"):
        await b.put(module("core", [text("changed")]))


@pytest.mark.parametrize(
    "pin",
    [
        "core@latest",
        "core@0",
        "core@01",
        "core@-1",
        "https://x/core@1",
        "core@9007199254740992",
        "core@1e0",
    ],
)
async def test_non_exact_module_pins_refuse(pin: str) -> None:
    store = CompositionRegistry()
    with pytest.raises(PromptCompositionError):
        await store.put(surface([include(pin)]))


@pytest.mark.parametrize(
    "name", ["../core", "core//part", "core/", "/core", "é", "core..part", "1core"]
)
async def test_unsafe_names_refuse_without_normalization(name: str) -> None:
    with pytest.raises(PromptCompositionError):
        await CompositionRegistry().put(module(name, []))


async def test_missing_cycle_and_surface_as_module_are_closed() -> None:
    store = await registry(surface([include("missing@1")]))
    with pytest.raises(PromptCompositionError, match="dependency_missing"):
        await compile_surface(store, "surface@1")
    store = await registry(
        module("A", [include("B@1")]), module("B", [include("A@1")]), surface([include("A@1")])
    )
    with pytest.raises(PromptCompositionError, match="dependency_cycle") as caught:
        await compile_surface(store, "surface@1")
    assert len(caught.value.diagnostic.origins) <= 34
    store = await registry(surface([include("surface@1")]))
    with pytest.raises(PromptCompositionError, match="module_required"):
        await compile_surface(store, "surface@1")


async def test_equal_constraints_repeat_but_distinct_values_refuse_with_both_origins() -> None:
    fact = {"key": "audience", "value": "developer"}
    store = await registry(
        module("core", [text("same")], constraints=[fact, fact]),
        surface([include("core@1")], constraints=[fact]),
    )
    assert await output_text(await compile_surface(store, "surface@1")) == "same"
    await store.put(module("other", [], constraints=[{"key": "audience", "value": "operator"}]))
    await store.put(surface([include("core@1"), include("other@1")], revision=2))
    with pytest.raises(PromptCompositionError, match="constraint_conflict") as caught:
        await compile_surface(store, "surface@2")
    assert {origin.coordinate for origin in caught.value.diagnostic.origins} == {
        "core@1",
        "other@1",
    }
    assert "developer" not in str(caught.value)
    assert "operator" not in str(caught.value)


@pytest.mark.parametrize("declarations", [[], [{"name": "x", "type": "integer"}]])
async def test_surface_must_declare_compatible_dependency_inputs(declarations: object) -> None:
    store = await registry(
        module("core", [variable("x")], variables=[{"name": "x", "type": "string"}]),
        surface([include("core@1")], variables=declarations),
    )
    with pytest.raises(PromptCompositionError, match="variable_type_conflict"):
        await compile_surface(store, "surface@1")


async def test_owning_definition_must_declare_each_variable_and_unused_inputs_remain_required() -> (
    None
):
    store = await registry(
        module("core", [variable("x")]),
        surface([include("core@1")], variables=[{"name": "x", "type": "string"}]),
    )
    with pytest.raises(PromptCompositionError, match="variable_undeclared"):
        await compile_surface(store, "surface@1")
    store = await registry(
        surface([text("constant")], variables=[{"name": "unused", "type": "string"}])
    )
    build = await compile_surface(store, "surface@1")
    with pytest.raises(PromptCompositionError, match="input_keys_mismatch"):
        await render_build(build, {})
    assert await output_text(build, {"unused": "value"}) == "constant"


@pytest.mark.parametrize(
    "values", [{}, {"x": "a", "extra": "secret"}, {"x": 1}, {"x": True}, {"x": None}]
)
async def test_exact_input_types_and_keys_refuse_without_values(values: object) -> None:
    store = await registry(surface([variable("x")], variables=[{"name": "x", "type": "string"}]))
    with pytest.raises(PromptCompositionError) as caught:
        await render_build(await compile_surface(store, "surface@1"), values)
    assert "secret" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize("value", [True, 1.0, 9007199254740992])
async def test_integer_inputs_never_accept_booleans_floats_or_unsafe_values(value: object) -> None:
    store = await registry(surface([variable("x")], variables=[{"name": "x", "type": "integer"}]))
    with pytest.raises(PromptCompositionError):
        await render_build(await compile_surface(store, "surface@1"), {"x": value})


async def test_definition_build_and_render_limits_refuse_without_partial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(composition, "MAX_DEFINITIONS", 1)
    store = await registry(module("first", []))
    with pytest.raises(PromptCompositionError, match="registry_definition_limit"):
        await store.put(module("second", []))
    monkeypatch.setattr(composition, "MAX_DEFINITIONS", 4096)
    monkeypatch.setattr(composition, "MAX_DEFINITION_BYTES", 100)
    with pytest.raises(PromptCompositionError, match="registry_bytes_limit"):
        await CompositionRegistry().put(module("first", [text("x" * 100)]))
    monkeypatch.setattr(composition, "MAX_DEFINITION_BYTES", 8 * 1024 * 1024)
    store = await registry(
        module("core", [text("A"), text("A")]), surface([include("core@1"), include("core@1")])
    )
    monkeypatch.setattr(composition, "MAX_EXPANDED_LEAVES", 3)
    with pytest.raises(PromptCompositionError, match="expanded_leaves_limit"):
        await compile_surface(store, "surface@1")
    monkeypatch.setattr(composition, "MAX_EXPANDED_LEAVES", 16384)
    build = await compile_surface(store, "surface@1")
    monkeypatch.setattr(composition, "MAX_RENDERED_BYTES", 3)
    with pytest.raises(PromptCompositionError, match="rendered_bytes_limit"):
        await render_build(build, {})


async def test_depth_diagnostic_and_expansion_visit_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    store = await registry(
        module("A", [text("A")]), module("B", [include("A@1")]), surface([include("B@1")])
    )
    monkeypatch.setattr(composition, "MAX_DEPENDENCY_DEPTH", 1)
    with pytest.raises(PromptCompositionError, match="dependency_depth_limit"):
        await compile_surface(store, "surface@1")
    monkeypatch.setattr(composition, "MAX_DEPENDENCY_DEPTH", 32)
    monkeypatch.setattr(composition, "MAX_EXPANSION_VISITS", 1)
    with pytest.raises(PromptCompositionError, match="expansion_visits_limit"):
        await compile_surface(store, "surface@1")
    monkeypatch.setattr(composition, "MAX_EXPANSION_VISITS", 1048576)
    store = await registry(surface([text("A"), text("A"), text("B"), text("B")]))
    monkeypatch.setattr(composition, "MAX_DIAGNOSTICS", 1)
    with pytest.raises(PromptCompositionError, match="diagnostics_limit"):
        await compile_surface(store, "surface@1")


async def test_empty_module_dag_is_not_exponentially_expanded() -> None:
    store = await registry(module("empty", []))
    prior = "empty@1"
    for index in range(10):
        name = f"level{index}"
        await store.put(module(name, [include(prior)] * 100))
        prior = name + "@1"
    await store.put(surface([include(prior)]))
    build = await compile_surface(store, "surface@1")
    assert await output_text(build) == ""
    assert obj(seq(build.wire_payload()["outputs"])[0])["leaves"] == []


def test_portable_descriptor_and_parsed_definition_are_deeply_immutable() -> None:
    original = {"nested": {"values": ["original"]}}
    frozen = PortableDocument.from_value(original)
    original["nested"]["values"].append("caller mutation")
    view = frozen.wire_payload()
    view["nested"] = "changed"
    assert frozen.wire_payload() == {"nested": {"values": ["original"]}}
    parsed = definition_from_wire(module("core", [text("literal")]))
    with pytest.raises(ValueError, match="frozen"):
        parsed.name = "changed"


async def test_untrusted_compiled_payload_cannot_bypass_exact_shape_or_leaf_bounds() -> None:
    store = await registry(surface([text("A")]))
    build = await compile_surface(store, "surface@1")
    wire = build.wire_payload()
    wire["schema_version"] = True
    with pytest.raises(PromptCompositionError):
        PromptBuild(canonical=canonical_bytes(wire))
    wire = build.wire_payload()
    obj(seq(obj(seq(wire["outputs"])[0])["leaves"])[0])["variable"] = "undeclared"
    with pytest.raises(PromptCompositionError, match="build_invalid"):
        PromptBuild(canonical=canonical_bytes(wire))


@pytest.mark.parametrize(
    "change",
    [
        "unknown_output",
        "unknown_message",
        "role",
        "content",
        "backwards",
        "range",
        "unicode",
        "gap",
        "missing",
        "out_of_order",
        "output_index",
        "text_variable",
        "variable_missing",
        "separator_origin",
        "separator_path",
        "leaf_origin",
        "leaf_path",
        "trailing_separator",
    ],
)
async def test_imported_render_refuses_malformed_output_or_attribution(change: str) -> None:
    store = await registry(
        surface(
            [text("A"), text("😀"), variable("x"), text("")],
            separator="\r\n",
            variables=[{"name": "x", "type": "string"}],
        )
    )
    rendered = await render_build(await compile_surface(store, "surface@1"), {"x": "é"})
    wire = rendered.wire_payload()
    output = obj(wire["output"])
    messages = seq(output["messages"])
    spans = seq(wire["source_map"])
    first, separator, astral, variable_span = (
        obj(spans[0]),
        obj(spans[1]),
        obj(spans[2]),
        obj(spans[4]),
    )
    if change == "unknown_output":
        output["unknown"] = "forbidden"
    elif change == "unknown_message":
        obj(messages[0])["unknown"] = "forbidden"
    elif change == "role":
        obj(messages[0])["role"] = "tool"
    elif change == "content":
        obj(messages[0])["content"] = None
    elif change == "backwards":
        first["end"] = -1
    elif change == "range":
        obj(spans[-1])["end"] = 999
    elif change == "unicode":
        astral["end"] = 4
    elif change == "gap":
        first["start"] = 1
    elif change == "missing":
        spans.pop()
    elif change == "out_of_order":
        spans.reverse()
    elif change == "output_index":
        first["output_index"] = 1
    elif change == "text_variable":
        first["variable"] = "x"
    elif change == "variable_missing":
        variable_span["variable"] = None
    elif change == "separator_origin":
        obj(separator["origin"])["coordinate"] = "surface@1"
    elif change == "separator_path":
        obj(separator["origin"])["occurrence_path"] = [0, 99]
    elif change == "leaf_origin":
        obj(first["origin"])["coordinate"] = "$compiler"
    elif change == "leaf_path":
        obj(first["origin"])["occurrence_path"] = [1, 0]
    else:
        spans.pop()
    with pytest.raises(PromptCompositionError) as caught:
        PromptRender(canonical=canonical_bytes(wire))
    assert caught.value.__context__ is None


async def test_capabilities_are_versioned_immutable_values() -> None:
    capabilities = await composition_capabilities()
    assert capabilities.compiler_version == "strata-modular-compiler/1"
    assert capabilities.registry_definitions == 4096
    assert capabilities.rendered_utf8_bytes == 8 * 1024 * 1024
    assert capabilities.portable_json_bytes == 64 * 1024 * 1024
    view = capabilities.model_dump()
    view["registry_definitions"] = 1
    assert (await composition_capabilities()).registry_definitions == 4096


async def test_provenance_target_and_input_fingerprint_participate_in_identity() -> None:
    source = {
        "repository": "fixture/repository",
        "commit": "a" * 40,
        "path": "source.txt",
        "source_file_sha256": "b" * 64,
        "selector": "literal",
        "dialect": "fixture-v1",
    }
    definitions = [
        surface(
            [text("constant")], provenance=source, variables=[{"name": "unused", "type": "string"}]
        )
    ]
    first = await compile_surface(await registry(*definitions), "surface@1")
    changed = copy.deepcopy(definitions)
    obj(changed[0]["provenance"])["commit"] = "c" * 40
    second = await compile_surface(await registry(*changed), "surface@1")
    assert first.build_id != second.build_id
    one = await render_build(first, {"unused": "first value"})
    two = await render_build(first, {"unused": "second value"})
    assert one.wire_payload()["output"] == two.wire_payload()["output"]
    assert one.wire_payload()["input_fingerprint"] != two.wire_payload()["input_fingerprint"]
    assert one.render_id != two.render_id
    changed = copy.deepcopy(definitions)
    obj(seq(obj(changed[0]["target"])["messages"])[0])["role"] = "user"
    assert (await compile_surface(await registry(*changed), "surface@1")).build_id != first.build_id


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "part",
        "field",
        "role",
        "revision",
        "variables",
        "parts",
        "messages",
        "total_parts",
    ],
)
async def test_authoring_schema_and_per_definition_bounds_are_closed(change: str) -> None:
    value = surface([text("literal")])
    target = obj(value["target"])
    if change == "target":
        target["kind"] = "provider_callback"
    elif change == "part":
        obj(seq(target["messages"])[0])["parts"] = [{"kind": "expression", "source": "forbidden"}]
    elif change == "field":
        value["executable_import"] = "forbidden"
    elif change == "role":
        obj(seq(target["messages"])[0])["role"] = "tool"
    elif change == "revision":
        value["revision"] = True
    elif change == "variables":
        value["variables"] = [{"name": f"x{index}", "type": "string"} for index in range(257)]
    elif change == "parts":
        obj(seq(target["messages"])[0])["parts"] = [text("")] * 1025
    elif change == "messages":
        target["messages"] = [{"role": "system", "separator": "", "parts": []}] * 129
    else:
        target["messages"] = [{"role": "system", "separator": "", "parts": [text("")] * 600}] * 2
    with pytest.raises(PromptCompositionError) as caught:
        await CompositionRegistry().put(value)
    assert "forbidden" not in str(caught.value)
    assert caught.value.__context__ is None


async def test_exact_supported_depth_and_one_level_over_limit() -> None:
    store = await registry(module("level0", [text("deep")]))
    for index in range(1, 33):
        await store.put(module(f"level{index}", [include(f"level{index - 1}@1")]))
    await store.put(surface([include("level31@1")]))
    assert await output_text(await compile_surface(store, "surface@1")) == "deep"
    await store.put(surface([include("level32@1")], revision=2))
    with pytest.raises(PromptCompositionError, match="dependency_depth_limit"):
        await compile_surface(store, "surface@2")


async def test_exact_eight_mib_utf8_output_and_one_byte_over_limit() -> None:
    store = await registry(surface([variable("x")], variables=[{"name": "x", "type": "string"}]))
    build = await compile_surface(store, "surface@1")
    value = "😀" * (8 * 1024 * 1024 // 4)
    rendered = await render_build(build, {"x": value})
    assert obj(seq(rendered.wire_payload()["source_map"])[0])["end"] == 8 * 1024 * 1024
    with pytest.raises(PromptCompositionError, match="rendered_bytes_limit"):
        await render_build(build, {"x": value + "A"})
