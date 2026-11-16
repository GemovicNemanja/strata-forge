/** Cross-language fixture and adverse-input checks for the non-authoritative reference consumer. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  canonicalBytes, consumeBuild, consumeRender, hashDomain, parsePortableJSON, renderBuild,
} from "./prompt_reference.mts";

const bytes = JSON.parse(readFileSync(new URL("../tests/fixtures/prompts/portable-bytes-v1.json", import.meta.url), "utf8")) as {
  cases: { value: unknown; canonical_utf8_hex: string; sha256: string }[];
};
const corpus = JSON.parse(readFileSync(new URL("../tests/fixtures/prompts/composition-renders-v1.json", import.meta.url), "utf8")) as {
  cases: { name: string; values: unknown; expected_output: unknown; build_canonical_utf8_hex: string;
    build_id: string; render_canonical_utf8_hex: string; render_id: string }[];
};
for (const [index, fixture] of bytes.cases.entries()) test(`portable byte/hash fixture ${index}`, () => {
  assert.equal(canonicalBytes(fixture.value).toString("hex"), fixture.canonical_utf8_hex);
  assert.equal(hashDomain(parsePortableJSON(canonicalBytes(fixture.value))), fixture.sha256);
});
for (const fixture of corpus.cases) test(`Python build/render fixture ${fixture.name}`, () => {
  const build = consumeBuild(Buffer.from(fixture.build_canonical_utf8_hex, "hex").toString("utf8"));
  assert(Object.isFrozen(build));
  assert.equal(build.build_id, fixture.build_id);
  const rendered = renderBuild(build, fixture.values);
  assert(Object.isFrozen(rendered));
  assert.equal(rendered.render_id, fixture.render_id);
  assert.equal(Buffer.from(rendered.canonical).toString("hex"), fixture.render_canonical_utf8_hex);
  const output = (JSON.parse(rendered.canonical) as { output: unknown }).output;
  assert.deepEqual(output, fixture.expected_output);
  assert.equal(consumeRender(rendered.canonical).render_id, fixture.render_id);
  assert.equal(renderBuild(build, fixture.values).render_id, fixture.render_id);
});

for (const raw of ["-0", "-00", "+1", "01", "1.0", "1e0", "1E2", "NaN", "Infinity", "-Infinity", "9007199254740992", '"\\ud800"', '"\\udfff"', '{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', "\ufeff{}"])
  test(`invalid portable JSON ${JSON.stringify(raw)}`, () => assert.throws(() => parsePortableJSON(raw)));
test("invalid UTF8 refuses rather than replacing bytes", () => assert.throws(() => parsePortableJSON(new Uint8Array([0xff]))));
for (const value of [undefined, -0, 1.1, NaN, Infinity, new Date(0), new Map(), new Set(), 1n, "\ud800", "\udfff", [, 1], { "é": 1 }])
  test(`host value refuses ${typeof value}`, () => assert.throws(() => canonicalBytes(value)));
test("accessor and hidden properties never execute or disappear", () => {
  let read = false;
  const accessor = { get value() { read = true; return "not portable"; } };
  assert.throws(() => canonicalBytes(accessor));
  assert.equal(read, false);
  const hidden = Object.defineProperty({}, "secret", { value: 1 });
  assert.throws(() => canonicalBytes(hidden));
  const proxy = new Proxy({}, { getPrototypeOf() { read = true; return Object.prototype; } });
  assert.throws(() => canonicalBytes(proxy));
  assert.equal(read, false);
  assert.throws(() => hashDomain(accessor));
  assert.equal(read, false);
  assert.throws(() => canonicalBytes(new Array(2 ** 32 - 1)));
});
test("prototype spelling remains ordinary data", () => {
  const parsed = parsePortableJSON('{"__proto__":{"literal":true}}');
  assert.equal(Object.getPrototypeOf(parsed), null);
  assert.equal(canonicalBytes(parsed).toString(), '{"__proto__":{"literal":true}}');
});
test("U+FEFF inside a string or leaf remains data while a document BOM refuses", () => {
  assert.equal(canonicalBytes("\ufeff").toString("hex"), "22efbbbf22");
  assert.equal(parsePortableJSON('"\\ufeff"'), "\ufeff");
  assert.throws(() => parsePortableJSON(new Uint8Array([0xef, 0xbb, 0xbf, 123, 125])));
  assert.equal(new TextDecoder("utf8", { fatal: true, ignoreBOM: true }).decode(Buffer.from("\ufeff")), "\ufeff");
});
test("inherited domain and array elements never count as portable own data", () => {
  Object.defineProperty(Object.prototype, "domain", { value: "inherited", configurable: true });
  try { assert.throws(() => hashDomain({})); }
  finally { Reflect.deleteProperty(Object.prototype, "domain"); }
  const inherited = Object.setPrototypeOf([1], { 0: 2 });
  assert.throws(() => canonicalBytes(inherited));
  const hidden = Object.defineProperty([1], "extra", { value: 2 });
  assert.throws(() => canonicalBytes(hidden));
});
test("cycles and nesting are bounded", () => {
  const cycle: unknown[] = []; cycle.push(cycle);
  assert.throws(() => canonicalBytes(cycle));
  assert.throws(() => parsePortableJSON("[".repeat(65) + "null" + "]".repeat(65)));
});

type RecordValue = Record<string, unknown>;
function record(value: unknown): RecordValue { assert(value !== null && typeof value === "object" && !Array.isArray(value)); return value as RecordValue; }
function list(value: unknown): unknown[] { assert(Array.isArray(value)); return value; }
const first = corpus.cases[0];
for (const change of ["version", "extra", "pin", "lock_order", "variable", "origin", "leaf_shape", "target", "diagnostic"]) test(`malformed Python build refuses ${change}`, () => {
  const value = record(JSON.parse(Buffer.from(first.build_canonical_utf8_hex, "hex").toString()));
  const output = record(list(value.outputs)[0]), leaf = record(list(output.leaves)[0]);
  if (change === "version") value.schema_version = true;
  else if (change === "extra") value.arbitrary = "forbidden";
  else if (change === "pin") record(list(value.dependency_lock)[0]).coordinate = "surface@latest";
  else if (change === "lock_order") list(value.dependency_lock).push(list(value.dependency_lock)[0]);
  else if (change === "variable") record(list(value.variables)[0]).type = "object";
  else if (change === "origin") record(leaf.origin).coordinate = "unknown@1";
  else if (change === "leaf_shape") leaf.variable = "undeclared";
  else if (change === "target") value.kind = "tool";
  else value.diagnostics = [{ code: "duplicate_literal", severity: "advisory", origins: [] }];
  assert.throws(() => consumeBuild(canonicalBytes(value).toString()));
});
for (const change of ["output", "message", "role", "content", "backwards", "range", "unicode", "gap", "missing", "order", "index", "variable", "origin", "path", "separator"]) test(`malformed Python render refuses ${change}`, () => {
  const value = record(JSON.parse(Buffer.from(first.render_canonical_utf8_hex, "hex").toString()));
  const output = record(value.output), message = record(list(output.messages)[0]);
  const spans = list(value.source_map), span = record(spans[0]);
  if (change === "output") output.extra = "forbidden";
  else if (change === "message") message.extra = "forbidden";
  else if (change === "role") message.role = "tool";
  else if (change === "content") message.content = null;
  else if (change === "backwards") span.end = -1;
  else if (change === "range") span.end = 999;
  else if (change === "unicode") span.end = 2;
  else if (change === "gap") span.start = 1;
  else if (change === "missing") spans.length = 0;
  else if (change === "order") spans.reverse();
  else if (change === "index") span.output_index = 9;
  else if (change === "variable") span.variable = "x";
  else if (change === "origin") record(span.origin).coordinate = "$compiler";
  else if (change === "path") record(span.origin).occurrence_path = [1, 0];
  else record(spans[1]).kind = "text";
  assert.throws(() => consumeRender(canonicalBytes(value).toString()));
});
test("render inputs are exact and invocation IDs stay separate", () => {
  const build = consumeBuild(Buffer.from(first.build_canonical_utf8_hex, "hex").toString());
  for (const values of [{}, { x: true }, { x: 1 }, { x: "a", attempt_id: "execution-2" }]) assert.throws(() => renderBuild(build, values));
  const scalar = corpus.cases.find((item) => item.name === "exact-integer-boolean")!;
  const typed = consumeBuild(Buffer.from(scalar.build_canonical_utf8_hex, "hex").toString());
  assert.throws(() => renderBuild(typed, { n: true, b: true }));
});
