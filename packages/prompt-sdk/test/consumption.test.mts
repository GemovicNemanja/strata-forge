/** Synthetic pinned bytes and the unchanged Python-render corpus, without a provider. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { canonicalBytes, consumePinnedBuild, renderConsumed, type LocalBuildSource } from "../src/index.mts";

const corpus = JSON.parse(readFileSync(new URL("../../../tests/fixtures/prompts/composition-renders-v1.json", import.meta.url), "utf8")) as {
  cases: { name: string; values: unknown; build_canonical_utf8_hex: string; build_id: string; render_id: string; render_canonical_utf8_hex: string }[];
};
const serverBytes = readFileSync(new URL("./fixtures/stored-build-http-v1.json", import.meta.url));
const server = JSON.parse(serverBytes.toString("utf8")) as { body_utf8: string; body_sha256: string; body_size_bytes: number };
const source: LocalBuildSource = { kind: "caller_bytes", partition: "fixture:local", reference: null };
const sha = (raw: Uint8Array) => createHash("sha256").update(raw).digest("hex");
function consume(raw: Uint8Array, changes: Record<string, unknown> = {}) {
  return consumePinnedBuild(raw, { expectedBuildId: sha(raw), expectedSizeBytes: raw.length, source, ...changes });
}
for (const fixture of corpus.cases) test(`same immutable Python render ${fixture.name}`, () => {
  const raw = Buffer.from(fixture.build_canonical_utf8_hex, "hex"), build = consume(raw);
  assert.equal(build.build_id, fixture.build_id);
  const rendered = renderConsumed(build, fixture.values);
  assert.equal(rendered.render_id, fixture.render_id);
  assert.equal(Buffer.from(rendered.canonical).toString("hex"), fixture.render_canonical_utf8_hex);
  assert(Object.isFrozen(build) && Object.isFrozen(build.source));
});
test("actual server golden body preserves original bytes and only descriptive source", () => {
  assert.equal(sha(serverBytes), "5efc2497eac64c41fcdca329f085875d396be6d9ffe9f4bfff48e0e65aa83cb7");
  const raw = Buffer.from(server.body_utf8), build = consume(raw);
  assert.equal(build.build_id, server.body_sha256); assert.equal(build.size_bytes, server.body_size_bytes);
  raw.fill(0); assert.equal(build.canonical, server.body_utf8);
  const rendered = renderConsumed(build, { question: "{{literal}} 😀é\r\n" });
  assert(JSON.stringify(JSON.parse(rendered.canonical).output).includes("{{literal}}"));
  const other = consume(Buffer.from(server.body_utf8), { source: { ...source, partition: "fixture:other" } });
  assert.equal(renderConsumed(other, { question: "{{literal}} 😀é\r\n" }).canonical, rendered.canonical);
});
for (const changes of [
  { expectedBuildId: "0".repeat(64) }, { expectedBuildId: "f".repeat(64) + "\n" },
  { expectedSizeBytes: 0 }, { expectedSizeBytes: true }, { expectedSizeBytes: 1 }, { expectedSizeBytes: 8 * 1024 * 1024 + 1 },
  { source: { ...source, kind: "server_authorized" } }, { source: { ...source, partition: "private-canary\n" } },
  { source: { ...source, reference: { resource_id: "private-canary" } } },
]) test(`bad local pin/source ${JSON.stringify(changes)}`, () => {
  assert.throws(() => consume(Buffer.from(server.body_utf8), changes), (error: unknown) => error instanceof Error && !error.message.includes("private-canary"));
});
for (const raw of [Buffer.alloc(0), Buffer.from("{}"), Buffer.from([0xff]), Buffer.from('{"schema_version":1,"schema_version":1}'), Buffer.from("\ufeff{}")]) test(`invalid bytes still refuse ${raw.length}`, () => assert.throws(() => consume(raw)));
for (const values of [{}, { question: true }, { question: 1 }, { question: "ok", extra: "no" }]) test(`existing typed binding refusal ${JSON.stringify(values)}`, () => assert.throws(() => renderConsumed(consume(Buffer.from(server.body_utf8)), values)));
test("accessor/proxy metadata never executes; mutable descriptors and forged snapshots cannot substitute", () => {
  let reads = 0;
  const accessor = Object.defineProperty({ expectedBuildId: server.body_sha256, expectedSizeBytes: server.body_size_bytes }, "source", { get() { reads++; return source; }, enumerable: true });
  assert.throws(() => consumePinnedBuild(Buffer.from(server.body_utf8), accessor as never));
  const proxy = new Proxy(source, { getPrototypeOf() { reads++; return null; } });
  assert.throws(() => consume(Buffer.from(server.body_utf8), { source: proxy })); assert.equal(reads, 0);
  const declared = { ...source }, build = consume(Buffer.from(server.body_utf8), { source: declared }); declared.partition = "changed";
  assert.equal(build.source.partition, "fixture:local");
  assert.throws(() => renderConsumed({ ...build, build_id: "0".repeat(64) }, { question: "x" }));
});
test("byte-carrier own accessors cannot replace native bounds or execute during its copy", () => {
  const raw = new Uint8Array(Buffer.from(server.body_utf8)); let reads = 0;
  for (const name of ["byteLength", "length", "buffer", "byteOffset", "constructor"]) Object.defineProperty(raw, name, { get() { reads++; throw new Error("caller_getter"); } });
  const build = consumePinnedBuild(raw, { expectedBuildId: server.body_sha256, expectedSizeBytes: server.body_size_bytes, source });
  assert.equal(build.build_id, server.body_sha256); assert.equal(reads, 0);
});
for (const ending of ["\n", "\r", "\u2028", "\u2029"]) test(`complete scalar/locator grammar rejects trailing line ending ${JSON.stringify(ending)}`, () => {
  assert.throws(() => consume(Buffer.from(server.body_utf8), { expectedBuildId: server.body_sha256 + ending }));
  assert.throws(() => consume(Buffer.from(server.body_utf8), { source: { ...source, partition: "local" + ending } }));
  const reference = { project_id: "a".repeat(32), surface_resource_id: "b".repeat(32), resource_id: "e".repeat(32), version_id: "f".repeat(32), manifest_sha256: "1".repeat(64) };
  const declared = { ...source, reference }, build = consume(Buffer.from(server.body_utf8), { source: declared });
  reference.version_id = "0".repeat(32);
  assert.equal(build.source.reference?.version_id, "f".repeat(32)); assert(Object.isFrozen(build.source.reference));
  assert.throws(() => consume(Buffer.from(server.body_utf8), { source: { ...source, reference: { ...reference, version_id: reference.version_id + ending } } }));
});
test("exact8MiB schema-valid local template retains its bytes without enlarging the cap", () => {
  const wire = JSON.parse(server.body_utf8) as { outputs: { leaves: { kind: string; text: string | null }[] }[]; diagnostics: unknown[] };
  wire.diagnostics = [];
  const leaf = wire.outputs[0].leaves.find(item => item.kind === "text")!;
  leaf.text = ""; leaf.text = "x".repeat(8 * 1024 * 1024 - canonicalBytes(wire).length);
  const raw = canonicalBytes(wire); assert.equal(raw.length, 8 * 1024 * 1024);
  assert.equal(consume(raw).canonical, raw.toString("utf8"));
  assert.throws(() => consume(Buffer.concat([raw, Buffer.from("x")])));
});
