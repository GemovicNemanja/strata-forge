/** Fake-only application bytes and owned promise lifetime; no provider or external I/O. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { inspect } from "node:util";
import test from "node:test";
import {
  captureOnce, consumePinnedBuild, prepareInvocation, LocalByteResponse, PreparedInvocation,
  PromptCaptureError, validateAttemptMetadata, validateInvocationMetadata, canonicalBytes,
  type AttemptParents, type LocalByteTransport, type AttemptMetadata,
} from "../src/index.mts";
const corpus = JSON.parse(readFileSync(new URL("../../../tests/fixtures/prompts/composition-renders-v1.json", import.meta.url), "utf8")) as { cases: { expected_output: { kind: string }; values: unknown; build_canonical_utf8_hex: string; build_id: string; render_id: string }[] };
const fixture = corpus.cases.find(item => item.expected_output.kind === "chat")!;
const noParent: AttemptParents = { retry_of: null, fallback_of: null, continuation_of: null };
const sha = (body: Uint8Array) => createHash("sha256").update(body).digest("hex");
async function prepared() {
  const raw = Buffer.from(fixture.build_canonical_utf8_hex, "hex");
  return prepareInvocation(consumePinnedBuild(raw, { expectedBuildId: fixture.build_id, expectedSizeBytes: raw.length, source: { kind: "caller_bytes", partition: "fixture:attempts", reference: null } }), fixture.values);
}
class Fake implements LocalByteTransport {
  calls: Uint8Array[] = [];
  readonly body: Uint8Array; readonly status: number | null;
  constructor(body: Uint8Array = Buffer.from("BODY CANARY"), status: number | null = 200) { this.body = body; this.status = status; }
  get credentials(): never { throw new Error("CONFIG CANARY"); }
  async exchange(body: Uint8Array) { this.calls.push(body); return new LocalByteResponse(this.body, this.status); }
}
const fixed = (code: string) => (error: unknown) => error instanceof PromptCaptureError && error.message === code && error.cause === undefined;
test("prepare/render and explicit byte handoff are distinct with private body-safe working objects", async () => {
  const invocation = await prepared(); assert.deepEqual(invocation.snapshot().attempts, []); assert.equal(invocation.rendered().render_id, fixture.render_id);
  const body = Buffer.from('{"settings":{"temperature":0.7},"message":"REQUEST CANARY"}'), fake = new Fake();
  const result = await captureOnce(invocation, body, fake, { parents: noParent, declaredBaseline: { sha256: sha(body), size_bytes: body.length } });
  assert.deepEqual(fake.calls, [body]); assert.equal(result.status, 200); assert.equal(Buffer.from(result.responseBody()).toString(), "BODY CANARY");
  assert.equal(result.attempt.baseline_relation, "same_bytes"); assert.equal(result.attempt.render_relation, "unknown");
  assert.equal(result.attempt.request!.original_sha256, sha(body)); assert.equal(result.attempt.provider, null); assert.equal(result.attempt.source_kind, "caller_bytes");
  for (const value of [invocation, result, new LocalByteResponse(Buffer.from("BODY CANARY"), 200)]) {
    assert(!inspect(value, { showHidden: true }).includes("CANARY")); assert(!inspect(value).includes("CANARY"));
    assert.throws(() => JSON.stringify(value), fixed("capture_invalid")); assert.deepEqual(Reflect.ownKeys(value), []);
    assert.deepEqual(structuredClone(value), {});
  }
  const detached = invocation.snapshot(); assert(Object.isFrozen(detached) && Object.isFrozen(detached.attempts));
  assert(!canonicalBytes(detached).toString().includes("CANARY")); invocation.close(); invocation.close(); assert(invocation.snapshot().closed);
  assert.throws(() => invocation.rendered(), fixed("capture_closed"));
  await assert.rejects(captureOnce(invocation, body, fake, { parents: noParent, declaredBaseline: null }), fixed("capture_closed"));
});
test("empty bytes/status zero are actual observations and snapshots preserve native byte copies", async () => {
  const invocation = await prepared(), fake = new Fake(Buffer.alloc(0), 0);
  const result = await captureOnce(invocation, Buffer.alloc(0), fake, { parents: noParent, declaredBaseline: null });
  assert.equal(result.status, 0); assert.equal(result.attempt.response!.size_bytes, 0); assert.equal(result.attempt.response!.original_sha256, sha(Buffer.alloc(0)));
  assert.equal(result.attempt.baseline_relation, "not_supplied"); assert(result.attempt.facts.response_observed);
  const responseBytes = Buffer.from("original"), response = new LocalByteResponse(responseBytes, 200); responseBytes.fill(0);
  const returned = response.responseBody(); returned.fill(0); assert.equal(Buffer.from(response.responseBody()).toString(), "original");
});
test("retry/fallback/continuation are explicit terminal links with no automated call", async () => {
  const invocation = await prepared(), baseline = { sha256: sha(Buffer.from("original")), size_bytes: 8 };
  let last = await captureOnce(invocation, Buffer.from("original"), new Fake(Buffer.from("error"), 429), { parents: noParent, declaredBaseline: baseline });
  assert.equal(last.attempt.outcome, "observed_response"); assert.equal(last.status, 429);
  for (const key of ["retry_of", "retry_of", "fallback_of", "continuation_of"] as const) {
    const next = await captureOnce(invocation, Buffer.from(key), new Fake(), { parents: { ...noParent, [key]: last.attempt.attempt_id }, declaredBaseline: baseline });
    assert.equal(next.attempt.render_id, last.attempt.render_id); assert.notEqual(next.attempt.attempt_id, last.attempt.attempt_id); assert.equal(next.attempt.baseline_relation, "changed_bytes"); last = next;
  }
  assert.equal(invocation.snapshot().attempts.length, 5);
  await assert.rejects(captureOnce(await prepared(), Buffer.from("x"), new Fake(), { parents: { ...noParent, retry_of: last.attempt.attempt_id }, declaredBaseline: null }), fixed("capture_invalid"));
});
for (const failure of [new Error("ERROR CANARY"), new PromptCaptureError("local_transport_timeout")]) test(`fixed thrown failure ${failure instanceof PromptCaptureError ? "timeout" : "transport"}`, async () => {
  const invocation = await prepared();
  const delegate = { async exchange(): Promise<LocalByteResponse> { throw failure; } };
  const code = failure instanceof PromptCaptureError ? "local_transport_timeout" : "local_transport_failed";
  await assert.rejects(captureOnce(invocation, Buffer.from("REQUEST CANARY"), delegate, { parents: noParent, declaredBaseline: null }), fixed(code));
  const meta = invocation.snapshot().attempts[0]; assert.equal(meta.outcome, "transport_error"); assert(meta.facts.request_captured); assert.equal(meta.response, null); assert(!JSON.stringify(meta).includes("CANARY"));
});
test("caller abort drains one uncooperative promise before slot reuse and ignores late body getters", async () => {
  const invocation = await prepared(), caller = new AbortController(); let release!: (value: LocalByteResponse) => void; let aborts = 0, calls = 0, lateReads = 0;
  const delegate: LocalByteTransport = {
    exchange(_body, { signal }) { calls++; signal.addEventListener("abort", () => { aborts++; assert.equal(signal.reason, "local_cancelled"); }); return new Promise(resolve => { release = resolve; }); },
  };
  let settled = false;
  const promise = captureOnce(invocation, Buffer.from("x"), delegate, { parents: noParent, declaredBaseline: null, signal: caller.signal });
  const caught = promise.then(() => { settled = true; assert.fail("retired success"); }, error => { settled = true; assert(fixed("local_cancelled")(error)); });
  assert.equal(calls, 1); caller.abort("CALLER REASON CANARY"); caller.abort("different"); await Promise.resolve();
  assert.equal(aborts, 1); assert.equal(settled, false);
  assert.throws(() => invocation.close(), fixed("capture_in_progress")); assert.throws(() => invocation.snapshot(), fixed("capture_in_progress"));
  await assert.rejects(captureOnce(invocation, Buffer.from("again"), new Fake(), { parents: noParent, declaredBaseline: null }), fixed("capture_in_progress"));
  const late = Object.defineProperty({}, "body", { get() { lateReads++; throw new Error("LATE CANARY"); } });
  release(late as LocalByteResponse); await caught; assert.equal(lateReads, 0);
  const meta = invocation.snapshot().attempts[0]; assert.equal(meta.outcome, "cancelled"); assert.equal(meta.response, null); assert.equal(meta.delegate_status, null); assert(!meta.facts.response_observed); assert(meta.gaps.includes("late_return_discarded"));
  assert((await captureOnce(invocation, Buffer.from("next"), new Fake(), { parents: noParent, declaredBaseline: null })).attempt.facts.response_observed);
});
test("already aborted signal has no ticket and reentrant capture/close cannot reuse slot", async () => {
  const invocation = await prepared(), caller = new AbortController(), fake = new Fake(); caller.abort("CANARY");
  await assert.rejects(captureOnce(invocation, Buffer.from("x"), fake, { parents: noParent, declaredBaseline: null, signal: caller.signal }), fixed("local_cancelled"));
  assert.equal(fake.calls.length, 0); assert.equal(invocation.snapshot().attempts.length, 0);
  const delegate: LocalByteTransport = { async exchange() { assert.throws(() => invocation.close(), fixed("capture_in_progress")); await assert.rejects(captureOnce(invocation, Buffer.from("nested"), fake, { parents: noParent, declaredBaseline: null }), fixed("capture_in_progress")); return new LocalByteResponse(Buffer.from("ok"), 200); } };
  await captureOnce(invocation, Buffer.from("x"), delegate, { parents: noParent, declaredBaseline: null }); assert.equal(invocation.snapshot().attempts.length, 1);
});
for (const shape of [{}, { exchange: 42 }, Object.defineProperty({}, "exchange", { get() { throw new Error("LOOKUP CANARY"); } })]) test("invalid declared behavior refuses before any ticket", async () => {
  const invocation = await prepared();
  await assert.rejects(captureOnce(invocation, Buffer.from("x"), shape as unknown as LocalByteTransport, { parents: noParent, declaredBaseline: null }), fixed("capture_invalid"));
  assert.equal(invocation.snapshot().attempts.length, 0);
});
test("native carriers do not execute length/config getters and forged/proxy/detached carriers refuse", async () => {
  const invocation = await prepared(); let reads = 0;
  const body = new Uint8Array(Buffer.from("actual"));
  for (const key of ["byteLength", "length", "buffer", "byteOffset", "constructor"]) Object.defineProperty(body, key, { get() { reads++; throw new Error("GETTER CANARY"); } });
  const fake = new Fake(); const result = await captureOnce(invocation, body, fake, { parents: noParent, declaredBaseline: null });
  assert.equal(result.attempt.request!.original_sha256, sha(Buffer.from("actual"))); assert.equal(reads, 0);
  const detached = new Uint8Array(1); structuredClone(detached.buffer, { transfer: [detached.buffer] });
  for (const invalid of [Object.create(Uint8Array.prototype), Object.create(Buffer.prototype), new Proxy(new Uint8Array(1), {}), detached]) {
    await assert.rejects(captureOnce(invocation, invalid, fake, { parents: noParent, declaredBaseline: null }), fixed("capture_invalid"));
  }
  assert.equal(fake.calls.length, 1); assert.equal(invocation.snapshot().attempts.length, 1);
  await assert.rejects(captureOnce(structuredClone(invocation), Buffer.from("x"), fake, { parents: noParent, declaredBaseline: null }), fixed("capture_invalid"));
  await assert.rejects(captureOnce(Object.create(PreparedInvocation.prototype), Buffer.from("x"), fake, { parents: noParent, declaredBaseline: null }), fixed("capture_invalid"));
});
test("exact16MiB observes complete originals while oversize refuses/omits with no partial hash", async () => {
  const invocation = await prepared(), body = Buffer.alloc(16 * 1024 * 1024, 120), fake = new Fake(body);
  const result = await captureOnce(invocation, body, fake, { parents: noParent, declaredBaseline: null });
  assert.equal(result.attempt.request!.size_bytes, body.length); assert.equal(result.attempt.response!.original_sha256, sha(body));
  const oversize = Buffer.alloc(body.length + 1, 120);
  await assert.rejects(captureOnce(invocation, oversize, fake, { parents: noParent, declaredBaseline: null }), fixed("capture_limit")); assert.equal(fake.calls.length, 1);
  await assert.rejects(captureOnce(invocation, Buffer.alloc(0), new Fake(oversize), { parents: noParent, declaredBaseline: null }), fixed("capture_response_limit"));
  const meta = invocation.snapshot().attempts[1]; assert(meta.facts.response_observed); assert.equal(meta.response!.coverage, "omitted_oversize"); assert.equal(meta.response!.original_sha256, null);
});
test("64 explicit attempts reserve budget; attempt65 cannot dispatch", async () => {
  const invocation = await prepared(), fake = new Fake(Buffer.alloc(0), 0);
  for (let index = 0; index < 64; index++) await captureOnce(invocation, Buffer.alloc(0), fake, { parents: noParent, declaredBaseline: null });
  assert.equal(invocation.snapshot().attempts.length, 64); assert.equal(fake.calls.length, 64);
  await assert.rejects(captureOnce(invocation, Buffer.alloc(0), fake, { parents: noParent, declaredBaseline: null }), fixed("capture_limit")); assert.equal(fake.calls.length, 64);
});
test("metadata parsers reject forged facts, wrong domains, order, parents, null substitutions and raw accessors", async () => {
  const invocation = await prepared(); const result = await captureOnce(invocation, Buffer.alloc(0), new Fake(), { parents: noParent, declaredBaseline: null }); const meta = result.attempt;
  const mutations: Record<string, unknown>[] = [
    { provider: "CANARY" }, { source_kind: "remote" }, { domain: "wrong" }, { version: true }, { attempt_id: meta.attempt_id + "\n" },
    { response: null }, { request: null }, { duration_us: -1 }, { duration_us: 0.5 }, { started_at: "2026-02-30T00:00:00.000000Z" },
    { baseline_relation: "same_bytes" }, { gaps: [] }, { gaps: [...meta.gaps, meta.gaps[0]] }, { facts: { ...meta.facts, response_observed: false } },
    { response: { ...meta.response, retained: true } }, { response: { ...meta.response, redacted_sha256: "f".repeat(64) } }, { usage: 0 },
    { error_code: "CANARY" }, { retired_at: meta.started_at }, { parents: { retry_of: "a".repeat(32), fallback_of: "b".repeat(32), continuation_of: null } },
  ];
  for (const change of mutations) assert.throws(() => validateAttemptMetadata({ ...meta, ...change }), fixed("capture_invalid"));
  let reads = 0; const hostile = Object.defineProperty({ ...meta }, "usage", { enumerable: true, get() { reads++; return null; } });
  assert.throws(() => validateAttemptMetadata(hostile), fixed("capture_invalid")); assert.equal(reads, 0);
  const ledger = invocation.snapshot();
  for (const changed of [{ attempts: [meta, meta] }, { invocation_id: "a".repeat(32) }, { attempts: [{ ...meta, parents: { ...noParent, retry_of: "b".repeat(32) } }] }, { extra: "CANARY" }]) assert.throws(() => validateInvocationMetadata({ ...ledger, ...changed }), fixed("capture_invalid"));
});

test("abort remains standard-name recognizable with closed error and fixed owned reason", async () => {
  const invocation = await prepared(), controller = new AbortController();
  const delegate: LocalByteTransport = { async exchange(_body, { signal }) { controller.abort("SECRET REASON CANARY"); assert.equal(signal.reason, "local_cancelled"); return new LocalByteResponse(Buffer.from("late"), 200); } };
  await assert.rejects(captureOnce(invocation, Buffer.from("x"), delegate, { parents: noParent, declaredBaseline: null, signal: controller.signal }), (error: unknown) => error instanceof PromptCaptureError && error.name === "AbortError" && error.code === "local_cancelled" && error.message === "local_cancelled" && error.cause === undefined);
  assert.equal(invocation.snapshot().attempts[0].outcome, "cancelled");
});
test("wrong promise/response shape and hostile thenables cannot execute getters or become success", async () => {
  let reads = 0;
  const thenable = Object.defineProperty({}, "then", { get() { reads++; throw new Error("THEN CANARY"); } });
  for (const returned of [thenable, new LocalByteResponse(Buffer.from("sync"), 200), Promise.resolve({}), Promise.resolve(Object.create(LocalByteResponse.prototype))]) {
    const invocation = await prepared();
    const delegate = { exchange() { return returned; } } as unknown as LocalByteTransport;
    await assert.rejects(captureOnce(invocation, Buffer.from("x"), delegate, { parents: noParent, declaredBaseline: null }), fixed("capture_response_invalid"));
    assert.equal(invocation.snapshot().attempts[0].outcome, "capture_error");
  }
  const promise = Promise.resolve(new LocalByteResponse(Buffer.from("actual"), 200));
  Object.defineProperty(promise, "then", { get() { reads++; throw new Error("THEN CANARY"); } });
  const result = await captureOnce(await prepared(), Buffer.from("x"), { exchange() { return promise; } }, { parents: noParent, declaredBaseline: null });
  assert.equal(Buffer.from(result.responseBody()).toString(), "actual"); assert.equal(reads, 0);
});

test("exact gap set and failure handoff facts reject coherent-looking impossible reports", async () => {
  const invocation = await prepared(), result = await captureOnce(invocation, Buffer.from("x"), new Fake(), { parents: noParent, declaredBaseline: null });
  for (const gap of ["request_not_dispatched", "response_not_observed", "response_oversize"]) assert.throws(() => validateAttemptMetadata({ ...result.attempt, gaps: [...result.attempt.gaps, gap] }), fixed("capture_invalid"));
  for (const [outcome, error_code] of [["transport_error", "local_transport_failed"], ["capture_error", "capture_response_invalid"]]) {
    const impossible = { ...result.attempt, request: null, response: null, delegate_status: null, facts: { retrieved: true, rendered: true, request_captured: false, response_observed: false }, outcome, error_code, gaps: ["render_projection_unobserved", "provider_boundary_unobserved", "body_not_retained", "request_not_dispatched", "response_not_observed"] };
    assert.throws(() => validateAttemptMetadata(impossible), fixed("capture_invalid"));
  }
});
