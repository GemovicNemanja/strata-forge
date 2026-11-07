/** Explicit local metadata-only byte observations; no provider, credentials, export or authority. */
import { createHash, randomUUID } from "node:crypto";
import { inspect, types as utilTypes } from "node:util";
import { canonicalBytes, parsePortableJSON, type ConsumedRender } from "./kernel.mts";
import { renderConsumed, type LocalBuildReference, type PinnedBuild } from "./consumption.mts";

export const MAX_ATTEMPTS = 64;
export const MAX_BODY_BYTES = 16 * 1024 * 1024;
export const MAX_ATTEMPT_METADATA_BYTES = 4096;
export const MAX_INVOCATION_METADATA_BYTES = 256 * 1024;
export type CaptureCode = "capture_invalid" | "capture_closed" | "capture_in_progress" | "capture_scope_invalid" | "capture_limit" | "local_transport_timeout" | "local_transport_failed" | "capture_response_invalid" | "capture_response_limit" | "local_cancelled" | "local_interrupted";
export type AttemptGap = "render_projection_unobserved" | "provider_boundary_unobserved" | "body_not_retained" | "request_not_dispatched" | "response_not_observed" | "response_oversize" | "late_return_discarded";
export interface AttemptParents { readonly retry_of: string | null; readonly fallback_of: string | null; readonly continuation_of: string | null }
export interface DeclaredBodyPin { readonly sha256: string; readonly size_bytes: number }
export interface ByteObservation {
  readonly profile: "opaque-application-bytes/v1"; readonly coverage: "complete" | "omitted_oversize";
  readonly size_bytes: number; readonly original_sha256: string | null; readonly redacted_sha256: null; readonly retained: false;
}
export interface AttemptMetadata {
  readonly domain: "strata-prompt-attempt-v1"; readonly version: 1; readonly invocation_id: string; readonly attempt_id: string;
  readonly source_kind: "caller_bytes"; readonly build_id: string; readonly render_id: string; readonly input_fingerprint: string;
  readonly parents: AttemptParents; readonly boundary: "sdk_local_delegate_bytes/v1"; readonly observation_source: "instrumented_local";
  readonly started_at: string; readonly retired_at: string | null; readonly ended_at: string; readonly duration_us: number;
  readonly facts: { readonly retrieved: true; readonly rendered: true; readonly request_captured: boolean; readonly response_observed: boolean };
  readonly request: ByteObservation | null; readonly response: ByteObservation | null; readonly delegate_status: number | null;
  readonly declared_baseline: DeclaredBodyPin | null; readonly baseline_relation: "not_supplied" | "not_observed" | "same_bytes" | "changed_bytes";
  readonly render_relation: "unknown"; readonly provider: null; readonly model: null; readonly model_revision: null; readonly usage: null; readonly cost: null;
  readonly outcome: "observed_response" | "transport_error" | "capture_error" | "cancelled" | "interrupted";
  readonly error_code: CaptureCode | null; readonly retention: "metadata_only"; readonly export_status: "not_configured"; readonly gaps: readonly AttemptGap[];
}
export interface InvocationMetadata {
  readonly domain: "strata-prompt-invocation-v1"; readonly version: 1; readonly invocation_id: string; readonly verified_at: string;
  readonly source_kind: "caller_bytes"; readonly source_partition: string; readonly source_reference: LocalBuildReference | null;
  readonly build_id: string; readonly build_size_bytes: number; readonly render_id: string; readonly input_fingerprint: string;
  readonly output_kind: "chat" | "artifact"; readonly output_count: number; readonly source_span_count: number;
  readonly renderer_profile: "strata-modular-compiler/1"; readonly portable_profile: "strata-prompt-json-v1";
  readonly retention: "metadata_only"; readonly attempts: readonly AttemptMetadata[]; readonly closed: boolean; readonly export_status: "not_configured";
}
const codes: readonly string[] = ["capture_invalid", "capture_closed", "capture_in_progress", "capture_scope_invalid", "capture_limit", "local_transport_timeout", "local_transport_failed", "capture_response_invalid", "capture_response_limit", "local_cancelled", "local_interrupted"];
const gaps: readonly string[] = ["render_projection_unobserved", "provider_boundary_unobserved", "body_not_retained", "request_not_dispatched", "response_not_observed", "response_oversize", "late_return_discarded"];
const errorCodes = new WeakMap<PromptCaptureError, CaptureCode>();
export class PromptCaptureError extends Error {
  readonly code: CaptureCode;
  readonly attempt_id: string | null;
  constructor(code: CaptureCode, attemptId: string | null = null) {
    const safe = typeof code === "string" && codes.includes(code) && (attemptId === null || hex(attemptId, 32));
    super(safe ? code : "capture_invalid");
    this.code = safe ? code : "capture_invalid"; this.name = this.code === "local_cancelled" ? "AbortError" : "PromptCaptureError"; this.attempt_id = safe ? attemptId : null;
    errorCodes.set(this, this.code);
    Object.freeze(this);
  }
}
function fail(code: CaptureCode = "capture_invalid", id: string | null = null): never { throw new PromptCaptureError(code, id); }
function must(value: unknown): asserts value { if (!value) fail(); }
function hex(value: unknown, length: number): value is string { return typeof value === "string" && value.length === length && !/[^a-f0-9]/.test(value); }
function count(value: unknown, max = Number.MAX_SAFE_INTEGER): value is number { return typeof value === "number" && Number.isSafeInteger(value) && !Object.is(value, -0) && value >= 0 && value <= max; }
function fields(value: unknown, names: readonly string[]): Record<string, unknown> {
  must(value !== null && typeof value === "object" && !utilTypes.isProxy(value) && [Object.prototype, null].includes(Object.getPrototypeOf(value)));
  const keys = Reflect.ownKeys(value);
  must(keys.length === names.length && keys.every(key => typeof key === "string" && names.includes(key)));
  const result: Record<string, unknown> = Object.create(null);
  for (const name of names) {
    const descriptor = Object.getOwnPropertyDescriptor(value, name);
    must(descriptor && descriptor.enumerable && "value" in descriptor); result[name] = descriptor.value;
  }
  return result;
}
function freeze<T>(value: T): T {
  if (value !== null && typeof value === "object") {
    for (const item of Object.values(value)) freeze(item);
    Object.freeze(value);
  }
  return value;
}
function copyMetadata(value: unknown, cap: number): unknown {
  let raw: Buffer | undefined;
  try { raw = canonicalBytes(value); } catch { /* Do not retain native or input diagnostics. */ }
  if (raw === undefined) fail();
  if (raw.length > cap) fail("capture_limit");
  return parsePortableJSON(raw);
}
function utc(value: unknown): void {
  must(typeof value === "string" && value.length === 27 && /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$/.test(value));
  const year = Number(value.slice(0, 4)), month = Number(value.slice(5, 7)), day = Number(value.slice(8, 10));
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  must(year >= 1 && month >= 1 && month <= 12 && day >= 1 && day <= days[month - 1] && Number(value.slice(11, 13)) <= 23 && Number(value.slice(14, 16)) <= 59 && Number(value.slice(17, 19)) <= 59);
}
function parents(value: unknown): AttemptParents {
  const result = fields(value, ["retry_of", "fallback_of", "continuation_of"]);
  const all = Object.values(result); must(all.every(item => item === null || hex(item, 32)) && all.filter(item => item !== null).length <= 1);
  return freeze(result) as unknown as AttemptParents;
}
function baseline(value: unknown): DeclaredBodyPin | null {
  if (value === null) return null;
  const result = fields(value, ["sha256", "size_bytes"]); must(hex(result.sha256, 64) && count(result.size_bytes, MAX_BODY_BYTES));
  return freeze(result) as unknown as DeclaredBodyPin;
}
function observation(value: unknown): ByteObservation | null {
  if (value === null) return null;
  const result = fields(value, ["profile", "coverage", "size_bytes", "original_sha256", "redacted_sha256", "retained"]);
  must(result.profile === "opaque-application-bytes/v1" && count(result.size_bytes) && result.redacted_sha256 === null && result.retained === false);
  must((result.coverage === "complete" && result.size_bytes <= MAX_BODY_BYTES && hex(result.original_sha256, 64)) || (result.coverage === "omitted_oversize" && result.size_bytes > MAX_BODY_BYTES && result.original_sha256 === null));
  return result as unknown as ByteObservation;
}
const attemptKeys = ["domain", "version", "invocation_id", "attempt_id", "source_kind", "build_id", "render_id", "input_fingerprint", "parents", "boundary", "observation_source", "started_at", "retired_at", "ended_at", "duration_us", "facts", "request", "response", "delegate_status", "declared_baseline", "baseline_relation", "render_relation", "provider", "model", "model_revision", "usage", "cost", "outcome", "error_code", "retention", "export_status", "gaps"];
export function validateAttemptMetadata(value: unknown): AttemptMetadata {
  const item = fields(copyMetadata(value, MAX_ATTEMPT_METADATA_BYTES), attemptKeys);
  must(item.domain === "strata-prompt-attempt-v1" && item.version === 1 && hex(item.invocation_id, 32) && hex(item.attempt_id, 32));
  must(item.source_kind === "caller_bytes" && hex(item.build_id, 64) && hex(item.render_id, 64) && hex(item.input_fingerprint, 64));
  parents(item.parents); const pin = baseline(item.declared_baseline), request = observation(item.request), response = observation(item.response);
  must(item.boundary === "sdk_local_delegate_bytes/v1" && item.observation_source === "instrumented_local");
  utc(item.started_at); utc(item.ended_at); if (item.retired_at !== null) utc(item.retired_at); must(count(item.duration_us));
  const facts = fields(item.facts, ["retrieved", "rendered", "request_captured", "response_observed"]);
  must(facts.retrieved === true && facts.rendered === true && facts.request_captured === (request !== null) && facts.response_observed === (response !== null));
  must((request === null || request.coverage === "complete") && (response === null || request !== null));
  if (item.outcome === "transport_error" || item.outcome === "capture_error") must(request !== null);
  const relation = pin === null ? "not_supplied" : request === null ? "not_observed" : pin.sha256 === request.original_sha256 && pin.size_bytes === request.size_bytes ? "same_bytes" : "changed_bytes";
  must(item.baseline_relation === relation && item.render_relation === "unknown");
  must(item.delegate_status === null || count(item.delegate_status, 999));
  for (const name of ["provider", "model", "model_revision", "usage", "cost"]) must(item[name] === null);
  must(item.retention === "metadata_only" && item.export_status === "not_configured");
  must(Array.isArray(item.gaps) && item.gaps.length <= 7 && item.gaps.every(gap => typeof gap === "string" && gaps.includes(gap)) && new Set(item.gaps).size === item.gaps.length);
  const selected = new Set(item.gaps);
  for (const gap of gaps.slice(0, 3)) must(selected.has(gap));
  must(selected.has("request_not_dispatched") === (request === null));
  must(selected.has("response_not_observed") === (response === null));
  must(selected.has("response_oversize") === (response?.coverage === "omitted_oversize"));
  if (response === null) must(item.delegate_status === null);
  const retired = item.outcome === "cancelled" || item.outcome === "interrupted";
  must(retired === (item.retired_at !== null));
  if (selected.has("late_return_discarded")) must(retired);
  if (item.outcome === "observed_response") must(response?.coverage === "complete" && item.error_code === null);
  else if (item.outcome === "capture_error") must((item.error_code === "capture_response_limit" && response?.coverage === "omitted_oversize") || (item.error_code === "capture_response_invalid" && response === null));
  else if (item.outcome === "transport_error") must(response === null && ["local_transport_timeout", "local_transport_failed"].includes(item.error_code as string));
  else if (item.outcome === "cancelled") must(response === null && item.error_code === "local_cancelled");
  else if (item.outcome === "interrupted") must(response === null && item.error_code === "local_interrupted");
  else fail();
  return freeze(item) as unknown as AttemptMetadata;
}
const invocationKeys = ["domain", "version", "invocation_id", "verified_at", "source_kind", "source_partition", "source_reference", "build_id", "build_size_bytes", "render_id", "input_fingerprint", "output_kind", "output_count", "source_span_count", "renderer_profile", "portable_profile", "retention", "attempts", "closed", "export_status"];
export function validateInvocationMetadata(value: unknown): InvocationMetadata {
  const item = fields(copyMetadata(value, MAX_INVOCATION_METADATA_BYTES), invocationKeys);
  must(item.domain === "strata-prompt-invocation-v1" && item.version === 1 && hex(item.invocation_id, 32)); utc(item.verified_at);
  must(item.source_kind === "caller_bytes" && typeof item.source_partition === "string" && item.source_partition.length > 0 && item.source_partition.length <= 128 && !/[^A-Za-z0-9._:-]/.test(item.source_partition));
  if (item.source_reference !== null) {
    const ref = fields(item.source_reference, ["project_id", "surface_resource_id", "resource_id", "version_id", "manifest_sha256"]);
    for (const name of ["project_id", "surface_resource_id", "resource_id", "version_id"]) must(hex(ref[name], 32)); must(hex(ref.manifest_sha256, 64));
  }
  must(hex(item.build_id, 64) && count(item.build_size_bytes, 8 * 1024 * 1024) && item.build_size_bytes >= 1 && hex(item.render_id, 64) && hex(item.input_fingerprint, 64));
  must(["chat", "artifact"].includes(item.output_kind as string) && count(item.output_count, 128) && item.output_count >= 1 && count(item.source_span_count, 32768));
  must(item.renderer_profile === "strata-modular-compiler/1" && item.portable_profile === "strata-prompt-json-v1" && item.retention === "metadata_only" && item.export_status === "not_configured" && typeof item.closed === "boolean");
  must(Array.isArray(item.attempts) && item.attempts.length <= MAX_ATTEMPTS);
  const seen = new Set<string>();
  for (const raw of item.attempts) {
    const attempt = validateAttemptMetadata(raw);
    must(attempt.invocation_id === item.invocation_id && attempt.build_id === item.build_id && attempt.render_id === item.render_id && attempt.input_fingerprint === item.input_fingerprint && !seen.has(attempt.attempt_id));
    for (const id of Object.values(attempt.parents)) must(id === null || seen.has(id)); seen.add(attempt.attempt_id);
  }
  return freeze(item) as unknown as InvocationMetadata;
}

const arrayBytes = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(Uint8Array.prototype), "byteLength")!.get as (this: Uint8Array) => number;
const byteSet = Uint8Array.prototype.set;
function byteSize(value: unknown): number {
  must(value !== null && typeof value === "object" && !utilTypes.isProxy(value) && utilTypes.isUint8Array(value) && [Uint8Array.prototype, Buffer.prototype].includes(Object.getPrototypeOf(value)));
  let size: number | undefined;
  try { size = arrayBytes.call(value as Uint8Array); } catch { /* Closed byte error below. */ }
  if (size === undefined) fail();
  return size;
}
function byteCopy(value: Uint8Array, size: number): Buffer {
  const copy = new Uint8Array(size); let valid = false;
  try { byteSet.call(copy, value); valid = true; } catch { /* Detached carriers never become empty observations. */ }
  if (!valid) fail();
  return Buffer.from(copy.buffer);
}
const hash = (value: Uint8Array) => createHash("sha256").update(value).digest("hex");
const now = () => new Date().toISOString().replace(/\.([0-9]{3})Z$/, ".$1000Z");
const identifier = () => randomUUID().replaceAll("-", "");
const observed = (body: Uint8Array): ByteObservation => Object.freeze({ profile: "opaque-application-bytes/v1", coverage: "complete", size_bytes: body.length, original_sha256: hash(body), redacted_sha256: null, retained: false });
interface ResponseState { readonly body: Uint8Array; readonly size: number; readonly status: number | null }
const responseStates = new WeakMap<LocalByteResponse, ResponseState>();
const exchangeStates = new WeakMap<CapturedExchange, { response: LocalByteResponse; metadata: AttemptMetadata }>();
interface Ticket {
  readonly attemptId: string; readonly startedAt: string; readonly start: bigint; readonly parents: AttemptParents; readonly baseline: DeclaredBodyPin | null; readonly request: ByteObservation;
  readonly controller: AbortController; handed: boolean; retiredAt: string | null; settled: boolean;
}
interface InvocationState { metadata: InvocationMetadata; render: ConsumedRender | null; active: Ticket | null }
const invocationStates = new WeakMap<PreparedInvocation, InvocationState>();
function invocationState(value: unknown): InvocationState {
  if (value === null || typeof value !== "object" || utilTypes.isProxy(value) || Object.getPrototypeOf(value) !== PreparedInvocation.prototype) fail();
  const state = invocationStates.get(value as PreparedInvocation); if (!state) fail(); return state;
}
function responseState(value: unknown): ResponseState {
  if (value === null || typeof value !== "object" || utilTypes.isProxy(value) || Object.getPrototypeOf(value) !== LocalByteResponse.prototype) fail("capture_response_invalid");
  const state = responseStates.get(value as LocalByteResponse); if (!state) fail("capture_response_invalid"); return state;
}
function exchangeState(value: unknown): { response: LocalByteResponse; metadata: AttemptMetadata } {
  if (value === null || typeof value !== "object" || utilTypes.isProxy(value) || Object.getPrototypeOf(value) !== CapturedExchange.prototype) fail();
  const state = exchangeStates.get(value as CapturedExchange); if (!state) fail(); return state;
}
export class LocalByteResponse {
  constructor(body: Uint8Array, status: number | null) {
    must(new.target === LocalByteResponse && (status === null || count(status, 999)));
    // Oversize returns need only a bounded native size observation, never a full body copy/hash.
    const size = byteSize(body);
    responseStates.set(this, { body: size > MAX_BODY_BYTES ? body : byteCopy(body, size), size, status }); Object.freeze(this);
  }
  get status(): number | null { return responseState(this).status; }
  responseBody(): Uint8Array { const value = responseState(this); must(byteSize(value.body) === value.size); return byteCopy(value.body, value.size); }
  toJSON(): never { return fail(); }
  [inspect.custom](): string { const value = responseStates.get(this); return value ? `LocalByteResponse(size_bytes=${value.size}, status=${value.status})` : "LocalByteResponse(unissued)"; }
}
export class PreparedInvocation {
  constructor() { fail(); }
  rendered(): ConsumedRender { const state = invocationState(this); if (!state.render) fail("capture_closed"); return Object.freeze({ ...state.render }); }
  snapshot(): InvocationMetadata { const state = invocationState(this); if (state.active) fail("capture_in_progress"); return validateInvocationMetadata(state.metadata); }
  close(): void { const state = invocationState(this); if (state.active) fail("capture_in_progress"); if (!state.metadata.closed) { state.render = null; state.metadata = validateInvocationMetadata({ ...state.metadata, closed: true }); } }
  toJSON(): never { return fail(); }
  [inspect.custom](): string { const value = invocationStates.get(this); return value ? `PreparedInvocation(invocation_id=${value.metadata.invocation_id}, attempts=${value.metadata.attempts.length}, active=${value.active !== null}, closed=${value.metadata.closed})` : "PreparedInvocation(unissued)"; }
}
export class CapturedExchange {
  constructor() { fail(); }
  get attempt(): AttemptMetadata { return validateAttemptMetadata(exchangeState(this).metadata); }
  get status(): number | null { return responseState(exchangeState(this).response).status; }
  responseBody(): Uint8Array { return exchangeState(this).response.responseBody(); }
  toJSON(): never { return fail(); }
  [inspect.custom](): string { const value = exchangeStates.get(this); return value ? `CapturedExchange(attempt_id=${value.metadata.attempt_id}, status=${responseState(value.response).status})` : "CapturedExchange(unissued)"; }
}
export interface LocalByteTransport { exchange(body: Uint8Array, options: { readonly signal: AbortSignal }): Promise<LocalByteResponse> }
export async function prepareInvocation(build: PinnedBuild, values: unknown, retention: "metadata_only" = "metadata_only"): Promise<PreparedInvocation> {
  if (retention !== "metadata_only") fail();
  const rendered = renderConsumed(build, values), wire = parsePortableJSON(rendered.canonical) as Record<string, unknown>;
  const output = wire.output as { kind: "chat" | "artifact"; messages?: unknown[] };
  const metadata = validateInvocationMetadata({
    domain: "strata-prompt-invocation-v1", version: 1, invocation_id: identifier(), verified_at: now(), source_kind: "caller_bytes",
    source_partition: build.source.partition, source_reference: build.source.reference, build_id: build.build_id, build_size_bytes: build.size_bytes,
    render_id: rendered.render_id, input_fingerprint: wire.input_fingerprint, output_kind: output.kind, output_count: output.kind === "chat" ? output.messages!.length : 1,
    source_span_count: (wire.source_map as unknown[]).length, renderer_profile: "strata-modular-compiler/1", portable_profile: "strata-prompt-json-v1", retention: "metadata_only", attempts: [], closed: false, export_status: "not_configured",
  });
  const result = Object.create(PreparedInvocation.prototype) as PreparedInvocation;
  invocationStates.set(result, { metadata, render: rendered, active: null }); return Object.freeze(result);
}
function record(state: InvocationState, ticket: Ticket, outcome: AttemptMetadata["outcome"], error: CaptureCode | null, response: ByteObservation | null = null, status: number | null = null, late = false): AttemptMetadata {
  if (state.active !== ticket) fail("capture_scope_invalid");
  const request = ticket.handed ? ticket.request : null;
  const selectedGaps: AttemptGap[] = ["render_projection_unobserved", "provider_boundary_unobserved", "body_not_retained"];
  if (request === null) selectedGaps.push("request_not_dispatched"); if (response === null) selectedGaps.push("response_not_observed");
  if (response?.coverage === "omitted_oversize") selectedGaps.push("response_oversize"); if (late) selectedGaps.push("late_return_discarded");
  const relation = ticket.baseline === null ? "not_supplied" : request === null ? "not_observed" : ticket.baseline.sha256 === request.original_sha256 && ticket.baseline.size_bytes === request.size_bytes ? "same_bytes" : "changed_bytes";
  const meta = state.metadata;
  const attempt = validateAttemptMetadata({
    domain: "strata-prompt-attempt-v1", version: 1, invocation_id: meta.invocation_id, attempt_id: ticket.attemptId, source_kind: "caller_bytes", build_id: meta.build_id, render_id: meta.render_id, input_fingerprint: meta.input_fingerprint,
    parents: ticket.parents, boundary: "sdk_local_delegate_bytes/v1", observation_source: "instrumented_local", started_at: ticket.startedAt, retired_at: ticket.retiredAt, ended_at: now(), duration_us: Number((process.hrtime.bigint() - ticket.start) / 1000n),
    facts: { retrieved: true, rendered: true, request_captured: request !== null, response_observed: response !== null }, request, response, delegate_status: status,
    declared_baseline: ticket.baseline, baseline_relation: relation, render_relation: "unknown", provider: null, model: null, model_revision: null, usage: null, cost: null, outcome, error_code: error, retention: "metadata_only", export_status: "not_configured", gaps: selectedGaps,
  });
  state.metadata = validateInvocationMetadata({ ...meta, attempts: [...meta.attempts, attempt] }); state.active = null; return attempt;
}
const aborted = Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")!.get as (this: AbortSignal) => boolean;
const promiseThen = Promise.prototype.then;
const addEvent = EventTarget.prototype.addEventListener, removeEvent = EventTarget.prototype.removeEventListener;
function isAborted(signal: AbortSignal | undefined): boolean {
  if (signal === undefined) return false;
  if (signal === null || typeof signal !== "object" || utilTypes.isProxy(signal)) fail();
  let result: boolean | undefined;
  try { result = aborted.call(signal); } catch { /* A forged signal cannot supply cancellation authority. */ }
  if (result === undefined) fail(); return result;
}
function cancelled(id: string | null): never { const error = new PromptCaptureError("local_cancelled", id); throw error; }
export async function captureOnce(invocation: PreparedInvocation, body: Uint8Array, transport: LocalByteTransport, options: { readonly parents: AttemptParents; readonly declaredBaseline: DeclaredBodyPin | null; readonly signal?: AbortSignal }): Promise<CapturedExchange> {
  const state = invocationState(invocation);
  if (state.metadata.closed) fail("capture_closed"); if (state.active) fail("capture_in_progress");
  // Options are closed own-data properties; the optional signal cannot smuggle other configuration.
  must(options !== null && typeof options === "object" && !utilTypes.isProxy(options));
  const names = Reflect.ownKeys(options); must(names.length === (names.includes("signal") ? 3 : 2));
  const selected = fields(options, names.includes("signal") ? ["parents", "declaredBaseline", "signal"] : ["parents", "declaredBaseline"]);
  const signal = selected.signal as AbortSignal | undefined;
  if (isAborted(signal)) cancelled(null);
  const size = byteSize(body); if (size > MAX_BODY_BYTES) fail("capture_limit");
  const requestBody = byteCopy(body, size), checkedParents = parents(selected.parents), pin = baseline(selected.declaredBaseline);
  // Only the explicitly declared behavior is resolved, once; never inspect configuration/error fields.
  let exchange: unknown;
  try { exchange = transport.exchange; } catch { /* The fixed preflight error has no caller cause. */ }
  if (typeof exchange !== "function") fail();
  const known = new Set(state.metadata.attempts.map(attempt => attempt.attempt_id));
  for (const id of Object.values(checkedParents)) if (id !== null && !known.has(id)) fail();
  if (known.size >= MAX_ATTEMPTS || canonicalBytes(state.metadata).length + MAX_ATTEMPT_METADATA_BYTES > MAX_INVOCATION_METADATA_BYTES) fail("capture_limit");
  const id = identifier(); if (known.has(id)) fail();
  const ticket: Ticket = { attemptId: id, startedAt: now(), start: process.hrtime.bigint(), parents: checkedParents, baseline: pin, request: observed(requestBody), controller: new AbortController(), handed: false, retiredAt: null, settled: false };
  const retire = () => {
    if (ticket.retiredAt === null) { ticket.retiredAt = now(); if (!ticket.settled) ticket.controller.abort("local_cancelled"); }
  };
  state.active = ticket;
  if (signal !== undefined) addEvent.call(signal, "abort", retire, { once: true });
  let raw: unknown, error: CaptureCode | null = null, fulfilled = false;
  try {
    if (isAborted(signal)) retire();
    if (ticket.retiredAt === null) {
      ticket.handed = true;
      // The single delegate promise remains owned and awaited through abort, even if it ignores its signal.
      const returned: unknown = Reflect.apply(exchange, transport, [Buffer.from(requestBody), Object.freeze({ signal: ticket.controller.signal })]);
      if (!utilTypes.isPromise(returned) || Object.getPrototypeOf(returned) !== Promise.prototype || Object.hasOwn(returned, "constructor")) {
        error = "capture_response_invalid";
      } else {
        // Bypass a caller-owned then accessor; this native promise remains observed until settlement.
        raw = await promiseThen.call(returned, (value: unknown) => value); fulfilled = true;
      }
    }
  } catch (failure: unknown) {
    error = failure !== null && typeof failure === "object" && !utilTypes.isProxy(failure) && errorCodes.get(failure as PromptCaptureError) === "local_transport_timeout" ? "local_transport_timeout" : "local_transport_failed";
  } finally { ticket.settled = true; }
  if (isAborted(signal)) retire();
  if (signal !== undefined) removeEvent.call(signal, "abort", retire);
  if (ticket.retiredAt !== null) {
    record(state, ticket, "cancelled", "local_cancelled", null, null, fulfilled); raw = undefined; cancelled(id);
  }
  if (error !== null) { record(state, ticket, error === "capture_response_invalid" ? "capture_error" : "transport_error", error); fail(error, id); }
  let response: ResponseState | undefined;
  try { response = responseState(raw); } catch { /* Native response data is never read on a wrong shape. */ }
  if (!response) { record(state, ticket, "capture_error", "capture_response_invalid"); fail("capture_response_invalid", id); }
  if (response.size > MAX_BODY_BYTES) {
    record(state, ticket, "capture_error", "capture_response_limit", { profile: "opaque-application-bytes/v1", coverage: "omitted_oversize", size_bytes: response.size, original_sha256: null, redacted_sha256: null, retained: false }, response.status); fail("capture_response_limit", id);
  }
  const attempt = record(state, ticket, "observed_response", null, observed(response.body), response.status);
  const result = Object.create(CapturedExchange.prototype) as CapturedExchange;
  exchangeStates.set(result, { response: raw as LocalByteResponse, metadata: attempt }); return Object.freeze(result);
}
