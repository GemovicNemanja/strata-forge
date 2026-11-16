/** Exact local pins around the existing kernel; source labels never grant native access. */
import { createHash } from "node:crypto";
import { types as utilTypes } from "node:util";
import { consumeBuild, renderBuild, type ConsumedRender } from "./kernel.mts";

export const MAX_CONSUMED_BUILD_BYTES = 8 * 1024 * 1024;
// The native accessor/copy bypass caller-owned length/buffer getters on a byte carrier.
const arrayBytes = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(Uint8Array.prototype), "byteLength")!.get as (this: Uint8Array) => number;
export interface LocalBuildReference {
  readonly project_id: string;
  readonly surface_resource_id: string;
  readonly resource_id: string;
  readonly version_id: string;
  readonly manifest_sha256: string;
}
export interface LocalBuildSource {
  readonly kind: "caller_bytes";
  readonly partition: string;
  readonly reference: LocalBuildReference | null;
}
export interface PinnedBuild {
  readonly canonical: string;
  readonly build_id: string;
  readonly size_bytes: number;
  readonly source: LocalBuildSource;
}
export interface ConsumeOptions {
  readonly expectedBuildId: string;
  readonly expectedSizeBytes: number;
  readonly source: LocalBuildSource;
}
function fail(code = "consumption_invalid"): never { throw new Error(code); }
function fields(value: unknown, names: string[]): Record<string, unknown> {
  if (value === null || typeof value !== "object" || utilTypes.isProxy(value) || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) fail();
  const keys = Reflect.ownKeys(value);
  if (keys.length !== names.length || keys.some(key => typeof key !== "string" || !names.includes(key))) fail();
  const result: Record<string, unknown> = Object.create(null);
  for (const name of names) {
    const descriptor = Object.getOwnPropertyDescriptor(value, name);
    if (!descriptor || !descriptor.enumerable || !("value" in descriptor)) fail();
    result[name] = descriptor.value;
  }
  return result;
}
function hex(value: unknown, length: number): value is string {
  return typeof value === "string" && value.length === length && !/[^a-f0-9]/.test(value);
}
function sourceSnapshot(value: unknown): LocalBuildSource {
  const source = fields(value, ["kind", "partition", "reference"]);
  if (source.kind !== "caller_bytes" || typeof source.partition !== "string" || source.partition.length < 1 || source.partition.length > 128 || /[^A-Za-z0-9._:-]/.test(source.partition)) fail("consumption_source_invalid");
  let reference: LocalBuildReference | null = null;
  if (source.reference !== null) {
    const ref = fields(source.reference, ["project_id", "surface_resource_id", "resource_id", "version_id", "manifest_sha256"]);
    if (!hex(ref.project_id, 32) || !hex(ref.surface_resource_id, 32) || !hex(ref.resource_id, 32) || !hex(ref.version_id, 32) || !hex(ref.manifest_sha256, 64)) fail("consumption_source_invalid");
    reference = Object.freeze({ project_id: ref.project_id, surface_resource_id: ref.surface_resource_id, resource_id: ref.resource_id, version_id: ref.version_id, manifest_sha256: ref.manifest_sha256 });
  }
  return Object.freeze({ kind: "caller_bytes", partition: source.partition, reference });
}
export function consumePinnedBuild(raw: Uint8Array, options: ConsumeOptions): PinnedBuild {
  const selected = fields(options, ["expectedBuildId", "expectedSizeBytes", "source"]);
  if (!hex(selected.expectedBuildId, 64) || typeof selected.expectedSizeBytes !== "number" || !Number.isSafeInteger(selected.expectedSizeBytes) || selected.expectedSizeBytes < 1 || selected.expectedSizeBytes > MAX_CONSUMED_BUILD_BYTES) fail();
  if (raw === null || typeof raw !== "object" || utilTypes.isProxy(raw) || !utilTypes.isUint8Array(raw) || ![Uint8Array.prototype, Buffer.prototype].includes(Object.getPrototypeOf(raw))) fail();
  if (arrayBytes.call(raw) !== selected.expectedSizeBytes) fail("consumption_pin_mismatch");
  const source = sourceSnapshot(selected.source), snapshot = new Uint8Array(selected.expectedSizeBytes);
  Uint8Array.prototype.set.call(snapshot, raw);
  const bytes = Buffer.from(snapshot.buffer);
  if (createHash("sha256").update(bytes).digest("hex") !== selected.expectedBuildId) fail("consumption_pin_mismatch");
  let canonical: string | undefined;
  try { canonical = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes); } catch { /* The public boundary emits a fixed error below. */ }
  if (canonical === undefined) fail();
  const build = consumeBuild(canonical);
  return Object.freeze({ canonical: build.canonical, build_id: build.build_id, size_bytes: bytes.length, source });
}
export function renderConsumed(build: PinnedBuild, values: unknown): ConsumedRender {
  const snapshot = fields(build, ["canonical", "build_id", "size_bytes", "source"]);
  if (typeof snapshot.canonical !== "string" || snapshot.canonical.length > MAX_CONSUMED_BUILD_BYTES || Buffer.byteLength(snapshot.canonical) > MAX_CONSUMED_BUILD_BYTES) fail();
  const checked = consumePinnedBuild(Buffer.from(snapshot.canonical), {
    expectedBuildId: snapshot.build_id as string, expectedSizeBytes: snapshot.size_bytes as number,
    source: snapshot.source as LocalBuildSource,
  });
  return renderBuild(checked, values);
}
