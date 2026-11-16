/** Canonical metadata vectors only; there is no Node provider/stream observation adapter. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { canonicalBytes } from "../src/index.mts";

interface MetadataFixture {
  profile: string;
  vectors: { name: string; metadata: unknown; canonical_utf8_hex: string; metadata_sha256: string }[];
}
const fixture = JSON.parse(readFileSync(new URL("../../../tests/fixtures/prompts/provider-stream-v1.json", import.meta.url), "utf8")) as MetadataFixture;
assert.equal(fixture.profile, "strata-provider-stream-metadata-fixture-v1");
for (const vector of fixture.vectors) test(`body-free stream metadata bytes: ${vector.name}`, () => {
  const raw = canonicalBytes(vector.metadata);
  assert.equal(raw.toString("hex"), vector.canonical_utf8_hex);
  assert.equal(createHash("sha256").update(raw).digest("hex"), vector.metadata_sha256);
});
