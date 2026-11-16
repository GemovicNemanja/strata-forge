/** Independent typed-recipe vectors only; no Node provider adapter or wire capture. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { canonicalBytes } from "../src/index.mts";
interface Recipe { kind: string; value?: unknown; decimal?: string; bits?: string; items?: Recipe[]; entries?: [string, Recipe][] }
const fixture = JSON.parse(readFileSync(new URL("../../../tests/fixtures/prompts/provider-input-v1.json", import.meta.url), "utf8")) as { vectors: { name: string; native_recipe: Recipe; tagged_utf8_hex: string; tagged_sha256: string }[]; projections: { name: string; method: string; model_recipe: Recipe; arguments_recipe: Recipe; projection: { canonical_tagged_sha256: string; encoded_size_bytes: number; selected_present_fields_count: number; omitted_fields_count: number }; framed_utf8_hex: string }[] };
function tag(recipe: Recipe): unknown[] {
  if (recipe.kind === "null") return ["null"];
  if (recipe.kind === "boolean") { assert.equal(typeof recipe.value, "boolean"); return ["bool", recipe.value]; }
  if (recipe.kind === "string") { assert.equal(typeof recipe.value, "string"); return ["string", recipe.value]; }
  if (recipe.kind === "integer") { const value = BigInt(recipe.decimal!); assert.equal(value.toString(), recipe.decimal); return ["integer", value.toString()]; }
  if (recipe.kind === "binary64") {
    const bytes = Buffer.from(recipe.bits!, "hex"); assert.equal(bytes.length, 8);
    const value = bytes.readDoubleBE(); assert(Number.isFinite(value)); const encoded = Buffer.alloc(8); encoded.writeDoubleBE(value); assert.equal(encoded.toString("hex"), recipe.bits);
    return ["binary64", encoded.toString("hex")];
  }
  if (recipe.kind === "array") return ["array", recipe.items!.map(tag)];
  assert.equal(recipe.kind, "object"); return ["object", recipe.entries!.map(([key, child]) => [key, tag(child)])];
}
const sha = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
for (const vector of fixture.vectors) test(`explicit Python/Node typed value ${vector.name}`, () => {
  const bytes = canonicalBytes(tag(vector.native_recipe)); assert.equal(bytes.toString("hex"), vector.tagged_utf8_hex); assert.equal(sha(bytes), vector.tagged_sha256);
});
const chat = ["messages", "temperature", "max_tokens", "top_p", "tools", "response_format", "seed", "stop", "tool_choice", "parallel_tool_calls"];
const responses = ["input", "instructions", "tools", "temperature", "max_output_tokens", "top_p", "text", "reasoning", "include", "store", "parallel_tool_calls", "truncation"];
for (const vector of fixture.projections) test(`allowlisted field/presence fixture ${vector.name}`, () => {
  assert.equal(vector.arguments_recipe.kind, "object"); const argumentsByName = new Map(vector.arguments_recipe.entries!);
  assert(["acompletion", "aresponses"].includes(vector.method)); const names = vector.method === "acompletion" ? chat : responses;
  const fields = names.map(name => [name, argumentsByName.has(name) ? ["present", tag(argumentsByName.get(name)!)] : ["absent"]]);
  const frame = { domain: "strata-provider-input-projection-v1", profile: "forge-provider-client-input/v1", method: vector.method, provider_model_id: tag(vector.model_recipe), fields };
  const raw = canonicalBytes(frame); assert.equal(raw.toString("hex"), vector.framed_utf8_hex); assert.equal(sha(raw), vector.projection.canonical_tagged_sha256); assert.equal(raw.length, vector.projection.encoded_size_bytes);
  const present = names.filter(name => argumentsByName.has(name)).length; assert.equal(present, vector.projection.selected_present_fields_count); assert.equal(argumentsByName.size - present, vector.projection.omitted_fields_count);
});
