/** Source-checkout conformance CLI consuming the owned private Node SDK. */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { canonicalBytes, consumeBuild, hashDomain, parsePortableJSON, renderBuild } from "../packages/prompt-sdk/src/index.mts";
export { canonicalBytes, consumeBuild, consumeRender, hashDomain, parsePortableJSON, renderBuild } from "../packages/prompt-sdk/src/index.mts";
type J = null | boolean | string | number | J[] | { [key: string]: J };
type O = { [key: string]: J };
const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
function must(value: unknown, code = "fixture_invalid"): asserts value { if (!value) throw new Error(code); }
function object(value: J): O { must(value !== null && typeof value === "object" && !Array.isArray(value)); return value as O; }
function array(value: J): J[] { must(Array.isArray(value)); return value; }
function string(value: J): string { must(typeof value === "string"); return value; }
function integer(value: J): number { must(typeof value === "number" && Number.isSafeInteger(value) && value >= 0); return value; }

function main(paths: string[]): void {
  let byteCases = 0, renderCases = 0;
  for (const path of paths) {
    const fixture = object(parsePortableJSON(readFileSync(path)));
    for (const item of array(fixture.cases)) {
      const entry = object(item);
      if (Object.hasOwn(entry, "canonical_utf8_hex")) {
        must(canonicalBytes(entry.value).toString("hex") === entry.canonical_utf8_hex && hashDomain(entry.value) === entry.sha256, "byte_fixture_mismatch"); byteCases++;
      } else {
        const build = consumeBuild(Buffer.from(string(entry.build_canonical_utf8_hex), "hex").toString("utf8"));
        const rendered = renderBuild(build, entry.values);
        must(build.build_id === entry.build_id && rendered.render_id === entry.render_id && Buffer.from(rendered.canonical).toString("hex") === entry.render_canonical_utf8_hex, "render_fixture_mismatch"); renderCases++;
      }
    }
    if (fixture.source_map_cases) array(fixture.source_map_cases).forEach((item) => {
      const entry = object(item), bytes = Buffer.from(string(entry.text));
      must(bytes.toString("hex") === entry.utf8_hex);
      array(entry.spans).forEach((value) => { const span = object(value); must(decoder.decode(bytes.subarray(integer(span.start), integer(span.end))) === span.text); });
    });
  }
  console.log(JSON.stringify({ byte_cases: byteCases, render_cases: renderCases }));
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { main(process.argv.slice(2)); }
  catch { console.error("prompt_conformance_failed"); process.exitCode = 1; }
}
