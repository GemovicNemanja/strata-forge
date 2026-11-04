/** Reference consumer for Python builds; it does not compile authored modules or publish identity. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { types as utilTypes } from "node:util";

type J = null | boolean | string | number | J[] | { [key: string]: J };
type O = { [key: string]: J };
const MAX_BYTES = 64 * 1024 * 1024;
const RENDER_BYTES = 8 * 1024 * 1024;
const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
function fail(code = "portable_invalid"): never { throw new Error(code); }
function must(value: unknown, code = "build_invalid"): asserts value { if (!value) fail(code); }
function object(value: J): O { must(value !== null && typeof value === "object" && !Array.isArray(value)); return value as O; }
function array(value: J): J[] { must(Array.isArray(value)); return value; }
function string(value: J): string { must(typeof value === "string"); return value; }
function integer(value: J, max = Number.MAX_SAFE_INTEGER): number {
  must(typeof value === "number" && Number.isSafeInteger(value) && !Object.is(value, -0) && value >= 0 && value <= max);
  return value;
}
function keys(value: O, expected: string[]): void {
  must(JSON.stringify(Object.keys(value).sort()) === JSON.stringify(expected.sort()));
}
function scalarString(value: string): void {
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const next = value.charCodeAt(++i);
      if (!(next >= 0xdc00 && next <= 0xdfff)) fail("portable_unicode_invalid");
    } else if (c >= 0xdc00 && c <= 0xdfff) fail("portable_unicode_invalid");
  }
}
function asciiKey(value: string): void { if (!/^[\x20-\x7e]*$/.test(value)) fail("portable_key_invalid"); }

export function canonicalBytes(value: unknown): Buffer {
  let size = 0;
  const ancestors = new Set<object>();
  const consume = (text: string): string => {
    size += Buffer.byteLength(text);
    if (size > MAX_BYTES) fail("portable_bytes_limit");
    return text;
  };
  const encode = (item: unknown, depth: number): string => {
    if (depth > 64) fail("portable_depth_limit");
    if (item === null) return consume("null");
    if (typeof item === "boolean") return consume(item ? "true" : "false");
    if (typeof item === "number") {
      if (!Number.isSafeInteger(item) || Object.is(item, -0)) fail("portable_number_invalid");
      return consume(String(item));
    }
    if (typeof item === "string") {
      if (item.length > MAX_BYTES) fail("portable_bytes_limit");
      scalarString(item);
      let quotedBytes = Buffer.byteLength(item) + 2;
      for (let index = 0; index < item.length; index++) {
        const c = item.charCodeAt(index);
        if (c === 34 || c === 92 || [8, 9, 10, 12, 13].includes(c)) quotedBytes++;
        else if (c < 32) quotedBytes += 5;
      }
      if (size + quotedBytes > MAX_BYTES) fail("portable_bytes_limit");
      return consume(JSON.stringify(item));
    }
    if (typeof item !== "object") fail("portable_type_invalid");
    if (utilTypes.isProxy(item)) fail("portable_type_invalid");
    if (ancestors.has(item)) fail("portable_cycle");
    ancestors.add(item);
    let result: string;
    if (Array.isArray(item)) {
      if (Object.getPrototypeOf(item) !== Array.prototype) fail("portable_type_invalid");
      if (item.length > MAX_BYTES / 2) fail("portable_bytes_limit");
      const entries: string[] = [];
      consume("[]" + ",".repeat(Math.max(0, item.length - 1)));
      for (let index = 0; index < item.length; index++) {
        const descriptor = Object.getOwnPropertyDescriptor(item, index);
        if (!descriptor || descriptor.get || descriptor.set) fail("portable_type_invalid");
        entries.push(encode(descriptor.value, depth + 1));
      }
      if (Object.keys(item).length !== item.length || Object.getOwnPropertyNames(item).length !== item.length + 1 || Object.getOwnPropertySymbols(item).length) fail("portable_type_invalid");
      result = "[" + entries.join(",") + "]";
    } else {
      if (![Object.prototype, null].includes(Object.getPrototypeOf(item)) || Object.getOwnPropertySymbols(item).length) fail("portable_type_invalid");
      const names = Object.getOwnPropertyNames(item).sort();
      if (names.length > MAX_BYTES / 4) fail("portable_bytes_limit");
      consume("{}" + ",".repeat(Math.max(0, names.length - 1)) + ":".repeat(names.length));
      result = "{" + names.map((name) => {
        asciiKey(name);
        const descriptor = Object.getOwnPropertyDescriptor(item, name)!;
        if (descriptor.get || descriptor.set || !descriptor.enumerable) fail("portable_type_invalid");
        return encode(name, depth + 1) + ":" + encode(descriptor.value, depth + 1);
      }).join(",") + "}";
    }
    ancestors.delete(item);
    return result;
  };
  return Buffer.from(encode(value, 0), "utf8");
}

export function parsePortableJSON(input: string | Uint8Array): J {
  let raw: string;
  try {
    if (typeof input === "string") { scalarString(input); raw = input; }
    else {
      if (input[0] === 0xef && input[1] === 0xbb && input[2] === 0xbf) fail();
      raw = decoder.decode(input);
    }
  } catch { fail("portable_json_invalid"); }
  if (Buffer.byteLength(raw) > MAX_BYTES) fail("portable_bytes_limit");
  let at = 0;
  const whitespace = () => { while (/[\x20\t\r\n]/.test(raw[at] ?? "") && at < raw.length) at++; };
  const literalString = (): string => {
    const start = at++;
    while (at < raw.length) {
      const char = raw[at++];
      if (char === "\\") { at++; continue; }
      if (char === '"') {
        let parsed: unknown;
        try { parsed = JSON.parse(raw.slice(start, at)); } catch { fail(); }
        if (typeof parsed !== "string") fail();
        scalarString(parsed);
        return parsed;
      }
    }
    return fail();
  };
  const value = (depth: number): J => {
    if (depth > 64) fail("portable_depth_limit");
    whitespace();
    const char = raw[at];
    if (char === '"') return literalString();
    if (char === "[") {
      at++; whitespace();
      const result: J[] = [];
      if (raw[at] === "]") { at++; return result; }
      while (true) {
        result.push(value(depth + 1)); whitespace();
        if (raw[at] === "]") { at++; return result; }
        if (raw[at++] !== ",") fail();
      }
    }
    if (char === "{") {
      at++; whitespace();
      const result = Object.create(null) as O;
      if (raw[at] === "}") { at++; return result; }
      while (true) {
        whitespace(); if (raw[at] !== '"') fail();
        const key = literalString(); asciiKey(key);
        if (Object.hasOwn(result, key)) fail("portable_duplicate_key");
        whitespace(); if (raw[at++] !== ":") fail();
        result[key] = value(depth + 1); whitespace();
        if (raw[at] === "}") { at++; return result; }
        if (raw[at++] !== ",") fail();
      }
    }
    for (const [token, result] of [["null", null], ["true", true], ["false", false]] as const) {
      if (raw.startsWith(token, at)) { at += token.length; return result; }
    }
    const start = at;
    while (/[A-Za-z0-9+.-]/.test(raw[at] ?? "") && at < raw.length) at++;
    const token = raw.slice(start, at);
    if (!/^(?:0|-[1-9][0-9]*|[1-9][0-9]*)$/.test(token) || token.length > 17) fail("portable_number_invalid");
    const result = Number(token);
    if (!Number.isSafeInteger(result)) fail("portable_integer_range");
    return result;
  };
  const result = value(0);
  whitespace(); if (at !== raw.length) fail("portable_json_invalid");
  canonicalBytes(result);
  return result;
}

const hash = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");
export function hashDomain(value: J): string {
  const encoded = canonicalBytes(value);
  const fields = object(value);
  if (!Object.hasOwn(fields, "domain") || typeof fields.domain !== "string" || !fields.domain) fail("portable_domain_required");
  return hash(encoded);
}
function coordinate(value: J): string {
  const text = string(value);
  const match = /^([A-Za-z][A-Za-z0-9_./-]{0,127})@([1-9][0-9]{0,15})$/.exec(text);
  must(match && !match[1].includes("..") && match[1].split("/").every(Boolean) && Number.isSafeInteger(Number(match[2])));
  return text;
}
function origin(value: J): O {
  const result = object(value); keys(result, ["coordinate", "occurrence_path"]);
  const name = string(result.coordinate);
  must(name.length > 0 && name.length <= 160 && /^[\x21-\x7e]+$/.test(name));
  const path = array(result.occurrence_path); must(path.length <= 34);
  path.forEach((item) => integer(item, 16384));
  return result;
}
function lock(value: J): O {
  const result = object(value); keys(result, ["coordinate", "content_sha256", "provenance"]);
  coordinate(result.coordinate); must(/^[a-f0-9]{64}$/.test(string(result.content_sha256)));
  if (result.provenance !== null) {
    const source = object(result.provenance);
    keys(source, ["repository", "commit", "path", "source_file_sha256", "selector", "dialect"]);
    must(/^[a-f0-9]{40}$/.test(string(source.commit)) && /^[a-f0-9]{64}$/.test(string(source.source_file_sha256)));
    for (const name of ["repository", "path", "selector", "dialect"]) must(string(source[name]).length > 0);
  }
  return result;
}

export interface ConsumedBuild { readonly canonical: string; readonly build_id: string }
export interface ConsumedRender { readonly canonical: string; readonly render_id: string }
function readBuild(raw: string): O {
  const build = object(parsePortableJSON(raw));
  must(canonicalBytes(build).toString("utf8") === raw);
  keys(build, ["domain", "schema_version", "dialect", "compiler_version", "surface", "dependency_lock", "variables", "constraints", "kind", "purpose", "descriptor", "outputs", "diagnostics"]);
  must(build.domain === "strata-prompt-build-v1" && build.schema_version === 1 && build.dialect === "strata-modules-v1" && build.compiler_version === "strata-modular-compiler/1");
  const locks = array(build.dependency_lock); must(locks.length > 0 && locks.length <= 4096);
  const coordinates = locks.map((item) => string(lock(item).coordinate));
  must(new Set(coordinates).size === coordinates.length && coordinates.join("\0") === [...coordinates].sort().join("\0"));
  const surface = canonicalBytes(lock(build.surface)).toString("utf8");
  must(locks.some((item) => canonicalBytes(item).toString("utf8") === surface));
  const variables = array(build.variables); must(variables.length <= 256);
  const names = variables.map((item) => {
    const declaration = object(item); keys(declaration, ["name", "type"]);
    must(/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(string(declaration.name)) && ["string", "integer", "boolean"].includes(string(declaration.type)));
    return string(declaration.name);
  });
  must(new Set(names).size === names.length);
  const constraints = array(build.constraints);
  const constraintNames = constraints.map((item) => {
    const fact = object(item); keys(fact, ["key", "value", "origins"]);
    string(fact.value); const sources = array(fact.origins); must(sources.length > 0);
    sources.forEach((source) => must(coordinates.includes(string(origin(source).coordinate))));
    return string(fact.key);
  });
  must(new Set(constraintNames).size === constraintNames.length);
  const outputs = array(build.outputs); must(outputs.length > 0 && outputs.length <= 128);
  const occurrences = new Set<string>(), literals = new Map<string, string>();
  let count = 0;
  outputs.forEach((item, index) => {
    const output = object(item); keys(output, ["role", "separator", "leaves"]); string(output.separator);
    must(output.role === null || ["system", "user", "assistant"].includes(string(output.role)));
    array(output.leaves).forEach((entry) => {
      must(++count <= 16384);
      const leaf = object(entry); keys(leaf, ["kind", "text", "variable", "origin"]);
      const source = origin(leaf.origin), path = array(source.occurrence_path);
      must(coordinates.includes(string(source.coordinate)) && path.length >= 2 && path[0] === index);
      const key = JSON.stringify(path); must(!occurrences.has(key)); occurrences.add(key);
      if (leaf.kind === "text") {
        must(leaf.variable === null); literals.set(JSON.stringify(source), string(leaf.text));
      } else must(leaf.kind === "variable" && leaf.text === null && names.includes(string(leaf.variable)));
    });
  });
  const diagnostics = array(build.diagnostics); must(diagnostics.length <= 64);
  diagnostics.forEach((item) => {
    const diagnostic = object(item); keys(diagnostic, ["code", "severity", "origins"]);
    must(diagnostic.code === "duplicate_literal" && diagnostic.severity === "advisory");
    const texts = array(diagnostic.origins).map((source) => literals.get(JSON.stringify(origin(source))));
    must(texts.length >= 2 && texts[0] && texts.every((text) => text === texts[0]));
  });
  if (build.kind === "chat") must(build.purpose === null && build.descriptor === null && outputs.every((item) => object(item).role !== null));
  else must(build.kind === "artifact" && outputs.length === 1 && object(outputs[0]).role === null && ["system_instruction", "tool_description", "skill", "text"].includes(string(build.purpose)) && (build.descriptor === null || object(build.descriptor)));
  return build;
}
export function consumeBuild(raw: string): ConsumedBuild {
  readBuild(raw);
  return Object.freeze({ canonical: raw, build_id: hash(raw) });
}

export function consumeRender(raw: string): ConsumedRender {
  const render = object(parsePortableJSON(raw));
  must(canonicalBytes(render).toString("utf8") === raw, "render_invalid");
  keys(render, ["domain", "schema_version", "build_id", "input_fingerprint", "output", "source_map"]);
  must(render.domain === "strata-prompt-render-v1" && render.schema_version === 1 && /^[a-f0-9]{64}$/.test(string(render.build_id)) && /^[a-f0-9]{64}$/.test(string(render.input_fingerprint)), "render_invalid");
  const output = object(render.output), contents: string[] = [];
  if (output.kind === "chat") {
    keys(output, ["kind", "messages"]); const messages = array(output.messages);
    must(messages.length > 0 && messages.length <= 128);
    messages.forEach((item) => { const message = object(item); keys(message, ["role", "content"]); must(["system", "user", "assistant"].includes(string(message.role))); contents.push(string(message.content)); });
  } else {
    keys(output, ["kind", "purpose", "content", "descriptor"]);
    must(output.kind === "artifact" && ["system_instruction", "tool_description", "skill", "text"].includes(string(output.purpose)) && (output.descriptor === null || object(output.descriptor)));
    contents.push(string(output.content));
  }
  const bytes = contents.map((content) => Buffer.from(content));
  must(bytes.reduce((sum, item) => sum + item.length, 0) <= RENDER_BYTES);
  const offsets = contents.map(() => 0), leaves = contents.map(() => 0), expectLeaf = contents.map(() => true);
  const occurrences = new Set<string>(); let lastIndex = 0;
  const spans = array(render.source_map); must(spans.length <= 32768);
  spans.forEach((item) => {
    const span = object(item); keys(span, ["output_index", "start", "end", "kind", "origin", "variable"]);
    const index = integer(span.output_index, 127), start = integer(span.start, RENDER_BYTES), end = integer(span.end, RENDER_BYTES);
    must(index >= lastIndex && index < bytes.length && start === offsets[index] && end >= start && end <= bytes[index].length, "render_invalid");
    try { decoder.decode(bytes[index].subarray(start, end)); } catch { fail("render_invalid"); }
    const source = origin(span.origin), path = array(source.occurrence_path);
    if (span.kind === "separator") {
      must(!expectLeaf[index] && span.variable === null && source.coordinate === "$compiler" && JSON.stringify(path) === JSON.stringify([index, leaves[index]]), "render_invalid");
      expectLeaf[index] = true;
    } else {
      must(expectLeaf[index] && path.length >= 2 && path[0] === index, "render_invalid"); coordinate(source.coordinate);
      must((span.kind === "text" && span.variable === null) || (span.kind === "variable" && /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(string(span.variable))), "render_invalid");
      const key = JSON.stringify(source); must(!occurrences.has(key)); occurrences.add(key);
      leaves[index]++; expectLeaf[index] = false;
    }
    offsets[index] = end; lastIndex = index;
  });
  must(offsets.every((offset, index) => offset === bytes[index].length) && leaves.every((count, index) => !count || !expectLeaf[index]), "render_invalid");
  return Object.freeze({ canonical: raw, render_id: hash(raw) });
}

export function renderBuild(snapshot: ConsumedBuild, values: unknown): ConsumedRender {
  const build = readBuild(snapshot.canonical); must(hash(snapshot.canonical) === snapshot.build_id);
  const inputs = object(parsePortableJSON(canonicalBytes(values)));
  const declarations = array(build.variables).map(object);
  must(JSON.stringify(Object.keys(inputs).sort()) === JSON.stringify(declarations.map((item) => string(item.name)).sort()), "input_keys_mismatch");
  const spellings = new Map<string, string>();
  declarations.forEach((item) => {
    const name = string(item.name), value = inputs[name];
    if (item.type === "string" && typeof value === "string") spellings.set(name, value);
    else if (item.type === "boolean" && typeof value === "boolean") spellings.set(name, value ? "true" : "false");
    else if (item.type === "integer" && typeof value === "number" && Number.isSafeInteger(value) && !Object.is(value, -0)) spellings.set(name, String(value));
    else fail("input_type_mismatch");
  });
  const sourceMap: J[] = [], texts: string[] = []; let total = 0;
  const outputs = array(build.outputs).map(object);
  outputs.forEach((output, index) => {
    let offset = 0; const chunks: string[] = [];
    array(output.leaves).forEach((item, leafIndex) => {
      const leaf = object(item);
      const emissions: [string, string, J, J][] = [];
      if (leafIndex) emissions.push([string(output.separator), "separator", { coordinate: "$compiler", occurrence_path: [index, leafIndex] }, null]);
      const text = leaf.kind === "text" ? string(leaf.text) : spellings.get(string(leaf.variable))!;
      emissions.push([text, string(leaf.kind), leaf.origin, leaf.variable]);
      emissions.forEach(([text, kind, origin, variable]) => {
        const length = Buffer.byteLength(text); total += length;
        must(total <= RENDER_BYTES, "rendered_bytes_limit");
        sourceMap.push({ output_index: index, start: offset, end: offset + length, kind, origin, variable });
        offset += length; chunks.push(text);
      });
    });
    texts.push(chunks.join(""));
  });
  const output: J = build.kind === "chat" ? {
    kind: "chat", messages: outputs.map((item, index) => ({ role: item.role, content: texts[index] })),
  } : { kind: "artifact", purpose: build.purpose, descriptor: build.descriptor, content: texts[0] };
  const payload: J = {
    domain: "strata-prompt-render-v1", schema_version: 1, build_id: snapshot.build_id,
    input_fingerprint: hashDomain({ domain: "strata-prompt-input-v1", values: inputs }),
    output, source_map: sourceMap,
  };
  return consumeRender(canonicalBytes(payload).toString("utf8"));
}

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
