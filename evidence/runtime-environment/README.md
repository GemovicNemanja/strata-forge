# Runtime distribution observation qualification

Runtime source is `199eab3fcb22d7ebc54989dd1ece238daa5ea21b`, based on selected
Forge `6efaeb1`. The only runtime addition is core/runtime_environment.py; existing
runtime files, public initializers, legacy snapshots, training catalogue, workers,
specs, dependencies, lock and version remain unchanged.

The internal producer observes the actual interpreter and nineteen fixed package
distribution metadata entries. It never equates installed metadata with successful
import, hardware, source/image/lock identity or execution eligibility. Absent,
invalid and failed metadata remain distinct. It reads no source URL, environment
secret or working directory. Trusted metadata finders are synchronous and are not
claimed to have a hard filesystem/time budget. There is no runner/API/MCP consumer.

The final exact-source cohort passes 230 tests in 3.01 seconds: 27 owned observation
controls plus existing reproducibility and training-capability controls. It covers
actual interpreter observation; fixed profile completeness; absent/invalid/failed
metadata; preserved control exceptions; canonical time/digest; immutable nested
values; copied/constructed/subclass refusal; exact portable synthetic golden/schema;
and a fresh flag-free isolated process. The embedded cold checker observed zero
socket, nested-process and optional-engine/provider import attempts while actually
capturing and encoding metadata. No separate hardware or installed-engine proof is
inferred. The earlier owned 26-test and overlapping 229-test cohorts are not added
to this final count.

Strict Pyright on the added runtime file reports zero errors and warnings using
the explicit retained Python interpreter. Its retained config notes that this
isolate has no local .venv; dependencies resolve through the explicit interpreter.
Initial three constructor literal-typing diagnostics and Ruff diagnostics are
preserved; explicit strict interpreter validation and final static checks pass.
No product assertion failed. Fixed boundary errors are raised outside raw handlers;
metadata exception details are not emitted or logged.

The schema/golden files are synthetic and structure-only: cross-field consistency,
complete profile ordering and exact numeric kinds still require the runtime
semantic validators. Output is capped at 16 KiB. No network installation, process
execution by the producer, provider/model/GPU action, publication, retained user
profile or research change occurred. Test subprocesses are explicitly owned cold
witnesses, not producer-side process creation. This leaves the broad environment,
attempt/admission and worker-profile tasks open.
