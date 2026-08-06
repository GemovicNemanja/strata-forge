"""Runnable pipeline entrypoints executed on a compute target.

These modules are launched on the user's VM by the control plane (strata-server),
not imported by the orchestrator. Each reads an inert run spec from the environment,
composes the forge primitives (serving / batch / storage), and appends ProgressEvents
to ``FORGE_PROGRESS_PATH`` for the orchestrator to tail.
"""
