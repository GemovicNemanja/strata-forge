"""Runnable pipeline entrypoints executed on a compute target.

These modules are launched on a compute target by an orchestrator rather than imported
in-process. Each reads an inert run spec from the environment, composes the library's
primitives (serving / batch / storage), and appends ProgressEvents to the file named by
``FORGE_PROGRESS_PATH`` so the orchestrator can tail progress over its own channel.
"""
