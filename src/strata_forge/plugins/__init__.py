"""Trusted plugins: named, pinned, operator-allow-listed code that runs on the user's own target.

A plugin is a Python distribution built from one commit of a git or Hugging Face repository. It
declares itself in a ``strata-plugin.json`` manifest (:mod:`~strata_forge.plugins.manifest`):
the entry points it registers under the ``strata_forge.plugins`` group, what each reads and
writes, and the closed configuration each accepts. Three rules make running it safe to offer:

- **Allow-listed and pinned** (:mod:`~strata_forge.plugins.pins`). An operator lists the exact
  builds that may run; a request names an allowed build and one of its entry points, never a
  repository, a commit or an import path of its own.
- **Checked without being run** (:mod:`~strata_forge.plugins.preflight`). A control plane reads
  the manifest at the pinned commit as data and checks a request against it; nothing from the
  repository executes there.
- **Installed from the pin, without credentials** (:mod:`~strata_forge.plugins.environment`). On
  the target, the runner builds the pinned build its own environment and verifies it.
- **Run apart from the credentials** (:mod:`~strata_forge.plugins.host`). The runner downloads
  the declared inputs itself and runs the plugin in a child process that never receives the
  run's Hub credential.

A ``step_scorer`` entry point is a process reward model with a forge-owned contract
(:mod:`~strata_forge.plugins.step_scoring`).
"""

from strata_forge.plugins.manifest import (
    ENTRY_POINT_GROUP,
    MANIFEST_SCHEMA,
    PLUGIN_MANIFEST_FILENAME,
    ConfigParam,
    EntryPointDecl,
    InputDecl,
    OutputDecl,
    PluginManifest,
    PluginManifestError,
    ScoreKind,
    parse_manifest,
)
from strata_forge.plugins.pins import (
    ALLOWED_PLUGINS_ENV,
    AllowedPlugin,
    PluginPin,
    PluginSource,
    installed_mismatch,
    parse_allow_list,
)
from strata_forge.plugins.preflight import (
    InputBinding,
    PluginRequest,
    PreflightCheck,
    PreflightReport,
    preflight,
)
from strata_forge.plugins.step_scoring import ContextSegment, Step, StepScorerSetup, Unscored

__all__ = [
    "ALLOWED_PLUGINS_ENV",
    "ENTRY_POINT_GROUP",
    "MANIFEST_SCHEMA",
    "PLUGIN_MANIFEST_FILENAME",
    "AllowedPlugin",
    "ConfigParam",
    "ContextSegment",
    "EntryPointDecl",
    "InputBinding",
    "InputDecl",
    "OutputDecl",
    "PluginManifest",
    "PluginManifestError",
    "PluginPin",
    "PluginRequest",
    "PluginSource",
    "PreflightCheck",
    "PreflightReport",
    "ScoreKind",
    "Step",
    "StepScorerSetup",
    "Unscored",
    "installed_mismatch",
    "parse_allow_list",
    "parse_manifest",
    "preflight",
]
