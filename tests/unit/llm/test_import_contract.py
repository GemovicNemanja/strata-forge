"""Fresh-process lazy imports preserve public identities without implicit transport work."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SURFACE = Path(__file__).parent / "fixtures/public_exports.json"

_GUARD = r"""
import importlib
import importlib.abc
import json
import sys
import time
attempts = []
imports = []
optional = {'torch','transformers','trl','peft','datasets','bitsandbytes','accelerate',
            'numpy','pyarrow','asyncssh','skypilot'}
transport = {'litellm','httpx','openai','anthropic','boto3','botocore','google'}
class Denied(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in optional | (transport if LIGHTWEIGHT else set()):
            imports.append(fullname)
            raise AssertionError('denied import: ' + fullname)
def audit(event, args):
    if event.startswith(('socket.','subprocess.','os.system','os.exec','os.posix_spawn')):
        attempts.append(event)
        raise AssertionError('denied effect: ' + event)
sys.meta_path.insert(0, Denied())
sys.addaudithook(audit)
start = time.perf_counter()
"""


def _fresh(script: str, *, lightweight: bool = True) -> dict[str, object]:
    env = {
        "HOME": str(Path.home()),
        "PATH": os.defpath,
        "PYTHONPATH": str(_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "OTEL_SDK_DISABLED": "true",
    }
    if not lightweight:
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and source-controlled scripts
        [
            sys.executable,
            "-B",
            "-c",
            "LIGHTWEIGHT = "
            + repr(lightweight)
            + "\n"
            + _GUARD
            + script
            + "\nassert imports == [], imports\nassert attempts == [], attempts\n"
            + "assert not (optional & set(sys.modules))\n"
            + "if LIGHTWEIGHT: assert not (transport & set(sys.modules))\n"
            + "print(json.dumps({'attempts': attempts, 'imports': imports, 'seconds': time.perf_counter()-start}))\n",
        ],
        env=env,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    value: dict[str, object] = json.loads(result.stdout)
    assert value["attempts"] == value["imports"] == []
    return value


@pytest.mark.parametrize(
    "target",
    [
        "strata_forge.llm",
        "strata_forge.llm.messages",
        "strata_forge.training.methods",
        "strata_forge.training.capabilities",
    ],
)
def test_lightweight_imports_need_no_transport_or_offline_flag(target: str) -> None:
    _fresh(
        "import os\n"
        "assert 'LITELLM_LOCAL_MODEL_COST_MAP' not in os.environ\n"
        "assert 'LITELLM_MODE' not in os.environ\n"
        f"module = importlib.import_module({target!r})\n"
        "assert 'strata_forge.llm.providers' not in sys.modules\n"
        "if hasattr(module, 'training_capabilities'): module.training_capabilities()\n"
    )


def test_dir_is_nonloading_and_first_attribute_access_caches_original_identity() -> None:
    _fresh(r"""
import strata_forge.llm as llm
assert 'SystemMessage' not in vars(llm)
assert 'LLMClient' not in vars(llm)
assert 'strata_forge.llm.messages' not in sys.modules
assert 'strata_forge.llm.client' not in sys.modules
assert set(llm.__all__) <= set(dir(llm))
assert 'strata_forge.llm.messages' not in sys.modules
from strata_forge.llm import SystemMessage
module = importlib.import_module('strata_forge.llm.messages')
assert SystemMessage is module.SystemMessage is llm.SystemMessage
assert vars(llm)['SystemMessage'] is SystemMessage
try:
    llm.nonexistent_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export accepted')
assert 'strata_forge.llm.client' not in sys.modules
""")


@pytest.mark.parametrize("first", ["package", "submodule", "from"])
def test_registry_singleton_survives_import_order(first: str) -> None:
    prefix = {
        "package": "import strata_forge.llm as llm\n",
        "submodule": "module = importlib.import_module('strata_forge.llm.registry')\n",
        "from": "from strata_forge.llm import registry as selected\n",
    }[first]
    _fresh(
        prefix
        + r"""
import strata_forge.llm as llm
module = importlib.import_module('strata_forge.llm.registry')
from strata_forge.llm import registry as selected
import strata_forge.llm.registry as dotted
assert selected is llm.registry is module.registry is dotted
assert type(module).__name__ == 'module'
assert isinstance(selected, module.Registry)
from strata_forge.llm import Registry, Model, Pricing
assert Registry is module.Registry
assert Model is module.Model and Pricing is module.Pricing
assert 'strata_forge.llm.providers' not in sys.modules
"""
    )


def test_full_namespace_star_import_and_submodule_identities_remain_unchanged() -> None:
    _fresh(
        f"from pathlib import Path\nsurface = json.loads(Path({str(_SURFACE)!r}).read_text())\n"
        + r"""
import strata_forge.llm as llm
assert llm.__all__ == surface['all']
assert len(llm.__all__) == 99
for name in llm.__all__:
    binding = surface['bindings'][name]
    original = getattr(importlib.import_module(binding['module']), binding['attribute'])
    assert getattr(llm, name) is original, name
    assert vars(llm)[name] is original, name
namespace = {}
exec('from strata_forge.llm import *', namespace)
assert set(namespace) - {'__builtins__'} == set(llm.__all__)
for name in llm.__all__:
    assert namespace[name] is getattr(llm, name), name
assert llm.registry is importlib.import_module('strata_forge.llm.registry').registry
""",
        lightweight=False,
    )


def test_static_typing_imports_and_runtime_registry_match_predecessor_surface() -> None:
    expected = json.loads(_SURFACE.read_text())
    tree = ast.parse((_ROOT / "src/strata_forge/llm/__init__.py").read_text())
    bindings = {
        alias.asname or alias.name: {"module": node.module, "attribute": alias.name}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("strata_forge.llm.")
        for alias in node.names
    }
    assert bindings == expected["bindings"]
