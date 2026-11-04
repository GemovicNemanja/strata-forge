# Installed evaluation policy adapter handoff

The worker accepts `ReadPolicy` from trusted installed Python code. The inert EvalSpec and
CLI never choose an adapter, import path or boolean grant. The default standalone command
remains closed with `custody_policy_unavailable`.

The coordinator owns the concrete server-issued approval verification and worker bootstrap.
An adapter must verify issuer/authentication, run/actor/workspace binding, exact immutable
refs, operation/scope, expiration/replay and the approved policy version, then combine fresh
portable markers/registry facts/lineage/canonical and historical aliases monotonically. Unknown
metadata or any positive protected fact must deny. A server-issued approval alone cannot
erase a later positive custody fact. Recheck authorization at execution, not only launch.

The minimal shape below is an adapter seam, **not** an implementation of ticket verification.
Neither protocol can be implemented with a caller's `sealed=false` or Hub readability.

```python
from typing import Literal, Protocol
from strata_forge.pipelines.eval_runner import ReadDecision

class VerifiedApprovalService(Protocol):
    async def require_read(self, *, bound_run_id: str, ref: str, repo_type: Literal["model", "dataset"], spec_sha256: str) -> None:
        """Verify authenticated issuer, bound actor/run/ref/scope/expiry; raise on failure."""
        ...

class FreshCustodyService(Protocol):
    async def resolve(self, ref: str, *, repo_type: Literal["model", "dataset"]) -> Literal["unsealed", "sealed", "locked", "unknown"]:
        """Resolve fresh monotone facts, including aliases/markers/lineage; no negative guessing."""
        ...

class InstalledEvaluationPolicy:
    def __init__(self, run_id: str, approval: VerifiedApprovalService, custody: FreshCustodyService):
        self.run_id, self.approval, self.custody = run_id, approval, custody

    async def decide(self, ref: str, *, repo_type: Literal["model", "dataset"], spec_sha256: str) -> ReadDecision:
        await self.approval.require_read(bound_run_id=self.run_id, ref=ref, repo_type=repo_type, spec_sha256=spec_sha256)
        state = await self.custody.resolve(ref, repo_type=repo_type)
        return ReadDecision(
            ref=ref, repo_type=repo_type, spec_sha256=spec_sha256, authorized=True, custody=state, policy_version="installed-adapter-v1"
        )
```

`authorized=True` above is produced only after the trusted verification service succeeds;
it is never deserialized from job params or treated as a user grant. The worker revalidates
the DTO, requires exact ref identity and `custody=unsealed`, and authorizes all refs before
reader preparation/model load/row reads. The real adapter also needs bounded fixed errors,
network/auth timeouts, freshness/replay tests and authenticated metadata transport. Registering
the server kind stays disabled until that concrete bridge is independently accepted.

Fresh Claude requests remain separate scopes: training defaults to final checkpoint, explicit
selection_split must exclude marked gate/test tiers, GPU budget is 25h hard/20h hold (research
owner/operator policy, not a universal worker default), and Ettin head mapping/step-zero
equivalence belongs to trusted TRAIN initialization. Evaluation never relaxes missing-head
loads or downloads the real checkpoint to simulate those acceptance cases.

Use the actual `EvalSpec.digest()` for the full stamped worker spec, including config/split/selection; the policy decision echoes that digest and repository domain, and the worker requires both to match exactly.
