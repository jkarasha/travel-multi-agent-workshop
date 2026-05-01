# ADR-0001: Langfuse Python SDK major version pin

- **Status:** Accepted
- **Date:** 2026-04-30
- **Bead:** `travel-multi-agent-workshop-7bl.1`
- **Epic:** `travel-multi-agent-workshop-7bl` — Replace LangSmith with self-hosted Langfuse

## Context

We are replacing LangSmith with self-hosted Langfuse for LLM observability,
tracing, and evaluation. Before any code is rewritten we need to lock down
which major version of the `langfuse` Python SDK we will pin in our
dependency files. The decision is foundational because it affects:

- Every `requirements.txt` / `pyproject.toml` in the workshop.
- Every `@traceable` → Langfuse decorator rewrite (~95 sites across 6 files).
- The LangChain/LangGraph callback handler import path.
- The semantics of the evaluation port (Datasets / Evaluations API).

The current LangSmith pin (`langsmith==0.4.3`) appears in:

- `02_completed/requirements.txt`
- `02_completed/python/src/app/requirements.txt`
- `01_exercises/requirements.txt`
- `01_exercises/pyproject.toml`

These four files are the migration surface for the dependency change.

### State of the SDK at decision time

PyPI (`https://pypi.org/pypi/langfuse/json`) reports:

- Latest version: **`4.5.1`**
- v4 was released in **March 2026** as a major refactor introducing the
  observation-centric data model.
- v3 (released 2025) is the previous major; v3 was itself a refactor onto
  OpenTelemetry semantics.
- `requires_python = ">=3.10,<4"` — compatible with our 3.11+ floor.
- Runtime deps are reasonable: `httpx`, `pydantic>=2`, `wrapt`, `backoff`,
  `opentelemetry-{api,sdk,exporter-otlp-proto-http}`.

The official v3→v4 upgrade guide
(`https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4`)
documents the breaking changes:

1. **Smart default span filtering.** v4 only forwards spans that are
   Langfuse-created, have `gen_ai.*` attributes, or come from known LLM
   instrumentation scopes (`openinference`, `langsmith`, `haystack`,
   `litellm`, …). Pre-v4 forwarded everything by default.
2. **`update_current_trace()` is decomposed** into
   `propagate_attributes()` (a context manager), `set_current_trace_io()`,
   and `set_current_trace_as_public()`. `release` and `environment` move
   to env vars (`LANGFUSE_RELEASE`, `LANGFUSE_TRACING_ENVIRONMENT`).
3. Public API namespace remap (`api.observations_v_2` → `api.observations`).

The `@observe()` decorator API itself is **unchanged** between v3 and v4.

### Verification of LangChain handler

Searched `langfuse/langfuse-python` on GitHub:

- `langfuse/langchain/CallbackHandler.py` is present on the v4 line.
- `langfuse/langchain/__init__.py` re-exports it.
- Therefore `from langfuse.langchain import CallbackHandler` resolves on v4.

## Decision

**Pin `langfuse>=4.0.0,<5` in all four dependency files.**

In the four files listed above, replace the line `langsmith==0.4.3` with:

```
langfuse>=4.0.0,<5
```

(In `01_exercises/pyproject.toml` this is the corresponding TOML entry
inside the `dependencies = [...]` array.)

## Alternatives considered

### A. `langfuse>=3.0.0,<4` (the bead's original default)

- **Pros:** more battle-tested; identical user-facing decorator API; matches
  the Langfuse blog posts and Stack Overflow answers from 2025.
- **Cons:** v4 has been GA for ~2 months at decision time and is the version
  that new docs target. We have no production v3 deployment to migrate from
  and no existing code that uses `update_current_trace()`, so the v4
  breaking changes do not cost us anything. Pinning to `<4` would also
  freeze us out of v4-only features (smart span filtering, observation-centric
  attribute propagation) that we *would* benefit from on day one.
- **Verdict:** rejected — v3 offers no advantage to a greenfield migration.

### B. Unbounded (`langfuse`)

- **Pros:** always latest.
- **Cons:** workshop reproducibility — a future v5 release could silently
  break participants mid-class.
- **Verdict:** rejected — workshops require pinned majors.

### C. Exact pin (`langfuse==4.5.1`)

- **Pros:** maximum reproducibility.
- **Cons:** misses patch fixes within the v4 line; conflicts with our
  permissive style elsewhere in the requirements files (most pkgs use `>=`).
- **Verdict:** rejected — caret-range is the right balance.

### D. Stay on LangSmith

- Out of scope; superseded by the parent epic decision to self-host.

## Consequences

### Positive

- New code uses the current, supported major.
- Smart span filter keeps traces focused on LLM activity automatically.
- `propagate_attributes()` model is a better fit for the multi-agent
  graph (router → triage → specialists), where `user_id` / `session_id`
  must reach every child observation.
- `LANGFUSE_TRACING_ENVIRONMENT` env var lets us cleanly separate
  `dev` / `workshop` / `eval` environments without code changes.

### Negative / risks

- v4 is younger; less community Q&A available than v3 if we hit edge cases.
  *Mitigation:* the fallback below.
- Because v4 default-filters non-LLM spans, if we later add OpenTelemetry
  instrumentation for HTTP/DB and *want* it forwarded to Langfuse, we will
  need to opt in via `should_export_span=...`. Acceptable for now;
  documented as a follow-up if/when we add infra tracing.

### Follow-ups unlocked

- `task-req` (`travel-multi-agent-workshop-z24`) — apply the pin to the
  three `requirements.txt` files.
- `task-pyproject` (`travel-multi-agent-workshop-zz4`) — apply the pin in
  `01_exercises/pyproject.toml`.
- All decorator-rewrite tasks (`task-rw-*`) consume `from langfuse import observe`
  and `from langfuse.langchain import CallbackHandler` per this version.

## Fallback clause

If during implementation v4 produces blocking friction (e.g.,
`@observe(as_type="generation")` signature change discovered late,
LangChain handler regression, OTel exporter compat issue with our Python
3.11 runtime), we may downgrade to `langfuse>=3.0.0,<4`. Any such fallback
must:

1. Update this ADR's Status to `Superseded` with a link to the follow-up ADR.
2. Re-run the full validation gate (`task-val-grep`, `task-val-smoke`,
   `task-val-eval`).

## References

- PyPI: <https://pypi.org/project/langfuse/>
- v3 → v4 upgrade guide: <https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4>
- LangChain integration: <https://langfuse.com/integrations/frameworks/langchain>
- Source of CallbackHandler: <https://github.com/langfuse/langfuse-python/blob/main/langfuse/langchain/CallbackHandler.py>
