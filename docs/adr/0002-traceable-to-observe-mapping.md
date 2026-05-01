# ADR-0002: `@traceable` → `@observe` mapping policy

- **Status:** Accepted
- **Date:** 2026-04-30
- **Bead:** `travel-multi-agent-workshop-7bl.2`
- **Epic:** `travel-multi-agent-workshop-7bl` — Replace LangSmith with self-hosted Langfuse
- **Depends on:** [ADR-0001](./0001-langfuse-sdk-version.md) (pinned to `langfuse>=4.0.0,<5`)

## Context

The codebase uses **95 `@traceable` decorator sites across 6 files**:

| File | Sites |
|---|---|
| `02_completed/mcp_server/mcp_http_server.py` | 24 (all bare) |
| `02_completed/python/src/app/travel_agents.py` | 7 (6 `llm` + 1 bare) |
| `02_completed/python/src/app/services/azure_cosmos_db.py` | 22 (mix) |
| `01_exercises/mcp_server/mcp_http_server.py` | 19 (all bare) |
| `01_exercises/python/src/app/travel_agents.py` | 6 (all `llm`) |
| `01_exercises/python/src/app/services/azure_cosmos_db.py` | 22 (mix) |
| **Total** | **100** |

A grep for `@traceable` across both subtrees confirms every site uses
**only one of three syntactic forms** — no `name=`, `metadata=`, `tags=`,
`client=`, `project_name=`, or other LangSmith-only kwargs appear anywhere:

1. `@traceable` (bare, no parens, no kwargs) — 64 sites
2. `@traceable(run_type="llm")` — 13 sites
3. `@traceable(run_type="retriever")` — 23 sites

This makes the migration mechanical and amenable to find-and-replace.

### Verification of v4 `@observe()` capabilities

Inspected `langfuse/_client/observe.py` on the v4 line of `langfuse-python`.

The `@observe` signature is:

```python
@observe(
    name: Optional[str] = None,
    as_type: Optional[ObservationTypeLiteralNoEvent] = None,
    capture_input: Optional[bool] = None,
    capture_output: Optional[bool] = None,
    transform_to_string: Optional[Callable[[Iterable], str]] = None,
)
```

Accepted `as_type` values (from the source docstring and runtime validation):

> `"generation"`, `"span"`, `"agent"`, `"tool"`, `"chain"`,
> **`"retriever"`**, `"embedding"`, `"evaluator"`, `"guardrail"`

This **changes our assumption** from the original bead description, which
said "Langfuse v3 has no first-class retriever span type." On v4 (and per
this ADR's prerequisite ADR-0001), `as_type="retriever"` IS first-class
and gets dedicated UI treatment in Langfuse.

Runtime safety: invalid `as_type` values do not raise; they log a warning
and silently default to `"span"`.

## Decision

### Mapping table (apply mechanically — no per-site judgement)

| LangSmith | Langfuse |
|---|---|
| `@traceable`                          | `@observe()` |
| `@traceable(run_type="llm")`          | `@observe(as_type="generation")` |
| `@traceable(run_type="retriever")`    | `@observe(as_type="retriever")` |

### Import header rewrite (per file)

For every file containing `@traceable`:

- **Remove** every line matching `^\s*from langsmith import traceable\s*$`
  (note: `01_exercises/python/src/app/services/azure_cosmos_db.py` has this
  import on **both** line 11 and line 15 — collapse to a single Langfuse
  import).
- **Add** exactly one new import near the other third-party imports:

  ```python
  from langfuse import observe
  ```

### Logger silencing — remove

Both `mcp_http_server.py` files contain a line that mutes the LangSmith
client logger; this becomes a dead reference once `langsmith` is uninstalled:

```python
logging.getLogger("langsmith.client").setLevel(logging.WARNING)  # delete this
```

There is no equivalent silencing required for Langfuse — its default log
level is already `WARNING`. (If in practice we discover Langfuse v4 is too
chatty at INFO during development, follow up with a separate ticket.)

### What this ADR explicitly does NOT cover

- **`user_id` / `session_id` propagation.** v4 uses
  `propagate_attributes(...)` (a context manager) for this. Out of scope
  here; handled in the LangGraph callback story (`7bl.7` →
  `task-cb-02`/`task-cb-01`).
- **Trace input/output enrichment** beyond what `@observe` captures
  automatically from function args/return. Out of scope.
- **`@observe(name=...)` overrides.** v4 uses the function name by default,
  matching our LangSmith status quo. We do not introduce custom names in
  this ADR — if a future enhancement needs them, do it as a follow-up.
- **Metadata enrichment.** Same — out of scope; follow-up if needed.

## Alternatives considered

### A. Shim — `from langfuse import observe as traceable`

Keep the old call sites verbatim and only swap the import line.

- **Pros:** smallest diff, fastest mechanical change, easy to revert.
- **Cons:**
  - Hides the migration from future readers; greppability for "what
    tracer are we on" suffers.
  - Asymmetric kwargs: `traceable(run_type="llm")` would silently NOT
    become a generation span, because `observe()` ignores `run_type` and
    accepts `as_type` instead. We'd need a wrapper to translate, which is
    more code than the rewrite itself.
  - Workshop attendees would see a misleading API surface; the workshop
    is *the* primary deliverable so pedagogy matters.
- **Verdict:** rejected.

### B. Map `run_type="retriever"` to `@observe()` with metadata tag

(The bead's original default, written before we'd verified v4's capabilities.)

```python
@observe()
def search(...):
    langfuse.update_current_observation(metadata={"langfuse_legacy_type": "retriever"})
    ...
```

- **Pros:** Works on any version; no reliance on v4-specific span types.
- **Cons:**
  - Adds an in-body call at every retriever site (~23 sites) — not purely
    mechanical anymore.
  - Loses the Langfuse UI's first-class retriever filtering.
  - Requires a `get_client()` reference in scope, which not every file has.
- **Verdict:** rejected — superseded by the v4 source-code verification.

### C. Map `run_type="retriever"` to `@observe(as_type="span")` (default)

Just drop the retriever distinction entirely.

- **Pros:** Simplest possible mapping.
- **Cons:** Loses semantic information that the original code went out of
  its way to record. Cosmos DB query spans become indistinguishable from
  generic ones in the UI.
- **Verdict:** rejected — preserves less info than the LangSmith original,
  for no benefit.

### D. Map `run_type="llm"` to `@observe(as_type="span")` then enrich

- **Pros:** None we can identify.
- **Cons:** Loses generation-span metrics (token counts, model, latency
  per model) that Langfuse provides on `as_type="generation"`.
- **Verdict:** rejected.

## Consequences

### Positive

- 1:1 semantic preservation of the LangSmith-era intent.
- Pure mechanical rewrite — each per-file rewrite task is essentially a
  three-rule sed/regex change, low risk.
- Retriever spans get first-class visualization in the Langfuse UI.
- LLM spans automatically capture model/usage metrics on the
  `generation` span type.

### Negative / risks

- 100 sites is a lot of edits. Mitigation: rewrite tasks are scoped to
  ONE file each (see `task-rw-*` beads), so blast radius per task is
  small and each can be validated independently with `pytest -k <file>`
  or a manual import smoke check.
- If a future maintainer adds a fourth `@traceable(run_type=...)` value
  before the migration completes, this ADR's table won't cover it.
  Mitigation: the validation gate `task-val-grep` greps for any residual
  `langsmith` reference and any unhandled `run_type=`.

### Rewrite recipe (for the per-file tasks)

The per-file `task-rw-*` tasks should apply this transform exactly:

1. Open the file.
2. Replace import lines per the "Import header rewrite" section above.
3. Apply, in order:
   ```
   @traceable(run_type="retriever")  →  @observe(as_type="retriever")
   @traceable(run_type="llm")        →  @observe(as_type="generation")
   @traceable                         →  @observe()
   ```
   Order matters — do the parameterized forms before the bare form so the
   bare-form regex doesn't eat the parameterized prefixes.
4. Delete any `logging.getLogger("langsmith.client").setLevel(...)` line.
5. Run `python -c "import <module>"` to confirm imports resolve.
6. Grep the file: zero matches for `langsmith` and zero matches for
   `traceable` (case-sensitive).

### Exception list

**None.** The full inventory was 100 sites in 3 syntactic forms. No site
uses any kwarg outside `run_type`. No site requires per-call special
handling.

If the rewrite tasks discover a site this ADR did not anticipate
(e.g., a pattern introduced after this date), the rewrite task should
stop, append the case to this ADR's exception list, and resume.

## References

- ADR-0001 (SDK version pin): [./0001-langfuse-sdk-version.md](./0001-langfuse-sdk-version.md)
- v4 `@observe` source: <https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/observe.py>
- Langfuse observation types: <https://langfuse.com/docs/observability/data-model>
