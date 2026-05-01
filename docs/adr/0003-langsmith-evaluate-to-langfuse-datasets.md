# ADR-0003: LangSmith `aevaluate()` → Langfuse Datasets/Evaluations mapping

- **Status:** Accepted
- **Date:** 2026-04-30
- **Bead:** `travel-multi-agent-workshop-7bl.3`
- **Epic:** `travel-multi-agent-workshop-7bl` — Replace LangSmith with self-hosted Langfuse
- **Depends on:** [ADR-0001](./0001-langfuse-sdk-version.md) (`langfuse>=4.0.0,<5`)

## Context

Three evaluation scripts under `01_exercises/evaluation/` use the LangSmith
async eval API:

- `e2e_evaluation.py` (174 lines)
- `routing_evaluation.py` (183 lines)
- `tool_usage_evaluation.py` (172 lines)

All three follow the same shape:

```python
from langsmith import Client

client = Client(api_key=..., api_url="https://api.smith.langchain.com")

# 1. Bootstrap dataset
if client.has_dataset(dataset_name=NAME):
    client.delete_dataset(dataset_id=client.read_dataset(NAME).id)
dataset = client.create_dataset(dataset_name=NAME, description=...)
client.create_examples(dataset_id=dataset.id, examples=[...])

# 2. Run + score in one call
results = await client.aevaluate(
    target_async_fn,             # async (inputs: dict) -> dict
    data=NAME,
    evaluators=[answer_quality, correctness, humanness],
    experiment_prefix="travel-e2e",
    num_repetitions=1,
    max_concurrency=4,
    metadata={...},
)
```

### Concrete signatures used

**Target functions** (one per script, all `async`):

```python
async def run_travel_agent_e2e(inputs: dict) -> dict:
    # inputs == {"question": "..."}
    return {"answer": str, "message_count": int, "thread_id": str}
```

**Evaluators** — two flavors, both expect `outputs` (the target's return value)
and `reference_outputs` (the dataset row's ground-truth `outputs`):

```python
# Heuristic (sync) — heuristic_evaluators.py
def correct_routing(outputs: dict, reference_outputs: dict) -> bool: ...
def required_tools_called(outputs: dict, reference_outputs: dict) -> bool: ...
def tool_call_accuracy(outputs: dict, reference_outputs: dict) -> float: ...

# LLM-as-judge (async) — llm_judges.py
async def answer_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> bool: ...
async def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool: ...
async def humanness(inputs: dict, outputs: dict, reference_outputs: dict) -> int: ...
```

Returns are **bare scalars** (bool / float / int). LangSmith infers the
score name from the function name and converts bools to 0/1.

**Dataset rows** are JSON files at `01_exercises/evaluation/datasets/*.json`:

```json
{
  "inputs":  { "question": "Find me hotels in Barcelona" },
  "outputs": { "answer": "...", "expected_tools": [...], "expected_agent": "hotel", "should_extract_preferences": false }
}
```

The `outputs` key in the JSON is the **ground truth** (what becomes
`reference_outputs` at evaluation time).

### What LangSmith's `aevaluate()` actually did for us

1. Created an "experiment" record grouping all runs together.
2. Iterated dataset rows.
3. Spun up `max_concurrency=4` async workers, each calling
   `target_async_fn(row.inputs)` and recording a Run.
4. Called every evaluator with `(inputs, outputs, reference_outputs)`,
   batched scores, and persisted them under the experiment.
5. Returned an `experiment_results` object with aggregation helpers.

**Tracing was explicitly disabled during these runs** via
`os.environ["LANGCHAIN_TRACING_V2"] = "false"` — the original author
chose speed over observability. We will reverse that for Langfuse
(see Decision §5).

### What Langfuse v4 gives us

The v4 SDK has dataset support but no `aevaluate()`-equivalent helper.
The user-facing surface is:

```python
lf = Langfuse()  # or get_client()

# Bootstrap (idempotent — see Decision §3)
lf.create_dataset(name=..., description=..., metadata=...)
lf.create_dataset_item(dataset_name=..., input=..., expected_output=..., metadata=...)

# Iterate
dataset = lf.get_dataset(name)
for item in dataset.items:
    with item.run(run_name="experiment-name") as root_span:
        output = await target_async_fn(item.input)
        root_span.update(output=output)
        for ev in evaluators:
            score_value = await ev(item.input, output, item.expected_output) \
                          if iscoroutinefunction(ev) else \
                          ev(output, item.expected_output)
            lf.score_current_trace(name=ev.__name__, value=score_value)
```

`item.run(...)` is a context manager that creates the eval-run trace and
links it to the dataset item, so the Langfuse UI groups all runs of an
experiment together (the equivalent of LangSmith's "experiment").

## Decision

### §1 — Pattern (apply mechanically to all 3 eval scripts)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor       # only if a sync target
from inspect import iscoroutinefunction
from langfuse import get_client

async def run_eval(
    target,                  # async callable: (inputs: dict) -> dict
    dataset_name: str,
    evaluators: list,
    run_name: str,
    metadata: dict | None = None,
    concurrency: int | None = None,
):
    lf = get_client()
    dataset = lf.get_dataset(dataset_name)
    sem = asyncio.Semaphore(concurrency or int(os.getenv("LANGFUSE_EVAL_CONCURRENCY", "4")))
    summary: dict[str, list[float]] = {}

    async def _one(item):
        async with sem:
            with item.run(run_name=run_name, run_metadata=metadata or {}) as root:
                output = await target(item.input)
                root.update(output=output)
                for ev in evaluators:
                    if iscoroutinefunction(ev):
                        v = await ev(item.input, output, item.expected_output)
                    else:
                        v = ev(output, item.expected_output)
                    v = float(bool(v)) if isinstance(v, bool) else float(v)
                    lf.score_current_trace(name=ev.__name__, value=v)
                    summary.setdefault(ev.__name__, []).append(v)

    await asyncio.gather(*(_one(it) for it in dataset.items))
    lf.flush()
    return summary
```

This helper lives in **`01_exercises/evaluation/_lf_runner.py`** (created
by the `task-eval-bootstrap` bead) and is the **only** Langfuse-specific
code the three eval scripts need to import.

### §2 — Concurrency: `asyncio.Semaphore`, not `ThreadPoolExecutor`

**Revision from bead default.** The bead specified
`concurrent.futures.ThreadPoolExecutor`. Rejected: the existing target
functions are `async` (they `await graph.ainvoke(...)`), so a thread pool
would force an `asyncio.run()` per worker (multi-loop chaos) or require
`run_until_complete` plumbing. `asyncio.Semaphore(N)` + `gather` matches
the existing async target shape with no boilerplate.

Default `N=4`, matching the LangSmith `max_concurrency=4`. Override via
`LANGFUSE_EVAL_CONCURRENCY` env var.

### §3 — Dataset bootstrap: idempotent upsert (do not delete)

```python
def upsert_dataset(lf, name, description, examples, metadata=None):
    try:
        lf.create_dataset(name=name, description=description, metadata=metadata or {})
    except Exception:
        pass  # already exists; harmless

    existing = {(it.input, str(it.expected_output)) for it in lf.get_dataset(name).items}
    for ex in examples:
        key = (ex["inputs"], str(ex["outputs"]))
        if key in existing:
            continue
        lf.create_dataset_item(
            dataset_name=name,
            input=ex["inputs"],
            expected_output=ex["outputs"],
        )
```

**Behavior change vs. LangSmith** (intentional): the e2e script today
**deletes the dataset** every run, losing run history. The Langfuse port
preserves history — every run shows up under the same dataset, which is
how the Langfuse UI was designed to work.

If a workshop attendee really wants a clean slate they can delete via the
UI; we will document this in the bootstrap-doc bead.

### §4 — Evaluator signature normalization

Adapter rules in the runner (above):

- Sync evaluators are called as `ev(output, reference_output)` — the
  current sync evaluators use that 2-arg form.
- Async evaluators are called as `await ev(input, output, reference_output)`
  — the current async judges use the 3-arg form.
- Return value normalization: `bool → 0.0 / 1.0`, anything else cast to
  `float`. Strings or dicts will raise.

The evaluator modules under `01_exercises/evaluation/evaluators/` are
**NOT modified** by the eval-port tasks. They have no LangSmith imports
and their signatures are reused as-is.

### §5 — Tracing during eval: ON (reversed from current)

Today the scripts run `os.environ["LANGCHAIN_TRACING_V2"] = "false"` to
make eval faster. With Langfuse self-hosted (no public-internet round
trip) the cost of tracing is sub-millisecond per span and the value of
having the full agent trace attached to every eval row is the whole point
of the migration.

**Decision:** the port removes that environment line and lets the
configured `LangfuseCallbackHandler` (from story `7bl.7`) trace every
eval run. Each run will appear in the Langfuse UI grouped under the
dataset run name and linked to its dataset item.

### §6 — Score naming

Score `name` = the evaluator function's `__name__` (e.g.,
`answer_quality`, `correct_routing`, `tool_call_accuracy`). This:

- Matches LangSmith's behavior, so workshop docs that name scores stay
  accurate.
- Aggregates cleanly in the Langfuse UI's per-dataset score column.

If a future evaluator needs a different display name, override via the
optional `name=` arg in a thin wrapper — do not change the runner.

### §7 — Result aggregation: local summary print

LangSmith returned an `experiment_results` object with `.to_pandas()`.
Langfuse has no equivalent. Replacement: the runner returns a dict
`{evaluator_name: [scores]}` and each eval script prints:

```
======================================================
EVALUATION SUMMARY
======================================================
answer_quality   mean=0.83  n=12
correctness      mean=0.92  n=12
humanness        mean=4.25  n=12
```

The Langfuse UI is the source of truth for per-row drill-down; the local
print is for CI/CLI eyeballing.

## Alternatives considered

### A. Adopt Langfuse's "experiment runner" community examples verbatim

Langfuse's docs ship a `run_experiment` example that uses
`asyncio.create_task` directly. Rejected: that example doesn't bound
concurrency and will hammer Azure OpenAI quotas on a 50-row dataset.
Our `Semaphore`-bounded helper is ~5 lines longer and safe.

### B. Synchronous port (drop async from the eval scripts)

- **Pros:** Simplest possible runner.
- **Cons:** Target functions call `await graph.ainvoke(...)`. Removing
  async would require rewriting the agent invocation path, which is out
  of scope and error-prone.
- **Verdict:** rejected.

### C. Skip dataset upload; iterate the JSON file directly

- **Pros:** No dataset bootstrap step at all; pure local loop.
- **Cons:** Loses Langfuse UI's "Datasets" page — the entire reason this
  story exists. You'd lose per-row drill-down, score aggregation, and
  the ability to re-run an experiment against the same items.
- **Verdict:** rejected.

### D. Keep tracing OFF during eval (preserve current behavior)

- **Pros:** Faster runs by some milliseconds per span.
- **Cons:** Defeats half the point of the migration. With local Langfuse
  the latency is negligible.
- **Verdict:** rejected. Default ON.

### E. ThreadPoolExecutor (the bead's original default)

- See §2 above. Rejected because target functions are async.

## Consequences

### Positive

- One small helper (`_lf_runner.py`) absorbs all the framework difference.
  The 3 eval scripts each shrink: their `aevaluate(...)` block becomes
  `await run_eval(...)`.
- Per-row traces appear in the Langfuse UI alongside the score, giving
  workshop attendees a debugger view they did NOT have on LangSmith
  (because tracing was off).
- Dataset history is preserved across runs (was lost on every e2e run).

### Negative / risks

- Score aggregation is now a local print, not a returned object. Anyone
  who built tooling on top of `results.to_pandas()` would break — but
  nobody has.
- If a target function is changed from async to sync in the future,
  the runner will hit `TypeError: object dict is not awaitable`. We
  could add an `iscoroutinefunction` check around the target call too,
  symmetric with the evaluator handling. Followup, not gating.
- Idempotent upsert relies on a `(input, str(expected_output))` tuple
  for dedup. If two examples have identical inputs but different ground
  truths, the duplicate guard would let both through (correct). If two
  examples are byte-identical, only one survives (also correct). Edge
  case: dict key ordering when comparing — mitigated by `str()` coercion
  on a Python dict whose ordering is insertion-stable since 3.7.

### Where this ADR is consumed

- `task-eval-bootstrap` (`travel-multi-agent-workshop-kjf`) builds
  `_lf_runner.py` per §1 and `upsert_dataset(...)` per §3.
- `task-eval-e2e`, `task-eval-routing`, `task-eval-tool` rewrite each
  script's `aevaluate(...)` block to call `run_eval(...)` and replace
  the `Client(...)` bootstrap with `upsert_dataset(...)`.

## Open questions / explicit non-decisions

- **LLM-as-judge model.** Out of scope — judges keep their existing
  Azure OpenAI deployment. ADR is silent on whether to switch them to
  the Langfuse "Evaluations" feature (server-side scheduled judges);
  punt to a future ADR if/when needed.
- **Cost tracking.** Langfuse can compute token cost per generation when
  models are configured in the UI. Out of scope here; addressed in the
  bootstrap-doc bead.
- **Snapshotting expected outputs.** Some teams snapshot a "good run" as
  expected output. Out of scope.

## References

- ADR-0001: SDK version pin → [./0001-langfuse-sdk-version.md](./0001-langfuse-sdk-version.md)
- ADR-0002: `@traceable` mapping → [./0002-traceable-to-observe-mapping.md](./0002-traceable-to-observe-mapping.md)
- Langfuse Datasets docs: <https://langfuse.com/docs/datasets/overview>
- Langfuse Evaluation docs: <https://langfuse.com/docs/evaluation/overview>
