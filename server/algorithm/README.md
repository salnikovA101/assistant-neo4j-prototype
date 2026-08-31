# Retrieval pipeline

Edge-native GraphRAG pipeline:

S1 embed sq → S2 ANN (L per index → merge → L_raw_max) →
S2b Ettin CE (full L_raw_max pool → keep L) →
S3 per-sq graphs (anchors=L + bridges) **once** →
S4 carousel (one tour / sq / round, shared `p`) until path budget →
reshape SPINE+FANS (for viz / S5 spine dedup) → format UNIT as the walk-ordered tour →
S5 exact spine-evidence-seq dedup (keep first, carousel order) →
assistant gets the full S5 pool (`accepted` = `accepted_all`).

`effort` only picks `max_paths_low|medium|high` (how many units to mine).
It comes from the UI search-depth control (`server/core/turn_state.py`),
not from the assistant model. Defaults live in `Params`.

S2/S3 optionally restrict to one relationship `run_id` from `server/config.yaml`
(`Params.run_id`). Non-empty: Cypher 25 `SEARCH … WHERE r.run_id = $run_id`
inside the existing per-type vector indexes (same L / L_raw_max). Indexes must
include `WITH [r.run_id]` — `python scripts/vectorize_edges.py --recreate-indexes`.
Empty `run_id`: unfiltered ANN (legacy `queryRelationships`) and a startup warning.
`rerank_enabled` in `server/config.yaml` (default for deploy: false) is passed into
`Params`; false skips Ettin and keeps ANN sim order.

### S4

Carousel over sq graphs in dict order: round 1 is always complete (one tour
per graph if DP finds a path). Later rounds continue until `effort_max_paths()`;
the last round may be partial.

Shared `p` starts at 1 on the UNION of all graph `edge_key`s. After a tour,
every walk key gets `p *= s4_p_decay` (default 0.7; `0` zeros the keys).
Ranks are recomputed each step — no freeze.

On a graph, **all** edges (anchors and bridges) share one list. Sort by
`(CE|sim) · p`. `p` only moves order; prize/cost amounts come from rank:

- `r ≤ prize_top`: `prize_rank_max · (K−r+1)/K` (linear `+1 → ~0`)
- `r > prize_top`: `−prize_rank_max · x^s4_cost_power` (`s4_cost_power=1.5`),
  `x=(r−K)/(N−K)`; last rank pays `prize_rank_max`

DP inside a tour is unchanged: length `min_path_len`..`max_hops`,
no edge/evidence revisit, drop if fewer than `s4_min_prize_edges` prize arcs
or score ≤ 0. Global start (all edges); score = Σ rank contribs.

Star walks reshaped to SPINE + FANS via `unit_reshape` (UI roles + S5).
Print tags (`linger_hubs`): entry unmarked; rays **and** exit get `@Hub`.

### S5

Exact `spine_evidence_seq` duplicates drop; the **first** (carousel order) is
kept. The assistant sees
the whole S5 pool in carousel order.

Unit = hop-DP tour in walk order (not spine-then-FANS dump). Each edge prints
as a card: `Label: A —REL→ Label: B` and the verbatim quote with
`(source; conf)` on the quote line. `@Hub` on a triple means still at that
vertex (sibling incidents, not the next process step). Endpoints use primary
Neo4j labels from
`Microbe|Metabolite|StarterCulture|EnvironmentCondition` (extra labels dropped).
Evidence may not repeat inside one unit.

ANN/CE/bridges run once per question. Eval headline recall is on the full
`accepted` set; recall@k uses a **score-sorted copy** of that pool.

## Run

```bash
# unit
.venv/bin/python -m pytest tests/algorithm tests/test_graph_viz.py -q

# eval (needs Neo4j + local Ollama embeddings; mock_decompose SLM unless --sq-cache)
.venv/bin/python tests/evaluate_v6.py --effort auto --limit 1 --sq-cache
# reports → tests/reports/v6/ (wiped each run)
# sq cache (open) → tests/reports/v6_cache/ (preserved)

# Build S3 graph cache once, then sweep S4 params without ANN/CE:
.venv/bin/python tests/evaluate_v6.py --sq-cache --graph-cache
.venv/bin/python tests/evaluate_v6.py --sq-cache \
  --graph-cache --graph-cache-read-only
# graph cache → tests/reports/v6_cache/graphs/{qid}.json
# Invalidates on L / L_raw_max / bridge_top / branch_cap / rerank_enabled /
# run_id / sq texts / framing id change.
```

Wired to the assistant via `ask_subgraph` (`server/tools/subgraph_search.py`).
