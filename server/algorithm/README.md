# Algorithm V6

Edge-native GraphRAG pipeline:

S1 embed sq → S2 ANN (L per index → merge → L_raw_max) →
S2b Ettin CE (full L_raw_max pool → keep L) →
S3 N+1 graphs (anchors=L + bridges) **once** →
S4 profitable tours until path budget (TOARP, prize once) → reshape SPINE+FANS
(for viz / S5 spine dedup) → format UNIT as the walk-ordered tour →
S5 exact spine-evidence-seq dedup, keep up to budget →
emit: sort by S4 score, cap (`emit_top_k_*`), drop score cliff (`emit_score_frac`).

`effort` only sets how many units to mine (low=10 / medium=15 / hard=20)
and how many to emit (5 / 10 / 15). It comes from the UI search-depth control
(`server/core/turn_state.py`), not from the assistant model.

S2/S3 optionally restrict to one relationship `run_id` from `server/config.yaml`
(`Params.run_id`). Non-empty: Cypher 25 `SEARCH … WHERE r.run_id = $run_id`
inside the existing per-type vector indexes (same L / L_raw_max). Indexes must
include `WITH [r.run_id]` — `python scripts/vectorize_edges.py --recreate-indexes`.
Empty `run_id`: unfiltered ANN (legacy `queryRelationships`) and a startup warning.
`rerank_enabled` in `server/config.yaml` (default for deploy: false) is passed into
`Params`; false skips Ettin and keeps ANN sim order.

### S4

- `s4_paths_per_graph` (default 3) profitable tours per S3 graph per fill
  round, length `min_path_len`..`max_hops` (default 1..10). Drop a tour
  with fewer than `s4_min_prize_edges` prize arcs or score ≤ 0. After each
  tour, collected arcs get local p=0 (prize once). S4 repeats rounds until
  the path budget is unique spines (or prize runs out). Collecting a prize
  arc does **not** promote demoted ANN into `prize_top` (frozen ranks).
- Global start (all edges); score = Σ rank contribs.
- Non-bridge edges ranked by `(CE|sim) · p`; top `prize_top` (default 50)
  get linear rank prizes. Prize is shared across graphs in one S4 pass.
- Demoted ANN pay `bridge_cost_c0·(1+γ·x²)·(2−p)`; structural bridges pay
  flat `bridge_struct_cost·(2−p)`.
- Star walks reshaped to SPINE + FANS via `unit_reshape` (UI roles + S5).
  Print tags (`linger_hubs`): entry unmarked; rays **and** exit get `@Hub`.

Unit = hop-DP tour in walk order (not spine-then-FANS dump). Each edge prints
as a card: `Label: A —REL→ Label: B` and the verbatim quote with
`(source; conf)` on the quote line. `@Hub` on a triple means still at that
vertex (sibling incidents, not the next process step). Endpoints use primary
Neo4j labels from
`Microbe|Metabolite|StarterCulture|EnvironmentCondition` (extra labels dropped).
Evidence may not repeat inside one unit; S5 drops exact SPINE evidence
duplicates in the pool.

ANN/CE/bridges run once per question. S4 fills the path cap in one
stage (low=10 / medium=15 / hard=20 accepted units), then emit cuts to
what the assistant sees: `emit_top_k_*` (easy=5 / medium=10 / hard=15),
then `emit_score_frac` (drop below 25% of the best score inside that cap).
Eval reports recall@1/2/3/5/10/15/20 on the uncut ranked pool; headline
metrics are on the emitted set.

## Run

```bash
# unit
.venv/bin/python -m pytest tests/algorithm tests/test_graph_viz.py -q

# eval (needs Neo4j + OpenRouter embeddings; mock_decompose SLM unless --sq-cache)
.venv/bin/python tests/evaluate_v6.py --effort auto --limit 1 --sq-cache
# reports → tests/reports/v6/ (wiped each run)
# sq cache → tests/reports/v6_cache/sq_cache.json (preserved)

# Build S3 graph cache once, then sweep S4 params without ANN/CE:
.venv/bin/python tests/evaluate_v6.py --sq-cache --graph-cache
.venv/bin/python tests/evaluate_v6.py --sq-cache \
  --graph-cache --graph-cache-read-only
# graph cache → tests/reports/v6_cache/graphs/{qid}.json
# Invalidates on L / L_raw_max / bridge_top / branch_cap / rerank_enabled /
# run_id / sq texts / framing id change.
```

Wired to the assistant via `ask_subgraph` (`server/tools/subgraph_search.py`).
