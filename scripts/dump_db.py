#!/usr/bin/env python3
"""Export Neo4j nodes and relationships for a run_id to JSON (no embeddings).

Writes dumps/db_dump_{run_id}.json at the repo root (not under scripts/).

  .venv/bin/python scripts/dump_db.py <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.utils.config import load_config  # noqa: E402

_DUMP_DIR = ROOT / "dumps"


def _driver(uri: str, user: str, password: str):
    candidates = [uri]
    if "host.docker.internal" in uri:
        candidates.append(uri.replace("host.docker.internal", "localhost"))
        candidates.append(uri.replace("host.docker.internal", "127.0.0.1"))
    last = None
    for cand in candidates:
        drv = GraphDatabase.driver(cand, auth=(user, password))
        try:
            drv.verify_connectivity()
            print(f"Connected: {cand}")
            return drv
        except ServiceUnavailable as e:
            last = e
            drv.close()
    raise last or RuntimeError("Neo4j unreachable")


def _strip_embeddings(props) -> dict:
    """Drop node embedding and r.evidence_embedding (and any *_embedding)."""
    if not props:
        return {}
    out = {}
    for key, value in dict(props).items():
        if key == "embedding" or key.endswith("_embedding"):
            continue
        out[key] = value
    return out


def dump_database(run_id: str) -> Path:
    """Выгружает узлы и связи из Neo4j по run_id в dumps/db_dump_{run_id}.json."""
    config = load_config()
    driver = _driver(config.neo4j.uri, config.neo4j.user, config.neo4j.password)

    data = {"nodes": [], "relationships": []}

    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (a)-[r {run_id: $run_id}]->(b) "
                "RETURN "
                "elementId(a) as a_id, labels(a) as a_labels, properties(a) as a_props, "
                "elementId(b) as b_id, labels(b) as b_labels, properties(b) as b_props, "
                "elementId(r) as r_id, type(r) as r_type, properties(r) as r_props",
                run_id=run_id,
            )

            seen_nodes = set()
            for record in result:
                a_id = record["a_id"]
                if a_id not in seen_nodes:
                    seen_nodes.add(a_id)
                    data["nodes"].append(
                        {
                            "id": a_id,
                            "labels": record["a_labels"],
                            "properties": _strip_embeddings(record["a_props"]),
                        }
                    )

                b_id = record["b_id"]
                if b_id not in seen_nodes:
                    seen_nodes.add(b_id)
                    data["nodes"].append(
                        {
                            "id": b_id,
                            "labels": record["b_labels"],
                            "properties": _strip_embeddings(record["b_props"]),
                        }
                    )

                data["relationships"].append(
                    {
                        "id": record["r_id"],
                        "type": record["r_type"],
                        "properties": _strip_embeddings(record["r_props"]),
                        "start": a_id,
                        "end": b_id,
                    }
                )
    finally:
        driver.close()

    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    output_file = _DUMP_DIR / f"db_dump_{run_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"Данные по run_id '{run_id}' успешно выгружены в файл: {output_file}")
    print(
        f"Выгружено {len(data['nodes'])} узлов "
        f"и {len(data['relationships'])} связей."
    )
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дамп базы данных по run_id")
    parser.add_argument("run_id", help="Идентификатор запуска (run_id)")
    args = parser.parse_args()
    dump_database(args.run_id)
