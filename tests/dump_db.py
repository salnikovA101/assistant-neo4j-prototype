"""Скрипт для экспорта данных из Neo4j по run_id в JSON-файл."""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from neo4j import GraphDatabase

from server.utils.config import load_config


def dump_database(run_id: str):
    """Выгружает узлы и связи из Neo4j по run_id и сохраняет в db_dump.json."""
    config = load_config()
    neo4j_config = config.neo4j

    uri = neo4j_config.uri
    user = neo4j_config.user
    password = neo4j_config.password

    print(f"Подключение к Neo4j по адресу: {uri}")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:
        print(f"Ошибка при подключении к Neo4j: {e}")
        return

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
                            "properties": record["a_props"],
                        }
                    )

                b_id = record["b_id"]
                if b_id not in seen_nodes:
                    seen_nodes.add(b_id)
                    data["nodes"].append(
                        {
                            "id": b_id,
                            "labels": record["b_labels"],
                            "properties": record["b_props"],
                        }
                    )

                data["relationships"].append(
                    {
                        "id": record["r_id"],
                        "type": record["r_type"],
                        "properties": record["r_props"],
                        "start": a_id,
                        "end": b_id,
                    }
                )
    except Exception as e:
        print(f"Ошибка при выгрузке данных: {e}")
    finally:
        driver.close()

    output_file = Path(__file__).parent / f"db_dump_{run_id}.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Данные по run_id '{run_id}' успешно выгружены в файл: {output_file}")
        print(
            f"Выгружено {len(data['nodes'])} узлов "
            f"и {len(data['relationships'])} связей."
        )
    except Exception as e:
        print(f"Ошибка при сохранении в файл: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дамп базы данных по run_id")
    parser.add_argument("run_id", help="Идентификатор запуска (run_id)")
    args = parser.parse_args()

    dump_database(args.run_id)
