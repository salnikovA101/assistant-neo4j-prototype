"""Скрипт для экспорта всех данных из Neo4j в JSON-файл."""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from neo4j import GraphDatabase

from server.utils.config import load_config


def dump_database():
    """Выгружает все узлы и связи из Neo4j и сохраняет в db_dump.json."""
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
            nodes_result = session.run(
                "MATCH (n) "
                "RETURN elementId(n) as id, labels(n) as labels, "
                "properties(n) as properties"
            )
            for record in nodes_result:
                data["nodes"].append(
                    {
                        "id": record["id"],
                        "labels": record["labels"],
                        "properties": record["properties"],
                    }
                )

            rels_result = session.run(
                "MATCH (n)-[r]->(m) "
                "RETURN elementId(r) as id, type(r) as type, "
                "properties(r) as properties, "
                "elementId(n) as start, elementId(m) as end"
            )
            for record in rels_result:
                data["relationships"].append(
                    {
                        "id": record["id"],
                        "type": record["type"],
                        "properties": record["properties"],
                        "start": record["start"],
                        "end": record["end"],
                    }
                )
    except Exception as e:
        print(f"Ошибка при выгрузке данных: {e}")
    finally:
        driver.close()

    output_file = Path(__file__).parent / "db_dump.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"База данных успешно выгружена в файл: {output_file}")
        print(
            f"Выгружено {len(data['nodes'])} узлов "
            f"и {len(data['relationships'])} связей."
        )
    except Exception as e:
        print(f"Ошибка при сохранении в файл: {e}")


if __name__ == "__main__":
    dump_database()
