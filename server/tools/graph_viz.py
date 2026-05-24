"""
Извлечение данных графа (nodes + edges) из Neo4j для визуализации.

Переиспользует последний успешный Cypher-запрос из GraphQA,
перезапуская его через raw driver для получения полных Node/Relationship объектов.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

_MATCH_RE = re.compile(
    r"(MATCH\s+.*?)(?=\s+RETURN\b)",
    re.DOTALL | re.IGNORECASE,
)

_NODE_COLORS: Dict[str, str] = {
    "Metabolite": "#c990c0",
    "Microbe": "#569480",
    "EnvironmentCondition": "#f0a85e",
}
_DEFAULT_COLOR = "#a5abb6"


class GraphVizExtractor:
    """
    Извлекает подграф из Neo4j для визуализации в UI.

    Работает на основе последнего успешного Cypher-запроса из GraphQA:
    1. Берёт MATCH-часть из оригинального запроса.
    2. Перезапускает с `RETURN *`, чтобы получить полные Node/Relationship объекты.
    3. Парсит результат в структуру {nodes: [...], edges: [...]}.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("GraphVizExtractor инициализирован")

    def close(self):
        self._driver.close()

    def _build_viz_cypher(self, original_cypher: str) -> Optional[str]:
        """
        Трансформирует оригинальный Cypher: берёт MATCH (+WHERE) часть,
        заменяет RETURN на `RETURN *` для получения полных объектов.
        """
        match = _MATCH_RE.search(original_cypher)
        if not match:
            logger.warning(
                f"Не удалось извлечь MATCH-клаузу из Cypher: {original_cypher}"
            )
            return None

        match_clause = match.group(1).strip()

        limit_match = re.search(r"LIMIT\s+(\d+)", original_cypher, re.IGNORECASE)
        limit = int(limit_match.group(1)) if limit_match else 50

        viz_cypher = f"{match_clause}\nRETURN * LIMIT {limit}"
        logger.info(f"Viz Cypher: {viz_cypher}")
        return viz_cypher

    def _extract_graph_from_records(
        self, records: list
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Парсит записи Neo4j (содержащие Node/Relationship объекты) в {nodes, edges}.
        """
        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges_list: List[Dict[str, Any]] = []
        seen_edges: set = set()

        for record in records:
            for value in record.values():
                self._process_value(value, nodes_map, edges_list, seen_edges)

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges_list,
        }

    def _process_value(
        self,
        value: Any,
        nodes_map: Dict[str, Dict[str, Any]],
        edges_list: List[Dict[str, Any]],
        seen_edges: set,
    ):
        """Рекурсивно обрабатывает значение из записи Neo4j."""
        from neo4j.graph import Node, Relationship, Path

        if isinstance(value, Node):
            self._add_node(value, nodes_map)
        elif isinstance(value, Relationship):
            self._add_relationship(value, nodes_map, edges_list, seen_edges)
        elif isinstance(value, Path):
            for node in value.nodes:
                self._add_node(node, nodes_map)
            for rel in value.relationships:
                self._add_relationship(rel, nodes_map, edges_list, seen_edges)
        elif isinstance(value, list):
            for item in value:
                self._process_value(item, nodes_map, edges_list, seen_edges)

    def _add_node(self, node: Any, nodes_map: Dict[str, Dict[str, Any]]):
        """Добавляет ноду в карту, если ещё нет."""
        node_id = str(node.element_id)
        if node_id in nodes_map:
            return

        labels = list(node.labels)

        primary_label = "Unknown"
        for lbl in labels:
            if lbl in _NODE_COLORS:
                primary_label = lbl
                break

        if primary_label == "Unknown" and labels:
            primary_label = labels[0]

        props = dict(node)
        name = props.get("name", f"Node {node_id[-6:]}")

        nodes_map[node_id] = {
            "id": node_id,
            "label": name,
            "group": primary_label,
            "color": _NODE_COLORS.get(primary_label, _DEFAULT_COLOR),
            "properties": props,
        }

    def _add_relationship(
        self,
        rel: Any,
        nodes_map: Dict[str, Dict[str, Any]],
        edges_list: List[Dict[str, Any]],
        seen_edges: set,
    ):
        """Добавляет связь в список, если ещё нет."""
        edge_id = str(rel.element_id)
        if edge_id in seen_edges:
            return
        seen_edges.add(edge_id)

        start_id = str(rel.start_node.element_id)
        end_id = str(rel.end_node.element_id)

        self._add_node(rel.start_node, nodes_map)
        self._add_node(rel.end_node, nodes_map)

        props = dict(rel)
        display_props = {}
        for k, v in props.items():
            display_props[k] = v

        edges_list.append(
            {
                "from": start_id,
                "to": end_id,
                "label": rel.type,
                "properties": display_props,
            }
        )

    def execute_and_parse(self, cypher: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Выполняет готовый Cypher-запрос и парсит результат в {nodes, edges}.

        Используется GraphFilterAgent, который самостоятельно генерирует Cypher.

        Args:
            cypher: Готовый Cypher-запрос для визуализации.

        Returns:
            Словарь {nodes: [...], edges: [...]}.
        """
        try:
            with self._driver.session(default_access_mode="READ") as session:
                result = session.run(cypher)
                records = list(result)

                if not records:
                    logger.info("Viz-запрос вернул пустой результат")
                    return {"nodes": [], "edges": []}

                graph_data = self._extract_graph_from_records(records)
                logger.info(
                    f"Граф извлечён: {len(graph_data['nodes'])} нод, "
                    f"{len(graph_data['edges'])} связей"
                )
                return graph_data

        except Exception as e:
            logger.error(f"Ошибка извлечения графа: {e}")
            return {"nodes": [], "edges": []}

    def extract(self, original_cypher: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Извлекает данные графа по оригинальному Cypher-запросу (legacy).

        Args:
            original_cypher: Оригинальный успешный Cypher из GraphQA.

        Returns:
            Словарь {nodes: [...], edges: [...]}.
        """
        viz_cypher = self._build_viz_cypher(original_cypher)
        if not viz_cypher:
            return {"nodes": [], "edges": []}

        return self.execute_and_parse(viz_cypher)
