# Cypher Query Examples

> These examples demonstrate correct query patterns with mandatory provenance fields.
> Always include `evidence`, `source_file`, and `chunk_id` in RETURN statements.

---

## Examples

### Example 1: Find nodes connected by relationship type and property filter

> Demonstrates: filtering by node property with case-insensitive regex, returning provenance fields.

```cypher
MATCH (c:EnvironmentCondition)-[r]-(m:Metabolite)
WHERE c.name =~ '(?i).*temperature.*'
RETURN c.name AS condition, type(r) AS relation, m.name AS metabolite,
       r.evidence AS evidence, r.source_file AS source_file, r.chunk_id AS chunk_id
LIMIT 20
```

---

### Example 2: Find multi-hop paths between node types

> Demonstrates: variable-length path traversal (1–3 hops), collecting nodes and relationship metadata from the path.

```cypher
MATCH path = (m:Microbe)-[*1..3]-(q:Quality)
WHERE m.name =~ '(?i).*pseudomonas.*'
RETURN
  [n IN nodes(path) | n.name] AS path_nodes,
  [r IN relationships(path) | {
    type: type(r),
    evidence: r.evidence,
    source_file: r.source_file,
    chunk_id: r.chunk_id,
    confidence: r.confidence
  }] AS path_rels
LIMIT 20
```

---

### Example 3: Find all relationships for a named node (undirected)

> Demonstrates: undirected relationship search by name, returning confidence score.

```cypher
MATCH (n)-[r]-(m)
WHERE n.name =~ '(?i).*TVC.*'
RETURN n.name AS source, type(r) AS relation, m.name AS target,
       r.evidence AS evidence, r.source_file AS source_file,
       r.chunk_id AS chunk_id, r.confidence AS confidence
LIMIT 20
```

---

### Example 4: Filter by relationship property (confidence threshold)

> Demonstrates: filtering relationships by numeric property, useful for high-confidence results only.

```cypher
MATCH (a)-[r]-(b)
WHERE r.confidence > 0.8
RETURN a.name AS source, b.name AS target, type(r) AS relation,
       r.evidence AS evidence, r.source_file AS source_file,
       r.chunk_id AS chunk_id, r.confidence AS confidence
LIMIT 20
```

---

### Example 5: Filter by run_id and community (directed)

> Demonstrates: directed relationship with multiple filters including run metadata and Leiden community.

```cypher
MATCH (a)-[r]->(b)
WHERE r.run_id = '20260419_legacy'
  AND a.leiden_community = 4
  AND b.leiden_community = 4
RETURN a.name AS source, type(r) AS relation, b.name AS target,
       r.evidence AS evidence, r.source_file AS source_file,
       r.chunk_id AS chunk_id, r.run_id AS run_id
LIMIT 20
```

---

### Example 6: Graph overview (all nodes with optional relationships)

> Demonstrates: OPTIONAL MATCH for nodes that may have no outgoing relationships.

```cypher
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n.name AS node, type(r) AS relation, m.name AS connected,
       r.evidence AS evidence, r.source_file AS source_file, r.chunk_id AS chunk_id
LIMIT 20
```

---

Previous successful queries in this session (use as reference):

{history}
