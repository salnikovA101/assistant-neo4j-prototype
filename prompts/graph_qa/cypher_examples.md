# Cypher Query Examples

> These examples demonstrate correct query patterns with mandatory provenance fields.
> Always include `evidence`, `source_file`, and `chunk_id` in RETURN statements.
> Strict labels to use: Microbe, Metabolite, EnvironmentCondition, StarterCulture.

---

## Examples

### Example 1: Find connected nodes by property filter (Node to Node)

> Demonstrates: filtering by node property with case-insensitive regex using strict node labels.

```cypher
MATCH (c:StarterCulture)-[r]-(m:Metabolite)
WHERE r.run_id = '{run_id}'
  AND c.name =~ '(?i).*kefir grain.*'
RETURN c.name AS culture, type(r) AS relation, m.name AS metabolite,
       r.evidence AS evidence, r.source_file AS source_file, r.chunk_id AS chunk_id
LIMIT {limit}
```

---

### Example 2: Find multi-hop paths between node types

> Demonstrates: variable-length path traversal (1–3 hops), collecting nodes and relationship metadata safely using ALL() for run_id.

```cypher
MATCH path = (m:Microbe)-[*1..3]-(c:EnvironmentCondition)
WHERE ALL(r IN relationships(path) WHERE r.run_id = '{run_id}')
  AND m.name =~ '(?i).*lactobacillus.*'
RETURN
  [n IN nodes(path) | n.name] AS path_nodes,
  [r IN relationships(path) | {{
    type: type(r),
    evidence: r.evidence,
    source_file: r.source_file,
    chunk_id: r.chunk_id,
    confidence: r.confidence
  }}] AS path_rels
LIMIT {limit}
```

---

### Example 3: Find all relationships for a specific node (Neighborhood)

> Demonstrates: finding the immediate environment of a specific entity. NOTE the strict use of node labels to prevent full database scans.

```cypher
MATCH (n:Microbe)-[r]-(m:Metabolite)
WHERE r.run_id = '{run_id}'
  AND n.name =~ '(?i).*helicobacter.*'
RETURN n.name AS source, type(r) AS relation, m.name AS target,
       r.evidence AS evidence, r.source_file AS source_file,
       r.chunk_id AS chunk_id, r.confidence AS confidence
LIMIT {limit}
```

---

### Example 4: Filter by relationship property (Confidence Threshold)

> Demonstrates: filtering relationships by a numeric property (confidence > 0.8), while maintaining strict node labels.

```cypher
MATCH (a:Microbe)-[r]-(b:Microbe)
WHERE r.run_id = '{run_id}'
  AND r.confidence > 0.8
RETURN a.name AS source, b.name AS target, type(r) AS relation,
       r.evidence AS evidence, r.source_file AS source_file,
       r.chunk_id AS chunk_id, r.confidence AS confidence
LIMIT {limit}
```

---

### Example 5: Multiple keyword search (Regex OR)

> Demonstrates: searching for multiple entities simultaneously using a regex OR `|` operator.

```cypher
MATCH (n:Microbe)-[r]-(m:Metabolite)
WHERE r.run_id = '{run_id}'
  AND n.name =~ '(?i).*(lactococcus|streptococcus).*'
RETURN n.name AS microbe, type(r) AS relation, m.name AS metabolite,
       r.evidence AS evidence, r.source_file AS source_file, r.chunk_id AS chunk_id
LIMIT {limit}
```

---

### Example 6: Safe OPTIONAL MATCH (Graph Overview for an Entity)

> Demonstrates: safely looking up a node that might not have relationships. The `run_id` filter is placed inline `{{run_id: '{run_id}'}}` inside the pattern to prevent `NULL` filtering bugs.

```cypher
MATCH (n:StarterCulture)
WHERE n.name =~ '(?i).*curd starter.*'
OPTIONAL MATCH (n)-[r {{run_id: '{run_id}'}}]-(m:Metabolite)
RETURN n.name AS node, type(r) AS relation, m.name AS connected,
       r.evidence AS evidence, r.source_file AS source_file, r.chunk_id AS chunk_id
LIMIT {limit}
```

---

### Example 7: Find Common Connections (Shared Neighbors)

> Demonstrates: finding a central node (e.g., Metabolite) that connects two different entities.

```cypher
MATCH (m1:Microbe)-[r1]-(c:Metabolite)-[r2]-(m2:Microbe)
WHERE r1.run_id = '{run_id}' 
  AND r2.run_id = '{run_id}'
  AND m1.name =~ '(?i).*lactobacillus.*'
  AND m2.name =~ '(?i).*saccharomyces.*'
RETURN m1.name AS microbe_1, type(r1) AS relation_1, c.name AS shared_metabolite, 
       type(r2) AS relation_2, m2.name AS microbe_2,
       r1.evidence AS evidence_1, r2.evidence AS evidence_2,
       r1.source_file AS source_file, r1.chunk_id AS chunk_id
LIMIT {limit}
```

---

### Example 8: Find interactions involving an EnvironmentCondition

> Demonstrates: querying EnvironmentConditions correctly. EnvironmentConditions are NEVER the source of relationships, so they must be queried via incoming relationships (e.g., REQUIRES or PRODUCES).

```cypher
MATCH (source:Microbe|Metabolite|StarterCulture)-[r:REQUIRES|PRODUCES|INHIBITS|STIMULATES]->(c:EnvironmentCondition)
WHERE r.run_id = '{run_id}'
  AND c.name =~ '(?i).*(pH|density|firmness).*'
RETURN source.name AS acting_entity, labels(source)[0] AS type, type(r) AS relation, c.name AS condition,
       r.evidence AS evidence, r.source_file AS source_file, r.chunk_id AS chunk_id
LIMIT {limit}
```

---

### Example 9: Describe a sequential biological process (Allergen reduction)

> Demonstrates: safely extracting a long variable-length path by filtering the start node and checking if ANY intermediate node matches required concepts (like proteins, enzymes, or abstract concepts without explicit labels).

```cypher
MATCH path = (start_node:Microbe|StarterCulture)-[*1..4]-(end_node)
WHERE ALL(r IN relationships(path) WHERE r.run_id = '{run_id}')
  AND start_node.name =~ '(?i).*(culture|lactobacillus|strain).*'
  AND ANY(n IN nodes(path) WHERE n.name =~ '(?i).*(beta-lactoglobulin|alpha-casein|endopeptidase).*')
  AND end_node.name =~ '(?i).*(allergenicity|allergenic potential|firmness).*'
RETURN 
  [n IN nodes(path) | {{name: n.name, labels: labels(n)}}] AS sequence_of_entities,
  [r IN relationships(path) | {{
    type: type(r),
    evidence: r.evidence,
    source_file: r.source_file,
    chunk_id: r.chunk_id,
    confidence: r.confidence
  }}] AS relationship_details
LIMIT {limit}
```

---

### Example 10: Complex impact mechanism (Starter Culture -> Metabolite -> End Effect)

> Demonstrates: finding how specific starter cultures or microbes affect the production of a metabolite (like GABA) and subsequent effects (bioavailability/stability). Uses broad node matching to catch abstract concepts.

```cypher
MATCH path = (s:StarterCulture|Microbe)-[*1..3]-(end_effect)
WHERE ALL(r IN relationships(path) WHERE r.run_id = '{run_id}')
  AND s.name =~ '(?i).*(lactobacillus|streptococcus|starter).*'
  AND ANY(n IN nodes(path) WHERE n.name =~ '(?i).*(GABA|gamma-aminobutyric acid|bioavailability|stability).*')
RETURN 
  [n IN nodes(path) | {{name: n.name, labels: labels(n)}}] AS sequence_of_entities,
  [r IN relationships(path) | {{
    type: type(r),
    evidence: r.evidence,
    source_file: r.source_file,
    confidence: r.confidence
  }}] AS relationship_details
LIMIT {limit}
```

---

### Example 11: Multi-criteria strain selection (Pathogen suppression & Enzymatic activity)

> Demonstrates: finding a Microbe or Starter Culture that fulfills multiple functional requirements (e.g., inhibiting a pathogen like Helicobacter AND affecting enzymes/proteins like urease or bacteriocins). Uses an OR regex to grab all relevant paths for the LLM to analyze and synthesize recommendations.

```cypher
MATCH path = (s:StarterCulture|Microbe)-[*1..3]-(target)
WHERE ALL(r IN relationships(path) WHERE r.run_id = '{run_id}')
  AND s.name =~ '(?i).*(starter|culture|lactobacillus|streptococcus|bifidobacterium|microbe).*'
  AND ANY(n IN nodes(path) WHERE n.name =~ '(?i).*(helicobacter|pylori|urease|bacteriocin).*')
RETURN 
  [n IN nodes(path) | {{name: n.name, labels: labels(n)}}] AS sequence_of_entities,
  [r IN relationships(path) | {{
    type: type(r),
    evidence: r.evidence,
    source_file: r.source_file,
    confidence: r.confidence
  }}] AS relationship_details
LIMIT {limit}
```
