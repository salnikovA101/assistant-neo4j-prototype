Node labels and properties:
- Microbe {name: STRING, leiden_community: INTEGER}
- StarterCulture {name: STRING, leiden_community: INTEGER}
- Metabolite {name: STRING, leiden_community: INTEGER}
- EnvironmentCondition {name: STRING, leiden_community: INTEGER}

Relationship types:
- PRODUCES, CONSUMES, INHIBITS, STIMULATES, REQUIRES

Valid relationships (Source -> RELATION -> Target):
- Microbe|StarterCulture -> PRODUCES|CONSUMES -> Metabolite|EnvironmentCondition
- Microbe|StarterCulture -> INHIBITS|STIMULATES -> Microbe|StarterCulture
- Microbe|StarterCulture -> REQUIRES -> Metabolite|EnvironmentCondition
- Metabolite -> INHIBITS|STIMULATES|PRODUCES|CONSUMES -> Metabolite
- Metabolite -> INHIBITS|STIMULATES -> Microbe|StarterCulture
- Metabolite -> REQUIRES -> EnvironmentCondition

IMPORTANT RULES: 
1. EnvironmentCondition is NEVER the source of any relationship.
2. INHIBITS and STIMULATES relationships NEVER target EnvironmentCondition.

Relationship properties (apply to all relationships):
- confidence: FLOAT  -- reliability score [0.0, 1.0]
- evidence: STRING   -- verbatim quote from source document
- source_file: STRING
- chunk_id: STRING
- run_id: STRING
