import asyncio
import json
import logging
import re
import time

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from server.utils.config import load_config

logger = logging.getLogger(__name__)


class SubqueriesOutput(BaseModel):
    subqueries: list[str]


def extract_json_array(content: str) -> list[str]:
    """
    Extracts and validates a JSON object containing a list of subqueries using Pydantic.
    """
    # Try to find a JSON block wrapped in ```json ... ``` or similar
    json_match = re.search(r"```(?:json)?\n?(.*?)\n?```", content, re.DOTALL)
    text_to_parse = content
    if json_match:
        text_to_parse = json_match.group(1).strip()

    # Try to find the first '{' and last '}' if direct parse fails
    start_idx = text_to_parse.find("{")
    end_idx = text_to_parse.rfind("}")

    if start_idx != -1 and end_idx != -1:
        text_to_parse = text_to_parse[start_idx : end_idx + 1]

    try:
        validated = SubqueriesOutput.model_validate_json(text_to_parse)
        return [sq.strip() for sq in validated.subqueries if sq.strip()]
    except (ValidationError, ValueError) as e:
        logger.error(f"Failed to validate JSON with Pydantic: {e}\nContent was: {text_to_parse}")
        return []


async def decompose_query(question: str, config=None) -> list[str]:
    """
    Stage 1: Agentic Chunking.
    Uses the configured LLM to decompose a complex biological query into a JSON array of subqueries.
    """
    if config is None:
        config = load_config()

    # Get the correct profile (as specified by user, we use tool_profile)
    profile_name = config.llm.tool_profile
    llm_profile = getattr(config.llm.profiles, profile_name, None)

    if not llm_profile:
        logger.warning(f"Profile {profile_name} not found. Falling back to 'other'.")
        llm_profile = config.llm.profiles.other

    client = AsyncOpenAI(
        base_url=llm_profile.base_url or None,
        api_key=llm_profile.api_key or "not-needed",
        timeout=120.0,
    )

    system_prompt = """
        You are an expert scientific data ontologist specializing in Biological Knowledge Graphs and Dense Information Retrieval.
        Your task is to decompose a complex, multi-part user question into a series of focused, atomic, DECLARATIVE scientific statements and action-oriented assertions.

        CRITICAL RULES:
        1. NO QUESTIONS: Do not write questions (do not use question words like "What", "Which", "How", "Why" and never use question marks).
        2. DECLARATIVE EVIDENCE FORMAT: Write statements as declarative assertions, physical mechanisms, or biological actions (e.g., "microorganism inhibits urease activity"). This must closely match the style of peer-reviewed scientific evidence texts.
        3. ATOMIC BUT CONTEXTUAL: Break down complex queries, but ensure EACH subquery retains enough specific context to be meaningful on its own. Avoid overly broad statements.
        - BAD: "casein yields specific tripeptides" (Too broad, loses context).
        - GOOD: "lactic acid bacteria decrypt specific tripeptides from casein".
        4. NO SEMANTIC REDUNDANCY: Do not generate multiple subqueries that mean the exact same thing. Consolidate overlapping ideas.
        - BAD: ["enzyme converts X to Y", "enzyme catalyzes X", "enzyme produces Y"].
        - GOOD: ["enzyme converts X to Y"].
        5. UNPACK LISTS WITH CONTEXT: If the user lists multiple items (e.g., minerals, environments), create a separate subquery for each, but repeat the necessary context for every single item.
        6. AVOID COMBINATORIAL EXPLOSION: If unpacking multiple nested lists creates too many subqueries (e.g., > 8), group related concepts. Do not unpack hierarchical terms redundantly (e.g., if the prompt says "rheological properties (stiffness, yield stress)", use the specific terms and drop the general one).
        7. NO META-STATEMENTS: Do not write about "strategies", "requirements", "formulations", "analysis", or "mechanisms of". Extract ONLY the direct biological, chemical, or physical facts.
        - BAD: "compositional requirements for a starter culture to suppress pathogen"
        - GOOD: "starter culture suppresses pathogen"
        - BAD: "mechanism of yeast stimulating enzyme activity"
        - GOOD: "yeast stimulates enzyme activity"
        8. STRICT CONSTRAINTS (NO INVENTIONS): Rely strictly on the entities, mechanisms, and terms explicitly provided in the user's input. Do NOT introduce external concepts.
        9. ONLY JSON: Only return a valid JSON object matching the schema. No conversational text.

        JSON Schema Requirement:
        You must return a JSON object with a single key "subqueries" containing a list of strings.

        EXAMPLE 1 (Basic decomposition):
        Input: "What are the metabolic products of Lactobacillus plantarum during fermentation and how do they inhibit Staphylococcus aureus?"
        Output:
        {
        "subqueries": [
            "Lactobacillus plantarum produces metabolic products during fermentation",
            "metabolic products of Lactobacillus plantarum inhibit Staphylococcus aureus"
        ]
        }

        EXAMPLE 2 (Handling lists and avoiding redundancy):
        Input: "How does the glutamate decarboxylase (GAD) enzyme convert glutamate into GABA, and what is its physiological role in brain and liver cells?"
        Output:
        {
        "subqueries": [
            "glutamate decarboxylase (GAD) enzyme converts glutamate into GABA",
            "glutamate decarboxylase (GAD) enzyme performs a physiological role in brain cells",
            "glutamate decarboxylase (GAD) enzyme performs a physiological role in liver cells"
        ]
        }

        EXAMPLE 3 (Avoiding meta-statements and combinatorial explosion):
        Input: "Describe a comprehensive biotechnological strategy for maximum enrichment of a product with GABA using strains of Lactiplantibacillus plantarum. What specific environmental parameters (osmotic stress, cofactors) need to be optimized, and how does co-cultivation with yeast (Saccharomyces cerevisiae) stimulate the activity of the glutamate decarboxylase (GAD) system?"
        Output:
        {
        "subqueries": [
            "Lactiplantibacillus plantarum enriches product with GABA",
            "osmotic stress affects GABA production by Lactiplantibacillus plantarum",
            "cofactors affect GABA production by Lactiplantibacillus plantarum",
            "Saccharomyces cerevisiae co-cultivation stimulates glutamate decarboxylase (GAD) system activity in Lactiplantibacillus plantarum"
        ]
        }
    """

    for attempt in range(3):
        try:
            params = {
                "model": llm_profile.model,
                "messages": [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": question},
                ],
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": 4096,
            }

            if hasattr(llm_profile, "think") and not llm_profile.think:
                params["reasoning_effort"] = "none"

            response = await client.chat.completions.create(**params)

            content = response.choices[0].message.content or ""
            content = content.strip()

            subqueries = extract_json_array(content)
            if subqueries:
                logger.info(f"Decomposed original query into {len(subqueries)} subqueries: {subqueries}")
                return subqueries
            else:
                logger.warning(f"Attempt {attempt + 1}/3: LLM returned empty or invalid JSON array.")

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/3 failed for Query Decomposition: {e}")

    logger.error("All 3 attempts for Query Decomposition failed. Returning original query as a fallback.")
    return [question]


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    test_query = "Formulate the requirements for the composition of a starter culture for curd aimed at suppressing Helicobacter pylori. What types of microorganisms should be included in this consortium to effectively inhibit the urease of the pathogen and synthesize bacteriocins?"

    print("Testing Stage 1: Semantic Decomposition")
    print(f"Original query: {test_query}\n")

    async def main():
        config = load_config()

        start_time = time.time()
        subqueries = await decompose_query(test_query, config)
        elapsed = time.time() - start_time

        print(f"--- STAGE 1 (Completed in {elapsed:.2f}s) ---")
        print("Resulting Subqueries:")
        print(json.dumps(subqueries, indent=2, ensure_ascii=False))

    asyncio.run(main())
