import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)

def format_label(labels: list[str]) -> str:
    allowed = {'Metabolite', 'Microbe', 'StarterCulture', 'EnvironmentCondition'}
    found = [l for l in labels if l in allowed]
    l = found[0] if found else (labels[0] if labels else "Entity")

    if l == 'StarterCulture':
        return 'StarterCulture'
    elif l == 'EnvironmentCondition':
        return 'EnvironmentCondition'
    return l


def _node_header(label: str, name: str) -> str:
    return f"{label}: {name}"


def serialize_path(path_obj: dict) -> str:
    """
    Serialize a path into structured hop blocks with evidence + source_file per edge.

    Example:
        Microbe: Lactococcus lactis
          -[:PRODUCES]-> Metabolite: diacetyl
          evidence: LAB produce diacetyl...
          source_file: PMC123.pdf
    """
    nodes = path_obj.get("nodes", [])
    rels = path_obj.get("relationships", [])  # list of lists of relationships

    if not nodes or not rels:
        return ""

    lines: list[str] = []

    for i, hop_rels in enumerate(rels):
        if i + 1 >= len(nodes):
            break

        source_node = nodes[i]
        target_node = nodes[i + 1]

        s_label = format_label(source_node.get("labels", []))
        t_label = format_label(target_node.get("labels", []))
        s_name = source_node.get("name", "Unknown")
        t_name = target_node.get("name", "Unknown")

        if i == 0:
            lines.append(_node_header(s_label, s_name))

        seen: set[tuple[str, str]] = set()
        emitted = False

        for rel in hop_rels:
            ev = (rel.get("evidence") or "").strip()
            src = (rel.get("source_file") or "").strip()
            key = (ev.lower(), src.lower())
            if key in seen:
                continue
            seen.add(key)

            r_type = rel.get("type", "RELATES_TO")
            lines.append(f"  -[:{r_type}]-> {_node_header(t_label, t_name)}")
            lines.append(f"  evidence: {ev}" if ev else "  evidence:")
            lines.append(f"  source_file: {src}" if src else "  source_file:")
            emitted = True

        if not emitted:
            lines.append(f"  -[:RELATES_TO]-> {_node_header(t_label, t_name)}")
            lines.append("  evidence:")
            lines.append("  source_file:")

    return "\n".join(lines).strip()


async def rerank_paths(original_query: str, paths: list[dict], batch_size: int = 4) -> list[dict]:
    """
    Stage 5: Serialization and Decoder Reranking.
    Serializes the topology paths and calls the local Qwen-0.6B Reranker.
    Adds 'semantic_score' and 'serialized_text' to each path.
    """
    if not paths:
        return []

    logger.info(f"Serializing {len(paths)} paths for reranking...")

    for p in paths:
        p["serialized_text"] = serialize_path(p)

    texts = [p["serialized_text"] for p in paths]

    url = "http://127.0.0.1:7997/rerank"

    # We batch requests to not overload the reranker socket or memory
    sem = asyncio.Semaphore(8)

    async def fetch_rerank(chunk_texts, start_idx):
        async with sem:
            async with httpx.AsyncClient() as client:
                try:
                    payload = {"query": original_query, "texts": chunk_texts}
                    resp = await client.post(url, json=payload, timeout=60.0)
                    resp.raise_for_status()
                    return resp.json(), start_idx
                except Exception as e:
                    logger.error(f"Reranker failed for batch starting at {start_idx}: {e}")
                    return [{"index": i, "score": 0.0} for i in range(len(chunk_texts))], start_idx

    tasks = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        tasks.append(fetch_rerank(chunk, i))

    logger.info(f"Sending {len(tasks)} batches to the reranker...")
    results = await asyncio.gather(*tasks)

    # Reassemble scores
    # Results is a list of tuples: (reranker_response_list, start_idx)
    # The response is [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.2}] relative to the chunk
    for chunk_res, start_idx in results:
        for item in chunk_res:
            relative_idx = item.get("index", 0)
            score = item.get("score", 0.0)
            absolute_idx = start_idx + relative_idx
            if absolute_idx < len(paths):
                paths[absolute_idx]["semantic_score"] = score

    logger.info("Reranking completed successfully.")
    return paths
