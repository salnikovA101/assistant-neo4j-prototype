import asyncio
import logging
import httpx

logger = logging.getLogger(__name__)

def format_label(labels: list[str]) -> str:
    allowed = {'Metabolite', 'Microbe', 'StarterCulture', 'EnvironmentCondition'}
    found = [l for l in labels if l in allowed]
    l = found[0] if found else (labels[0] if labels else "Entity")

    if l == 'StarterCulture':
        return 'starter culture'
    elif l == 'EnvironmentCondition':
        return 'environment condition'
    return l.lower()

def serialize_path(path_obj: dict) -> str:
    """
    Serializes a single path object into a string for LLM reranking.
    Expected output for single edge:
    'microbe name relates to metabolite name: evidence text, next metabolite name ...'
    Expected output for multiple edges:
    'microbe name [produces: evidence 1 | consumes: evidence 2] metabolite name'
    """
    nodes = path_obj.get("nodes", [])
    rels = path_obj.get("relationships", []) # Now this is a list of lists of relationships

    if not nodes or not rels:
        return ""

    parts = []

    for i, hop_rels in enumerate(rels):
        source_node = nodes[i]
        target_node = nodes[i+1]

        # Get labels safely and format them
        s_labels = source_node.get("labels", [])
        t_labels = target_node.get("labels", [])

        s_label = format_label(s_labels)
        t_label = format_label(t_labels)

        s_name = source_node.get("name", "Unknown")
        t_name = target_node.get("name", "Unknown")

        unique_edges = []
        seen_evidence = set()

        for rel in hop_rels:
            ev = rel.get("evidence", "").strip()
            ev_lower = ev.lower()

            # Deduplicate by evidence text (if evidence exists)
            if not ev or ev_lower not in seen_evidence:
                if ev:
                    seen_evidence.add(ev_lower)

                r_type = rel.get("type", "RELATES_TO").lower().replace("_", " ")
                if ev:
                    unique_edges.append(f"{r_type} {ev}")
                else:
                    unique_edges.append(f"{r_type}")

        if unique_edges:
            edge_str = " also ".join(unique_edges)
            hop_str = f"{s_label} {s_name} {edge_str} {t_label} {t_name}"
        else:
            hop_str = f"{s_label} {s_name} relates to {t_label} {t_name}"

        if i == 0:
            parts.append(hop_str)
        else:
            parts.append(f"next {hop_str}")

    return ", ".join(parts).strip()


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
