"""Text, KNNG, and KET-RAG retrieval strategies."""

from __future__ import annotations

import math

import numpy as np

from .graph_construction import top_indices


def serialise_chunks(chunks: list[dict], ids: list[int], label: str = "chunk") -> str:
    """
    Format selected chunks as labelled text for the answer-generation prompt.

    Inputs:
        chunks: All available chunk dictionaries.
        ids: Indices of the chunks to include, in retrieval order.
        label: Text placed before each chunk ID.

    Returns:
        One string containing the labelled selected chunks.
    """
    return "\n\n".join(f"[{label} {i}]\n{chunks[i]['text']}" for i in ids)


def serialise_subchunks(subchunks: list[dict], ids: list[int]) -> str:
    """
    Format KET evidence with both original chunk and subchunk IDs.

    Inputs:
        subchunks: All available KET subchunk dictionaries.
        ids: Indices of the subchunks to include, in retrieval order.

    Returns:
        One string containing the labelled selected subchunks.
    """
    parts: list[str] = []
    for subchunk_id in ids:
        subchunk = subchunks[subchunk_id]
        parts.append(
            f"[chunk {subchunk['parent_id']}, subchunk {subchunk['id']}]\n"
            f"{subchunk['text']}"
        )
    return "\n\n".join(parts)


def text_retrieve(
    chunks: list[dict],
    vectors: np.ndarray,
    query: np.ndarray,
    top_k: int,
) -> tuple[str, list[int]]:
    """
    Retrieve the globally most semantically similar chunks for Text RAG.

    Inputs:
        chunks: All chunk dictionaries.
        vectors: Normalized embedding matrix, one row per chunk.
        query: Normalized embedding vector of the user's query.
        top_k: Maximum number of chunks to retrieve.

    Returns:
        A pair containing the formatted context string and retrieved chunk IDs.
    """
    ids = top_indices(vectors @ query, top_k)
    return serialise_chunks(chunks, ids), ids


def knn_retrieve(
    chunks: list[dict],
    vectors: np.ndarray,
    adjacency: list[list[int]],
    query: np.ndarray,
    top_k: int,
) -> tuple[str, list[int]]:
    """
    Retrieve semantic seeds and their hybrid-KNN neighbours for KNNG-RAG.

    Inputs:
        chunks: All chunk dictionaries.
        vectors: Normalized embedding matrix, one row per chunk.
        adjacency: KNN adjacency list; one neighbour-ID list per chunk.
        query: Normalized embedding vector of the user's query.
        top_k: Maximum number of chunks in the final context.

    Returns:
        A pair containing the formatted context string and retrieved chunk IDs.
    """
    # Start from semantic seeds, expand their one-hop neighbourhood, then rerank.
    seed_count = max(1, min(top_k, math.ceil(top_k / 2)))
    seeds = top_indices(vectors @ query, seed_count)
    candidates = set(seeds)
    for seed in seeds:
        candidates.update(adjacency[seed])
    ordered = sorted(candidates, key=lambda i: float(vectors[i] @ query), reverse=True)[:top_k]
    return serialise_chunks(chunks, ordered), ordered


def ket_retrieve(
    index: dict,
    query: np.ndarray,
    top_k: int,
    theta: float,
) -> tuple[str, dict]:
    """
    Retrieve KET context using a demonstration adaptation of Algorithm 4.

    Unlike the paper's token budget lambda, this prototype divides a top-k
    subchunk count between the skeleton and keyword channels using theta.
    Skeleton retrieval is also a simplified counterpart of KG-Retrieval.

    Inputs:
        index: Loaded KET metadata, graphs, mappings, and embedding arrays.
        query: Normalized embedding vector of the user's query.
        top_k: Maximum number of subchunks in the final text context.
        theta: Share of top_k assigned to the skeleton retrieval channel.

    Returns:
        A pair containing the formatted KET context and retrieval diagnostics.
    """
    arrays = index["arrays"]
    skeleton_budget = min(top_k, max(0, round(theta * top_k)))
    keyword_budget = top_k - skeleton_budget
    subchunks = index["subchunks"]

    # Algorithm 4 line 1, adapted: retrieve from the skeleton channel.
    skeleton_candidates: set[int] = set()
    entity_ids: list[int] = []
    skeleton_facts: list[str] = []
    relation_facts: list[str] = []
    if skeleton_budget and len(index["entities"]):
        entity_ids = top_indices(arrays["entity_vectors"] @ query, max(3, skeleton_budget))
        parent_to_subs: dict[int, list[int]] = {}
        for sub in subchunks:
            parent_to_subs.setdefault(sub["parent_id"], []).append(sub["id"])
        for entity_id in entity_ids:
            entity = index["entities"][entity_id]
            description = " ".join(dict.fromkeys(entity["descriptions"]))
            skeleton_facts.append(f"Entity: {entity['name']} — {description}")
            for parent_id in entity["chunk_ids"]:
                skeleton_candidates.update(parent_to_subs.get(parent_id, []))
        seed_names = {index["entities"][i]["name"].casefold() for i in entity_ids}
        for relation in index["relations"]:
            source = str(relation.get("source", ""))
            target = str(relation.get("target", ""))
            if source.casefold() in seed_names or target.casefold() in seed_names:
                relation_name = relation.get("relation", "related to")
                description = relation.get("description", "")
                relation_facts.append(
                    f"Relation: {source} —[{relation_name}]→ {target}. {description}".strip()
                )
                skeleton_candidates.update(
                    parent_to_subs.get(int(relation.get("chunk_id", -1)), [])
                )
        skeleton_ids = sorted(
            skeleton_candidates,
            key=lambda i: float(arrays["subchunk_vectors"][i] @ query),
            reverse=True,
        )[:skeleton_budget]
    else:
        skeleton_ids = []

    # Algorithm 4 lines 2-4, adapted: collect about twice the keyword-channel
    # chunk budget from seed neighbourhoods, then semantically rerank.
    keyword_ids: list[int] = []
    candidates: set[int] = set()
    if keyword_budget and len(index["keywords"]):
        ranked_keywords = top_indices(
            arrays["keyword_vectors"] @ query,
            min(len(index["keywords"]), max(20, keyword_budget * 4)),
        )
        for keyword_id in ranked_keywords:
            keyword_ids.append(keyword_id)
            candidates.update(index["keyword_to_subchunks"][keyword_id])
            if len(candidates) >= 2 * keyword_budget:
                break
    candidates.difference_update(skeleton_ids)
    keyword_chunk_ids = sorted(
        candidates,
        key=lambda i: float(arrays["subchunk_vectors"][i] @ query),
        reverse=True,
    )[:keyword_budget]

    selected = skeleton_ids + keyword_chunk_ids
    graph_context = ""
    if skeleton_facts or relation_facts:
        graph_context = (
            "[knowledge graph skeleton]\n"
            + "\n".join(skeleton_facts + relation_facts[: max(6, skeleton_budget * 3)])
            + "\n\n"
        )
    # Algorithm 4 line 5: combine skeleton and keyword contexts.
    context = graph_context + serialise_subchunks(subchunks, selected)
    details = {
        "subchunk_ids": selected,
        "retrieved_sources": [
            {
                "chunk_id": subchunks[subchunk_id]["parent_id"],
                "subchunk_id": subchunks[subchunk_id]["id"],
            }
            for subchunk_id in selected
        ],
        "skeleton_entity_seeds": [index["entities"][i]["name"] for i in entity_ids],
        "skeleton_relation_facts": relation_facts,
        "keyword_seeds": [index["keywords"][i] for i in keyword_ids],
    }
    return context, details
