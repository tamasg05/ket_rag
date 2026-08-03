"""Text, KNNG, and KET-RAG retrieval strategies."""

from __future__ import annotations

import math

import numpy as np

from .graph_construction import format_relationship_text, top_indices


SKELETON_ENTITY_SEED_COUNT = 10


def _source_suffix(item: dict) -> str:
    """
    Format optional source, page, and table provenance for a context label.

    Inputs:
        item: Chunk or subchunk dictionary.

    Returns:
        Empty text or a comma-prefixed provenance suffix.
    """
    values: list[str] = []
    if item.get("source_name"):
        values.append(f"source {item['source_name']}")
    if item.get("page") is not None:
        values.append(f"page {item['page']}")
    if item.get("table_id"):
        values.append(f"table {item['table_id']}")
    if item.get("row_start") is not None:
        row_start = item["row_start"]
        row_end = item.get("row_end", row_start)
        values.append(f"rows {row_start}-{row_end}")
    return ", " + ", ".join(values) if values else ""


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
    return "\n\n".join(
        f"[{label} {i}{_source_suffix(chunks[i])}]\n"
        f"{chunks[i].get('source_text', chunks[i]['text'])}"
        for i in ids
    )


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
            f"[chunk {subchunk['parent_id']}, subchunk {subchunk['id']}"
            f"{_source_suffix(subchunk)}]\n"
            f"{subchunk.get('source_text', subchunk['text'])}"
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
    Skeleton retrieval follows KG-Retrieval's entity -> relationship -> text
    order. Relationship and text adjacency are the primary ranking signals;
    cosine similarity breaks ties because this demo uses counts, not tokens.

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
    entity_ids: list[int] = []
    skeleton_facts: list[str] = []
    relation_facts: list[str] = []
    selected_relationships: list[dict] = []
    if skeleton_budget and len(index["entities"]):
        # Algorithm 2 line 1: retrieve up to ten semantic entity seeds.
        entity_ids = top_indices(
            arrays["entity_vectors"] @ query,
            min(SKELETON_ENTITY_SEED_COUNT, len(index["entities"])),
        )
        parent_to_subs: dict[int, list[int]] = {}
        for sub in subchunks:
            parent_to_subs.setdefault(sub["parent_id"], []).append(sub["id"])

        # Count direct chunk-to-entity connections. These counts implement the
        # structural adjacency signal used by Algorithm 2 line 3.
        parent_adjacency: dict[int, int] = {}
        for entity_id in entity_ids:
            entity = index["entities"][entity_id]
            description = " ".join(dict.fromkeys(entity["descriptions"]))
            skeleton_facts.append(f"Entity: {entity['name']} — {description}")
            for parent_id in set(entity["chunk_ids"]):
                parent_adjacency[parent_id] = parent_adjacency.get(parent_id, 0) + 1

        # Algorithm 2 line 2: relationships touching two entity seeds have
        # higher adjacency than relationships touching only one. The paper
        # does not prescribe tie handling, so relationship-vector similarity
        # to the query provides a stable semantic tie-breaker.
        seed_names = {index["entities"][i]["name"].casefold() for i in entity_ids}
        relations = index["relations"]
        relation_vectors = arrays.get("relation_vectors")
        if len(relations) and (
            relation_vectors is None or len(relation_vectors) != len(relations)
        ):
            raise ValueError(
                "The KET index has no aligned relationship embeddings; "
                "load/build the index again to upgrade it."
            )
        ranked_relationships: list[tuple[int, int, float]] = []
        for relation_id, relation in enumerate(relations):
            source = str(relation.get("source", ""))
            target = str(relation.get("target", ""))
            adjacency_score = int(source.casefold() in seed_names) + int(
                target.casefold() in seed_names
            )
            if not adjacency_score:
                continue
            semantic_score = float(relation_vectors[relation_id] @ query)
            ranked_relationships.append(
                (relation_id, adjacency_score, semantic_score)
            )
        ranked_relationships.sort(
            key=lambda item: (item[1], item[2]),
            reverse=True,
        )

        # The paper stops by token length. This count-based demonstration keeps
        # a small relationship set proportional to the skeleton text budget.
        relationship_limit = max(1, skeleton_budget * 3)
        for relation_id, adjacency_score, semantic_score in ranked_relationships[
            :relationship_limit
        ]:
            relation = relations[relation_id]
            relation_facts.append(
                f"Relation: {format_relationship_text(relation)}"
            )
            parent_id = int(relation.get("chunk_id", -1))
            parent_adjacency[parent_id] = parent_adjacency.get(parent_id, 0) + 1
            selected_relationships.append(
                {
                    "relation_id": relation_id,
                    "source": str(relation.get("source", "")),
                    "target": str(relation.get("target", "")),
                    "adjacency_to_entity_seeds": adjacency_score,
                    "query_similarity": semantic_score,
                }
            )

        subchunk_adjacency: dict[int, int] = {}
        for parent_id, adjacency_score in parent_adjacency.items():
            for subchunk_id in parent_to_subs.get(parent_id, []):
                subchunk_adjacency[subchunk_id] = adjacency_score

        # Algorithm 2 line 3: adjacency to selected entities and relationships
        # leads the ranking; semantic similarity resolves equal adjacency.
        skeleton_ids = sorted(
            subchunk_adjacency,
            key=lambda i: (
                subchunk_adjacency[i],
                float(arrays["subchunk_vectors"][i] @ query),
            ),
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
            + "\n".join(skeleton_facts + relation_facts)
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
                **{
                    key: subchunks[subchunk_id][key]
                    for key in (
                        "source_name",
                        "source_url",
                        "page",
                        "table_id",
                        "row_start",
                        "row_end",
                    )
                    if subchunks[subchunk_id].get(key) not in (None, "")
                },
            }
            for subchunk_id in selected
        ],
        "skeleton_entity_seeds": [index["entities"][i]["name"] for i in entity_ids],
        "skeleton_relation_facts": relation_facts,
        "skeleton_relationships": selected_relationships,
        "keyword_seeds": [index["keywords"][i] for i in keyword_ids],
    }
    return context, details
