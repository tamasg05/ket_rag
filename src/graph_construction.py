"""Pure graph-construction helpers for KNNG-RAG and KET-RAG."""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .data_extraction import tokenize_words


def top_indices(scores: np.ndarray, count: int) -> list[int]:
    """
    Find the indices of the largest scores in descending order.

    Inputs:
        scores: One-dimensional array of numeric scores.
        count: Maximum number of indices to select.

    Returns:
        Selected integer indices, ordered from highest to lowest score.
    """
    count = max(0, min(int(count), len(scores)))
    if count == 0:
        return []
    selected = np.argpartition(-scores, count - 1)[:count]
    return selected[np.argsort(-scores[selected])].astype(int).tolist()


def build_hybrid_knn(texts: list[str], embeddings: np.ndarray, k: int) -> list[list[int]]:
    """
    For each chunk, select floor(k/2) lexical neighbours, then the remaining
    semantic neighbours.

    This is the KNN-graph initialization subroutine in lines 2-7 of KET-RAG
    Algorithm 3. It is not the complete KET-Index algorithm. The resulting
    graph is the retrieval index for the prototype's KNNG-RAG baseline, while
    KET-RAG uses it only as an intermediate structure for PageRank-based
    core-chunk selection.

    Inputs:
        texts: Chunk texts, one for every graph node.
        embeddings: Normalized chunk embedding matrix in the same order.
        k: Requested outgoing neighbour count per chunk.

    Returns:
        An adjacency list whose row ``i`` contains chunk ``i``'s lexical and
        semantic outgoing neighbour IDs.
    """
    n = len(texts)
    if n < 2:
        return [[] for _ in texts]
    k = max(2, min(int(k), n - 1))
    lexical_count = k // 2
    semantic_count = k - lexical_count

    binary_words = CountVectorizer(stop_words="english", binary=True).fit_transform(texts)
    co_occurrence = (binary_words @ binary_words.T).toarray()
    semantic = embeddings @ embeddings.T
    np.fill_diagonal(co_occurrence, -1)
    np.fill_diagonal(semantic, -np.inf)

    graph: list[list[int]] = []
    for i in range(n):
        lexical = top_indices(co_occurrence[i], lexical_count)
        semantic_scores = semantic[i].copy()
        semantic_scores[lexical] = -np.inf
        semantic_neighbours = top_indices(semantic_scores, semantic_count)
        graph.append(lexical + semantic_neighbours)
    return graph


def pagerank_scores(adjacency: list[list[int]]) -> dict[int, float]:
    """
    Calculate the structural-importance scores used by Algorithm 3 line 8.

    Inputs:
        adjacency: Directed adjacency list; each row contains outgoing node IDs.

    Returns:
        A mapping from every node ID to its PageRank score.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(adjacency)))
    for source, targets in enumerate(adjacency):
        graph.add_edges_from((source, target) for target in targets)
    return nx.pagerank(graph, alpha=0.85)


def select_core_chunks(adjacency: list[list[int]], beta: float) -> list[int]:
    """
    Select the PageRank-leading beta fraction from Algorithm 3 line 8.

    Inputs:
        adjacency: Directed KNN adjacency list.
        beta: Fraction of all chunks to retain as skeleton core chunks.

    Returns:
        Core chunk IDs in descending PageRank order.
    """
    if not adjacency:
        return []
    ranks = pagerank_scores(adjacency)
    core_count = max(1, math.ceil(float(beta) * len(adjacency)))
    return sorted(ranks, key=ranks.get, reverse=True)[:core_count]


def build_skeleton_graph(
    core_ids: list[int], extraction_records: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """
    Consolidate LLM extractions into the prototype's skeleton graph.

    This is the demonstration's simplified counterpart of KG-Index invoked by
    KET-RAG Algorithm 3 line 9. Entity names are merged case-insensitively;
    relations remain records associated with their source chunks.

    Inputs:
        core_ids: IDs of the chunks selected for skeleton extraction.
        extraction_records: LLM extraction records keyed by string chunk ID.

    Returns:
        A pair containing the merged entity records and relation records.
    """
    entities: dict[str, dict] = {}
    relations: list[dict] = []
    for chunk_id in core_ids:
        record = extraction_records[str(chunk_id)]
        for entity in record.get("entities", []):
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            key = name.casefold()
            item = entities.setdefault(
                key, {"name": name, "descriptions": [], "chunk_ids": []}
            )
            description = str(entity.get("description", "")).strip()
            if description:
                item["descriptions"].append(description)
            item["chunk_ids"].append(chunk_id)
        for relation in record.get("relations", []):
            relation_with_source = dict(relation)
            relation_with_source["chunk_id"] = chunk_id
            relations.append(relation_with_source)
    return list(entities.values()), relations


def format_relationship_text(relation: dict) -> str:
    """
    Create the textual relationship description used for embedding and prompts.

    Inputs:
        relation: Relationship record containing source, target, relation type,
            and an optional description.

    Returns:
        One readable string describing the relationship and its evidence.
    """
    source = str(relation.get("source", "")).strip()
    target = str(relation.get("target", "")).strip()
    relation_name = str(relation.get("relation", "related to")).strip()
    description = str(relation.get("description", "")).strip()
    statement = f"{source} --[{relation_name or 'related to'}]--> {target}".strip()
    return f"{statement}. {description}".strip() if description else statement


def _keyword_tokens(text: str) -> list[str]:
    """
    Tokenize case-insensitively and remove English stopwords.

    Inputs:
        text: Text from which keyword candidates are required.

    Returns:
        Lower-cased non-stopword tokens; inflected forms remain separate.
    """
    return [
        token.casefold()
        for token in tokenize_words(text)
        if token.casefold() not in ENGLISH_STOP_WORDS
    ]


def build_keyword_vocabulary(texts: list[str]) -> list[str]:
    """
    Create the Algorithm 3 line-1 vocabulary with one demo-specific filter.

    The paper permits every tokenized non-stopword. This prototype retains a
    term only when it occurs in at least two different subchunks.

    Inputs:
        texts: Subchunk texts from which to collect keyword candidates.

    Returns:
        Alphabetically sorted keywords occurring in at least two subchunks.
    """
    document_frequency: Counter[str] = Counter()
    for text in texts:
        # A repeated word in one subchunk counts only once toward min_df=2.
        document_frequency.update(set(_keyword_tokens(text)))
    return sorted(
        token for token, frequency in document_frequency.items() if frequency >= 2
    )


def build_keyword_subchunk_graph(
    subchunks: list[dict],
    sentences: list[str],
    sentence_vectors: np.ndarray,
    embedding_dimensions: int,
) -> tuple[list[str], list[list[int]], np.ndarray]:
    """
    Build the keyword-subchunk bipartite graph from Algorithm 3 lines 11-15.

    Subchunk creation (line 10) and paid embedding calls remain outside this
    pure helper. Each keyword vector is the normalized average embedding of
    sentences containing that keyword, as specified in line 13.

    Inputs:
        subchunks: Fine-grained subchunk dictionaries.
        sentences: Corpus sentences used to describe keyword context.
        sentence_vectors: Sentence embeddings in the same order as
            ``sentences``.
        embedding_dimensions: Width of each embedding vector.

    Returns:
        A tuple containing the keyword list, keyword-to-subchunk adjacency
        lists, and normalized keyword embedding matrix.
    """
    keywords = build_keyword_vocabulary([subchunk["text"] for subchunk in subchunks])
    if not keywords:
        raise ValueError("No keyword occurs in at least two subchunks")

    vectorizer = CountVectorizer(
        analyzer=_keyword_tokens,
        binary=True,
        vocabulary=keywords,
    )
    membership = vectorizer.fit_transform(
        [subchunk["text"] for subchunk in subchunks]
    ).tocsc()
    sentence_words = vectorizer.transform(sentences).tocsc()

    keyword_to_subchunks: list[list[int]] = []
    keyword_vectors: list[np.ndarray] = []
    fallback = np.zeros(embedding_dimensions, dtype=np.float32)
    for column in range(len(keywords)):
        sub_ids = membership.indices[
            membership.indptr[column] : membership.indptr[column + 1]
        ]
        keyword_to_subchunks.append(sub_ids.astype(int).tolist())
        sentence_ids = sentence_words.indices[
            sentence_words.indptr[column] : sentence_words.indptr[column + 1]
        ]
        average = (
            sentence_vectors[sentence_ids].mean(axis=0)
            if len(sentence_ids)
            else fallback
        )
        norm = np.linalg.norm(average)
        keyword_vectors.append(average / max(norm, 1e-12))

    return (
        keywords,
        keyword_to_subchunks,
        np.asarray(keyword_vectors, dtype=np.float32),
    )
