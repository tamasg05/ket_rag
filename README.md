# Comparing Three RAG Approaches: A Python Prototype

This annotated Python prototype compares:

1. **Text RAG** — global semantic search.
2. **KNNG-RAG** — semantic seeds followed by one-hop expansion in a hybrid
   chunk graph (`k/2` lexical neighbours and `k/2`
   semantic neighbours).
3. **KET-RAG** — A simplified implementation of Huang, Zhang, and Xiao’s approach, combining an entity/relation skeleton with a keyword–subchunk bipartite graph. See BibTeX citation below.

The corpus is the text file in `data/`. Index artifacts are stored under
`.rag_cache/` and reused whenever the corpus, model, chunking, and graph
parameters match.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Keep this entry in `.env`:

```text
GOOGLE_API_KEY=your-key
```

Optional model/index settings are `EMBEDDING_MODEL`, `GENERATION_MODEL`,
`EXTRACTION_MODEL`, `EMBEDDING_DIMENSIONS`, `CHUNK_WORDS`, `CHUNK_OVERLAP`,
`EMBED_BATCH_SIZE`, `EMBED_BATCH_DELAY`, and `EXTRACTION_BATCH_SIZE`.

The defaults are `gemini-embedding-001` at 768 dimensions and
`gemini-3.1-flash-lite` for inexpensive graph extraction and answering.

## Run

In project root:
```powershell
python -m src.gradio_app
```

After the server starts, put `http://127.0.0.1:7860` into your browser's
address bar and open it. Choose the graph parameters and click **Build/load
indexes**. The first build makes paid Gemini calls; later builds with the same
inputs load local artifacts. Changing the corpus, tokenization, or indexing
settings requires a new persistent index and new API calls. Then enter a
question and click **Compare answers**.

The first KET-RAG build can take several minutes because it extracts the
skeleton graph and then embeds entities, relationships, subchunks, and
sentences. Progress is shown in the UI. Extraction is checkpointed after every
batch, and embeddings are checkpointed periodically, so a stopped build can
resume when the same corpus, models, and graph parameters are selected.
Existing indexes created before relationship embeddings were added are
upgraded in place: only the missing relationship embeddings are created, and
graph extraction is not repeated. Gemini is asked for schema-constrained JSON.
If a multi-chunk response is malformed, truncated, or incomplete, the builder
automatically retries those chunks one at a time and checkpoints every
successful result. This fallback is slower, but it prevents one oversized
response from aborting the complete build.

The optional CLI builder is:

```powershell
python -m src.build_indexes --knn-k 6 --ket-k 6 --beta 0.2 --tau 1
```

Run the small offline test suite with:

```powershell
python -m unittest discover -s src/tests -v
```

## Parameters

- `top-k`: final context count used by all three retrievers.
- `temperature`: generation sampling temperature.
- KNNG `k`: hybrid chunk-graph out-degree.
- KET `k`: intermediate hybrid graph out-degree used for PageRank.
- KET `beta`: fraction of chunks sent to graph extraction.
- KET `tau`: each chunk is recursively split into `2**tau` sub-chunks.
- KET `theta`: divides retrieval between the knowledge-graph skeleton (`theta`)
  and keyword channel (`1 - theta`). Thus, `0` is keyword-only, `1` is
  skeleton-only, and the default `0.4` assigns roughly 40% to the skeleton.
- Skeleton extraction is limited to at most 20 entities and 30 relationships
  per core chunk. The limits are stated in the model prompt and enforced by the
  Python code to bound graph size, extraction time, and response size.

## Prototype scope

This is a readable local-search demonstration, not a production GraphRAG
system. The prototype uses chunk counts instead of the paper’s token budget and 
simplifies knowledge-graph retrieval. It is a demonstration-oriented adaptation 
of Algorithms 3 and 4 of Huang, Zhang, and Xiao (see BibTeX citation below).

## In line with the paper

The prototype implements the central elements of KET-RAG:

- A hybrid KNN graph with lexical and semantic neighbours.
- PageRank-based selection of a `beta` fraction of core chunks.
- LLM-based entity and relationship extraction only from the core chunks.
- Descriptions and embeddings for both entities and relationships.
- Skeleton retrieval ordered by entity, relationship, and supporting text
  adjacency, following Algorithm 2.
- Recursive splitting of each chunk into `2**tau` subchunks.
- A keyword–subchunk bipartite graph.
- Keyword vectors based on the average embeddings of relevant sentences.
- Skeleton and keyword retrieval channels divided by `theta`.

## Differences compared to the paper

| Area | Paper | Prototype | Likely consequence |
|---|---|---|---|
| Retrieval budget | Uses a token limit, `lambda`; the experimental default is 12,000 tokens. | Uses the number of chunks specified by `top-k`. | Long and short chunks count equally, so context size is less predictable and may contain less evidence. |
| Skeleton graph | Entities are nodes and relationships are graph edges; both have descriptions and embeddings. | Entity records and relationship-edge records both have descriptions and embeddings, but they are persisted as records, NumPy arrays, and source-chunk IDs rather than a fully navigable graph object. | Relationship semantics are retained, but explicit multi-hop graph traversal remains simpler than in a production graph index. |
| Extraction limits | The paper does not specify fixed per-chunk entity and relationship limits. | Each core chunk is limited to at most 20 entities and 30 relationships. | Graph size and extraction cost are bounded, but dense chunks can lose lower-priority facts. |
| Skeleton retrieval | Selects ten query-relevant entity seeds. Relationships are ranked by adjacency to those seeds, followed by adjacent supporting chunks, until the token budgets are filled. | Follows the same entity → relationship → text order. A relationship touching two entity seeds ranks above one touching a single seed. Supporting subchunks rank by their number of connections to the selected entities and relationships. If relationships or subchunks have equal adjacency scores, cosine similarity to the query breaks the tie. Fixed item counts replace token budgets. | Adjacency remains the primary ranking signal, but semantic tie-breaking and count limits can produce different context from the paper. |
| Graph rewiring | Skeleton edges are rewired from the original chunks to the appropriate fine-grained subchunks. | Entities and relationships retain parent chunk IDs, which are expanded to their child subchunks. | An entity can become associated with every subchunk of its parent, introducing irrelevant context. |
| Keyword vocabulary | All tokenized non-stopwords can become keywords. | All tokenized non-stopwords can become keywords, but a keyword must occur in at least two different subchunks. | Terms that occur in only one subchunk are absent from the keyword graph, which can affect questions about rare entities. |
| Sentence processing | Uses NLTK sentence tokenization in the experiments. | Uses NLTK Punkt configured for common English titles and abbreviations, and retains only sentence fragments containing at least three tokenized words. | Uncommon abbreviations may still produce inaccurate boundaries, and shorter fragments do not contribute to keyword vectors. |
| Initial chunks | Uses token-based chunks in the experiments. | Uses 450 word tokens with a 60-word overlap by default. Surrounding punctuation is removed from the word tokens, so `Hello, World!` becomes `Hello` and `World`; case and internal apostrophes are preserved. | The tokenizer, repeated content, and different chunk boundaries can change co-occurrence, PageRank, sentence averages, and retrieval rankings. |
| Models | Uses GPT-4o-mini and `text-embedding-3-small` in the experiments. | Uses `gemini-3.1-flash-lite` and `gemini-embedding-001` by default. | Retrieval, extraction, multilingual behaviour, cost, and answer style may differ from the paper's results. |
| Evaluation | Evaluates benchmark datasets containing thousands of paragraphs and 500 questions per dataset. | Uses one Sherlock Holmes book and manual demonstration queries. | The paper's reported accuracy and cost improvements cannot be attributed directly to this prototype. |

## Default parameter differences

| Parameter | Paper | Prototype default | Practical effect |
|---|---|---|---|
| KNN graph `K` | `2` | `6` | The prototype creates a denser intermediate graph, which can change PageRank scores and core-chunk selection. |
| Skeleton budget `beta` | `0.8` | `0.2` | The prototype sends 20% rather than 80% of chunks to entity and relationship extraction, reducing cost but also skeleton coverage. |
| Per-chunk extraction ceiling | No fixed ceiling is specified | 20 entities and 30 relationships | The ceiling keeps the demonstration manageable but can omit less important graph facts from dense chunks. |
| Retrieval balance `theta` | `0.4` | `0.4` | Both target 40% of the retrieval budget for the skeleton channel and 60% for the keyword channel. |
| Initial chunk size | 1,200 tokens in the low-cost configuration; 150 tokens in the high-accuracy configuration | 450 words with a 60-word overlap | Chunk boundaries and sizes differ substantially, affecting graph connections and retrieval granularity. |
| Number of splits `tau` | `3` with 1,200-token chunks; `0` with 150-token chunks | `1` | The paper's two configurations both produce subchunks of approximately 150 tokens, while the prototype divides each chunk into two approximately equal parts. |
| Retrieval limit | `lambda = 12,000` tokens | `top-k = 6` chunks or subchunks | The paper limits context by token length, while the prototype limits the number of selected text units. |

## Open Points and Further Work

1. **Direction of KNN neighbour expansion**

   Algorithm 3 writes each KNN edge as `(v_i, v_j)` after `v_i` selects its
   top-`K` lexical and semantic neighbours. Read literally, this describes a
   directed edge from `v_i` to `v_j`. However, the paper's experimental
   description says that KNNG-RAG includes the seed nodes' "neighbors" without
   specifying whether this means outgoing neighbours, incoming neighbours, or
   an undirected version of the graph.

   The prototype follows the literal directed interpretation during KNNG-RAG
   retrieval: it adds only the outgoing one-hop neighbours stored in
   `adjacency[seed]`. Nodes that point to the seed are not added. Incoming edges
   still affect PageRank and therefore influence KET-RAG's core-chunk selection.

   Further work could compare three retrieval variants:

   - **Outgoing only (current):** add edges from the seed to its selected
     neighbours.
   - **Incoming and outgoing:** symmetrize the graph during retrieval to improve
     recall, potentially at the cost of a larger and noisier candidate set.
   - **Mutual only:** include a neighbour only when both nodes select each other,
     which may improve precision but reduce recall.

   The variants should be evaluated using representative questions, retrieval
   accuracy, candidate-set size, and retrieval time before changing the default.

2. **Keyword lemmatization**

   The paper does not mention lemmatization, and the current prototype does not
   use it. The keyword vocabulary is case-insensitive, but inflected forms such
   as `investigate`, `investigated`, and `investigating` are separate keywords,
   and each form must occur in at least two subchunks to be retained. A future
   version could lemmatize non-stopword tokens before calculating their
   document frequency and building keyword-to-subchunk edges. This could
   improve keyword coverage, but it would be an extension beyond the method
   explicitly documented in the paper. The lemmatizer must also match the
   corpus language and should preserve important proper names and distinctions
   between genuinely different words.

3. **File-based persistence instead of a database**

   The prototype does not use a database. It stores index metadata and graph
   mappings in JSON files and embedding arrays in NumPy files inside a
   corpus-specific directory under `.rag_cache/`. This keeps the demonstration
   simple and allows completed indexing work to be reused, but it offers no
   database facilities for concurrent access, incremental updates, advanced
   queries, or management of very large indexes. A future production-oriented
   version could use a vector or graph database when those capabilities are
   required.

4. **Averaging different keyword contexts**

   The prototype represents each keyword by averaging the embeddings of all
   sentences containing it. It follows the KET-RAG approach represented by
   Algorithm 3, line 13. This is simple and efficient, but a word may have
   different meanings in different contexts. Their average may represent none
   of those meanings particularly well and can reduce retrieval accuracy for
   ambiguous keywords. A more advanced implementation could:

   1. retain multiple contextual vectors for each keyword;
   2. cluster the containing sentences by meaning and create one vector for
      each cluster.

## KET-RAG reference

```bibtex
@inproceedings{10.1145/3711896.3737012,
author = {Huang, Yiqian and Zhang, Shiqi and Xiao, Xiaokui},
title = {KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3737012},
doi = {10.1145/3711896.3737012},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {1003–1012},
numpages = {10},
keywords = {graphrag, indexing, retrieval-augmented generation},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```
