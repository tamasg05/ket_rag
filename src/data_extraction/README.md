# Data extraction and structure-aware chunking

The internal `data_extraction` package converts PDF documents or HTML pages
into a common structured representation and can then turn the extracted blocks
into chunks suitable for RAG applications. The common representation allows a
downstream RAG pipeline to process different source formats in the same way
while retaining useful structure such as headings, table rows, page numbers,
and source provenance.

**A central benefit is the structure-aware extraction of tables.** Instead of
flattening cells into an ambiguous sequence of text, the package reconstructs
headers, rows, columns, and their value relationships. When a table is rendered
into readable text or chunks, the appropriate column header is added to every
non-empty cell value to make its meaning explicit. For example, this row:

| Model | Power | Price |
| --- | --- | --- |
| A8 55 TFSI | 340 LE | 41256010 HUF |

is represented as:

```text
Model = A8 55 TFSI; Power = 340 LE; Price = 41256010 HUF.
```

This is much easier for retrieval and language models to interpret than
`A8 55 TFSI 340 LE 41256010 HUF`, where the role of each value is implicit.
Reliable table reconstruction is otherwise difficult, especially for PDFs
with rotated headings, merged visual rows, missing borders, or multiple tables
on one page.

Separating extraction from chunking and indexing also means that documents can
be parsed once, inspected, tested, and reused by different RAG systems without
repeating the potentially expensive or error-prone extraction step. PDF layout
interpretation is still heuristic, so new complex layouts should be covered by
regression fixtures before relying on them in production. In `blocks.json`,
headers and row cells remain stored separately; the explicit `column = value`
form is created during readable rendering and chunk construction.

The two main entry points are:

```python
from pathlib import Path

from src.data_extraction import build_chunks, extract_corpus

corpus = extract_corpus(
    [Path("specification.pdf")],
    output_directory=Path("extracted"),
)

chunks = build_chunks(
    corpus.blocks_path,
    output_path=corpus.blocks_path.with_name("chunks.json"),
    strategy="words",
    chunk_size=450,
    chunk_overlap=60,
)
```

`extract_corpus()` creates `blocks.json`, `corpus.txt`, and `sources.json` in a
content-addressed corpus directory. For PDF input, the source files are copied
under its `sources/` subdirectory. `build_chunks()` accepts either a block list
or a path to `blocks.json` and can optionally persist `chunks.json`.

## Chunk-size interpretation

`chunk_size=450` is a maximum target, not a required or minimum length. The
chunker does not add content merely to make every chunk equally long.

For ordinary text, consecutive blocks can be combined only while they belong
to the same source, page, and heading path and are not interrupted by a table.
The combined text is then divided into overlapping word ranges of at most the
configured size. If only a short paragraph or heading is available before a
structural boundary, the resulting chunk is correspondingly short.

Tables are processed separately. Complete rows are packed into a chunk until
adding another row would exceed the target size. Rows are not split merely to
reach a uniform chunk length. Consequently, a small table or the final rows of
a table can also produce a short chunk.

For example, a chunk such as:

```json
{
  "text": "Audi Hosszított tengelytáv",
  "source_text": "Audi Hosszított tengelytáv",
  "block_type": "text"
}
```

is likely a section heading that was flushed as text when the following table
was encountered. If the table already retains the same value as its caption or
heading path, the independent heading-only chunk is redundant.

Short chunks are not automatically incorrect. A concise paragraph may contain
a complete and important fact. However, structurally incomplete or duplicated
heading-only chunks can:

- occupy a limited `top-k` retrieval position without supplying the requested
  facts;
- create unnecessary embeddings and graph nodes;
- influence KNN connections and PageRank;
- cause graph-extraction work to be spent on little useful content; and
- overweight phrases repeated in both a heading chunk and a table caption.

## `text` and `source_text`

Each chunk contains two representations:

- `text` is a predictable word-token form without surrounding punctuation;
- `source_text` preserves punctuation, headings, and explicit table
  `column = value` relationships.

Removing punctuation is not inherently better for modern embedding models.
The comparison prototype embeds `text` for compatibility and consistency with
its lexical processing, while it supplies `source_text` to the answering model.
Another RAG application can choose either representation for embedding.

## Future Work

1. **Associate headings with following tables.** When a heading immediately
   introduces a table, store it in the table's `caption` or `heading_path` and
   include it in the table's `source_text`.

2. **Suppress duplicate heading-only chunks.** If a table already retains its
   introductory heading, do not also produce an independent chunk containing
   only the same heading.

3. **Support a configurable minimum text-chunk target.** Compatible short text
   blocks could be merged to reduce retrieval and indexing overhead. Merging
   must continue to respect structural boundaries: blocks should not be joined
   blindly across pages, tables, sources, or unrelated sections.

4. **Propagate vertically merged labels to their logical rows.** Some tables
   display an equipment level or category once across several physical rows. The
   extractor should repeat that value in every resulting logical row so each
   row remains independently interpretable.

5. **Represent cells spanning several columns explicitly.** A value centered
   across multiple variants can otherwise be split between columns. Future
   block-schema versions could retain `rowspan` and `colspan` metadata or
   expand the shared value into every affected logical cell.
