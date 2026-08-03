"""Gradio UI for comparing Text RAG, KNNG-RAG, and KET-RAG."""

from __future__ import annotations

import re

import gradio as gr

from .rag_comparison import RagComparison


service: RagComparison | None = None


def _progress_counts(message: str) -> tuple[int, int] | None:
    """
    Read the last completed/total pair from a status message.

    Inputs:
        message: Backend status text that may contain ``number/number``.

    Returns:
        The completed and total counts, or ``None`` when no pair is present.
    """
    matches = re.findall(r"(\d+)\s*/\s*(\d+)", message)
    if not matches:
        return None
    completed, total = matches[-1]
    return int(completed), int(total)


def get_service() -> RagComparison:
    """
    Return the single lazily created comparison service.

    Inputs:
        None.

    Returns:
        The process-wide ``RagComparison`` instance.
    """
    global service
    if service is None:
        service = RagComparison()
    return service


def build_indexes(
    knn_k,
    ket_k,
    beta,
    tau,
    use_url_corpus,
    url_text,
    use_pdf_corpus,
    pdf_files,
    progress=gr.Progress(),
):
    """
    Gradio callback that builds or loads the selected persistent indexes.

    Inputs:
        knn_k: KNNG-RAG neighbour slider value.
        ket_k: KET-RAG neighbour slider value.
        beta: KET skeleton-fraction slider value.
        tau: KET subchunk-split slider value.
        use_url_corpus: URL-corpus checkbox value.
        url_text: Multiline URL textbox value.
        use_pdf_corpus: PDF-corpus checkbox value.
        pdf_files: Uploaded PDF file paths.
        progress: Gradio progress reporter supplied by the framework.

    Returns:
        The UI status string and an update enabling the comparison button only
        after a successful build/load.
    """
    messages: list[str] = []

    def report(message: str):
        """
        Forward one backend status message to Gradio.

        Inputs:
            message: Human-readable build status, optionally with counts.

        Returns:
            None.
        """
        messages.append(message)
        # Backend progress messages use "completed/total". Passing that pair
        # to Gradio produces a real percentage instead of the previous 0.0%.
        counts = _progress_counts(message)
        if counts:
            completed, total = counts
            progress((completed, total), desc=message)
        else:
            # A phase without a count is indeterminate, not zero-percent.
            progress(None, desc=message)

    try:
        result = get_service().build(
            knn_k,
            ket_k,
            beta,
            tau,
            report,
            bool(use_url_corpus),
            url_text,
            bool(use_pdf_corpus),
            pdf_files,
        )
        status = result + "\n" + "\n".join(messages[-4:])
        return status, gr.update(interactive=True)
    except Exception as exc:
        status = f"Build failed: {type(exc).__name__}: {exc}"
        return status, gr.update(interactive=False)


def disable_compare():
    """
    Disable comparison after an index-defining UI value changes.

    Inputs:
        None; this callback is triggered by corpus and graph controls.

    Returns:
        A Gradio component update that makes the comparison button inactive.
    """
    return gr.update(interactive=False)


def compare(
    query,
    top_k,
    temperature,
    knn_k,
    ket_k,
    beta,
    tau,
    theta,
    use_url_corpus,
    url_text,
    use_pdf_corpus,
    pdf_files,
):
    """
    Gradio callback that compares answers from the three loaded RAG systems.

    Inputs:
        query: Question textbox value.
        top_k: Retrieval-count slider value.
        temperature: Answer-temperature slider value.
        knn_k: KNNG-RAG neighbour slider value.
        ket_k: KET-RAG neighbour slider value.
        beta: KET skeleton-fraction slider value.
        tau: KET subchunk-split slider value.
        theta: KET skeleton-retrieval-share slider value.
        use_url_corpus: URL-corpus checkbox value.
        url_text: Multiline URL textbox value.
        use_pdf_corpus: PDF-corpus checkbox value.
        pdf_files: Uploaded PDF file paths.

    Returns:
        Three answer strings and one retrieval-diagnostics dictionary.
    """
    try:
        return get_service().compare(
            query,
            top_k,
            temperature,
            knn_k,
            ket_k,
            beta,
            tau,
            theta,
            bool(use_url_corpus),
            url_text,
            bool(use_pdf_corpus),
            pdf_files,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return message, message, message, {"error": message}


with gr.Blocks(title="Compare Three RAG Systems") as demo:
    gr.Markdown(
        """
        # Compare three RAG systems
        Ask one question and compare three answers grounded in the selected data corpus.
        Build once; matching persistent artifacts are reused on later runs.
        """
    )

    with gr.Accordion("Data corpus", open=True):
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("#### Web-page corpus")
                with gr.Group():
                    use_url_corpus = gr.Checkbox(
                        value=False,
                        label="Build the corpus from web-page URLs",
                        info=(
                            "Select this for URL pages. Leave it unchecked for the "
                            "local text-file or PDF mode."
                        ),
                    )
                    url_text = gr.Textbox(
                        label="HTML page URLs (one per line)",
                        placeholder=(
                            "https://example.org/page-one\n"
                            "https://example.org/page-two"
                        ),
                        lines=5,
                        info=(
                            "Used only when the checkbox is selected. Static "
                            "headings, lists, paragraphs, and tables are saved "
                            "locally before indexing."
                        ),
                    )

            with gr.Column(scale=1, min_width=320):
                gr.Markdown("#### PDF corpus")
                with gr.Group():
                    use_pdf_corpus = gr.Checkbox(
                        value=False,
                        label="Build the corpus from uploaded PDF documents",
                        info=(
                            "Select either URL mode or PDF mode. When both are "
                            "unchecked, the configured local text file is used."
                        ),
                    )
                    pdf_files = gr.File(
                        label="PDF documents",
                        file_count="multiple",
                        file_types=[".pdf"],
                        type="filepath",
                    )

    query = gr.Textbox(
        label="Question",
        placeholder="For example: What are the main conclusions in the data?",
        lines=2,
    )

    with gr.Row():
        top_k = gr.Slider(1, 20, value=6, step=1, label="Top-k retrieved chunks")
        temperature = gr.Slider(0, 1.5, value=0.2, step=0.1, label="Answer temperature")

    with gr.Accordion("Graph/index parameters", open=True):
        gr.Markdown(
            "Changing these parameters selects a different persistent index. "
            "Click **Build/load indexes** before comparing."
        )
        with gr.Row():
            knn_k = gr.Slider(2, 20, value=6, step=2, label="KNNG-RAG k")
            ket_k = gr.Slider(2, 20, value=6, step=2, label="KET-RAG k")
            beta = gr.Slider(0.05, 1.0, value=0.2, step=0.05, label="KET-RAG beta")
            tau = gr.Slider(0, 3, value=1, step=1, label="KET-RAG tau")
            theta = gr.Slider(
                0, 1, value=0.4, step=0.1, label="KET-RAG theta (skeleton share)"
            )

    with gr.Row():
        build_button = gr.Button(
            "Build/load indexes", variant="primary", interactive=True
        )
        compare_button = gr.Button(
            "Compare answers", variant="primary", interactive=False
        )
    status = gr.Textbox(label="Index status", interactive=False, lines=2)

    with gr.Row():
        text_answer = gr.Textbox(label="1. Text RAG", lines=14)
        knn_answer = gr.Textbox(label="2. KNNG-RAG", lines=14)
        ket_answer = gr.Textbox(label="3. KET-RAG", lines=14)
    with gr.Accordion("Retrieval diagnostics", open=False):
        diagnostics = gr.JSON(label="Retrieved IDs and KET seed nodes")

    build_button.click(
        build_indexes,
        inputs=[
            knn_k,
            ket_k,
            beta,
            tau,
            use_url_corpus,
            url_text,
            use_pdf_corpus,
            pdf_files,
        ],
        outputs=[status, compare_button],
    )

    # Once one of these values changes, the currently loaded index can no
    # longer be assumed to match the UI selection. Query-only settings such as
    # top-k, temperature, theta, and the question do not require rebuilding.
    for index_control in (
        knn_k,
        ket_k,
        beta,
        tau,
        use_url_corpus,
        url_text,
        use_pdf_corpus,
        pdf_files,
    ):
        index_control.change(
            disable_compare,
            inputs=None,
            outputs=compare_button,
            queue=False,
        )
    compare_button.click(
        compare,
        inputs=[
            query,
            top_k,
            temperature,
            knn_k,
            ket_k,
            beta,
            tau,
            theta,
            use_url_corpus,
            url_text,
            use_pdf_corpus,
            pdf_files,
        ],
        outputs=[text_answer, knn_answer, ket_answer, diagnostics],
    )


if __name__ == "__main__":
    demo.launch()
