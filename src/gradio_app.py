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


def build_indexes(knn_k, ket_k, beta, tau, progress=gr.Progress()):
    """
    Gradio callback that builds or loads the selected persistent indexes.

    Inputs:
        knn_k: KNNG-RAG neighbour slider value.
        ket_k: KET-RAG neighbour slider value.
        beta: KET skeleton-fraction slider value.
        tau: KET subchunk-split slider value.
        progress: Gradio progress reporter supplied by the framework.

    Returns:
        A UI status string containing success details or the build error.
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
        result = get_service().build(knn_k, ket_k, beta, tau, report)
        return result + "\n" + "\n".join(messages[-4:])
    except Exception as exc:
        return f"Build failed: {type(exc).__name__}: {exc}"


def compare(query, top_k, temperature, knn_k, ket_k, beta, tau, theta):
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

    Returns:
        Three answer strings and one retrieval-diagnostics dictionary.
    """
    try:
        return get_service().compare(
            query, top_k, temperature, knn_k, ket_k, beta, tau, theta
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return message, message, message, {"error": message}


with gr.Blocks(title="Three RAG Systems — Sherlock Holmes") as demo:
    gr.Markdown(
        """
        # Compare three RAG systems
        Ask one question and compare answers grounded in *The Adventures of Sherlock Holmes*.
        Build once; matching persistent artifacts are reused on later runs.
        """
    )
    query = gr.Textbox(
        label="Question",
        placeholder="For example: Why did Irene Adler keep the photograph?",
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
        build_button = gr.Button("Build/load indexes")
        compare_button = gr.Button("Compare answers", variant="primary")
    status = gr.Textbox(label="Index status", interactive=False, lines=2)

    with gr.Row():
        text_answer = gr.Textbox(label="1. Text RAG", lines=14)
        knn_answer = gr.Textbox(label="2. KNNG-RAG", lines=14)
        ket_answer = gr.Textbox(label="3. KET-RAG", lines=14)
    with gr.Accordion("Retrieval diagnostics", open=False):
        diagnostics = gr.JSON(label="Retrieved IDs and KET seed nodes")

    build_button.click(
        build_indexes,
        inputs=[knn_k, ket_k, beta, tau],
        outputs=status,
    )
    compare_button.click(
        compare,
        inputs=[query, top_k, temperature, knn_k, ket_k, beta, tau, theta],
        outputs=[text_answer, knn_answer, ket_answer, diagnostics],
    )


if __name__ == "__main__":
    demo.launch()
