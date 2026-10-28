"""Optional CLI for building the same persistent indexes used by the UI."""

from __future__ import annotations

import argparse

from .rag_comparison import RagComparison


def main() -> None:
    """
    Parse CLI parameters and build the matching persistent indexes.

    Inputs:
        None directly; values are read from command-line arguments.

    Returns:
        None. Build progress and the final status are printed to the terminal.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--ket-k", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--tau", type=int, default=1)
    args = parser.parse_args()
    app = RagComparison()
    print(app.build(args.knn_k, args.ket_k, args.beta, args.tau, print))


if __name__ == "__main__":
    main()
