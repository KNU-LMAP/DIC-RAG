#!/usr/bin/env python
"""Single entry point for the four systems compared in the paper.

  python main.py --method llm           --questions q.txt --answers a.txt
  python main.py --method vrag          --questions q.txt --answers a.txt --pdf guide.pdf
  python main.py --method longllmlingua --questions q.txt --answers a.txt --pdf guide.pdf
  python main.py --method ssc           --questions q.txt --answers a.txt --pdf guide.pdf

The OpenAI key is read from the OPENAI_API_KEY environment variable.
"""

import argparse
import os
import sys

from dic_rag import METHODS
from dic_rag.evaluation import extract_answer_number, parse_correct_answers, parse_questions, run_evaluation
from dic_rag.systems import LLMSystem, RAGSystem


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--questions", required=True, help="question file")
    p.add_argument("--answers", required=True, help="answer-key file")
    p.add_argument("--pdf", nargs="+", help="knowledge-base PDF(s); required for all methods except llm")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--tag", default="", help="suffix for the output file names (e.g. level or run id)")
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--embedding-model", default="paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--k", type=int, default=20, help="number of retrieved chunks")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=100)
    p.add_argument("--lingua-ratio", type=float, default=0.4, help="LongLLMLingua compression ratio")
    p.add_argument("--ssc-ratio", type=float, default=0.4, help="SSC compression ratio")
    p.add_argument(
        "--legacy-lingua-state",
        action="store_true",
        help="reproduce the original experiments: LongLLMLingua overwrites the stored chunks, so compression "
        "accumulates over questions (off by default: every question is compressed from the original text)",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.method != "llm" and not args.pdf:
        p.error("--pdf is required for methods other than llm")
    return args


def build_system(args, client):
    if args.method == "llm":
        return LLMSystem(client, model=args.model, answer_parser=extract_answer_number)

    from dic_rag.data import DocumentProcessor
    from dic_rag.retrieval import VectorStore

    chunks = DocumentProcessor(args.chunk_size, args.chunk_overlap, verbose=args.verbose).process_pdf(args.pdf)
    store = VectorStore(args.embedding_model)
    store.build_index(chunks)

    compressor = None
    if args.method == "longllmlingua":
        from dic_rag.lingua import LongLLMLinguaCompressor

        compressor = LongLLMLinguaCompressor(
            compression_ratio=args.lingua_ratio, persist_compression=args.legacy_lingua_state, verbose=args.verbose
        )
    elif args.method == "ssc":
        from dic_rag.ssc import SSCCompressor

        compressor = SSCCompressor(compression_ratio=args.ssc_ratio, verbose=args.verbose)
    return RAGSystem(client, store, model=args.model, k=args.k, compressor=compressor, name=args.method)


def main(argv=None):
    args = get_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI()
    questions = parse_questions(args.questions)
    correct = parse_correct_answers(args.answers)
    if not questions:
        sys.exit("no questions could be parsed")
    system = build_system(args, client)
    run_evaluation(system, questions, correct, args.output_dir, args.tag)


if __name__ == "__main__":
    main()
