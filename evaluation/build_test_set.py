"""
Maps LegalBench-RAG's ground-truth (file_path, char_span) answers onto the
actual chunk_ids in our already-built legal corpus index.

A question's ground-truth snippet overlaps a chunk if their character
ranges intersect at all within the same file — this is the standard way to
turn span-based ground truth into chunk-based ground truth for Precision/
Recall/MRR evaluation (Stage 2.8).

Run this once; it saves a clean test_set.pkl that eval scripts read from,
so you don't re-parse cuad.json + re-scan all chunks every single run.
"""

from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

CUAD_JSON_PATH = Path("data/raw/benchmarks/cuad.json")
CHUNKS_PATH = Path("data/legal_index/legal_chunks.pkl")
OUTPUT_PATH = Path("evaluation/test_set.pkl")

RANDOM_SEED = 42  # reproducibility — same sample every run unless you change this


@dataclass
class EvalQuestion:
    question: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    ground_truth_answer: str = ""  # first snippet's answer text, for faithfulness comparison


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True if [a_start, a_end) and [b_start, b_end) intersect at all."""
    return a_start < b_end and b_start < a_end


def _build_file_index(chunks: list) -> dict:
    """Group chunks by file_path so we don't scan all 46,620 chunks per question."""
    by_file = defaultdict(list)
    for chunk in chunks:
        by_file[chunk.file_path].append(chunk)
    return by_file


def build_test_set(sample_size: int | None = 300) -> list[EvalQuestion]:
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    by_file = _build_file_index(chunks)

    with open(CUAD_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    all_tests = raw["tests"]

    if sample_size is not None and sample_size < len(all_tests):
        random.seed(RANDOM_SEED)
        all_tests = random.sample(all_tests, sample_size)

    logger.info(f"Mapping {len(all_tests)} questions onto chunk ground truth...")

    eval_questions = []
    unmapped_count = 0

    for test in all_tests:
        relevant_ids = set()
        first_answer = ""

        for snippet in test["snippets"]:
            file_path = snippet["file_path"]
            span_start, span_end = snippet["span"]
            if not first_answer:
                first_answer = snippet["answer"]

            candidates = by_file.get(file_path, [])
            for chunk in candidates:
                if _spans_overlap(span_start, span_end, chunk.char_start, chunk.char_end):
                    relevant_ids.add(chunk.chunk_id)

        if not relevant_ids:
            unmapped_count += 1
            continue  # skip questions whose ground truth didn't land in any chunk

        eval_questions.append(
            EvalQuestion(
                question=test["query"],
                relevant_chunk_ids=list(relevant_ids),
                ground_truth_answer=first_answer,
            )
        )

    logger.info(
        f"Built {len(eval_questions)} usable eval questions "
        f"({unmapped_count} skipped — no overlapping chunk found)"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(eval_questions, f)

    logger.info(f"Saved test set to {OUTPUT_PATH}")
    return eval_questions


if __name__ == "__main__":
    questions = build_test_set(sample_size=300)
    print(f"\nSample question: {questions[0].question}")
    print(f"Relevant chunk IDs: {questions[0].relevant_chunk_ids}")
    print(f"Ground truth answer: {questions[0].ground_truth_answer[:150]}")