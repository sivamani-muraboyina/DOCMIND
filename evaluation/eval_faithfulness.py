"""
Faithfulness + answer-relevancy evaluation — hand-built, no RAGAS dependency.

Same logic as RAGAS's Faithfulness metric: decompose the generated answer
into individual claims, check each claim against the retrieved context,
score = supported_claims / total_claims. Uses the SAME Groq LLM your app
already calls for generation — no new dependency, no environment conflict.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from groq import Groq
from loguru import logger

from backend.services.legal_corpus import initialize_legal_corpus
from backend.services.retriever import retrieve_legal
from backend.services.llm_chain import generate_answer
from evaluation.build_test_set import EvalQuestion  # required for pickle.load() to resolve this class

TEST_SET_PATH = Path("evaluation/test_set.pkl")
SAMPLE_SIZE = 5
JUDGE_MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


import time

def _judge_call(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Judge call failed ({e}), retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                logger.error(f"Judge call failed after {max_retries} attempts: {e}")
                raise


def decompose_into_claims(answer: str) -> list[str]:
    prompt = f"""Break this answer into individual, atomic FACTUAL claims about 
the contract's content — one per line. 

Do NOT include meta-statements like "the context discusses X" or "based on the 
provided information" — only extract concrete facts the answer states ABOUT THE 
CONTRACT itself (e.g. dates, parties, obligations, clauses, terms).

If the answer says there is NO such clause/term, extract that as a claim too 
(e.g. "There is no non-compete clause in this agreement").

ANSWER:
{answer}

CLAIMS (one per line, no numbering):"""
    raw = _judge_call(prompt)
    claims = [c.strip("- ").strip() for c in raw.split("\n") if c.strip()]
    return claims


def is_claim_supported(claim: str, context: str) -> bool:
    prompt = f"""Context: {context}

Claim: {claim}

Is this claim directly supported by the context above? Answer YES or NO only."""
    verdict = _judge_call(prompt)
    return "YES" in verdict.upper()


def compute_faithfulness(answer: str, context: str) -> float:
    claims = decompose_into_claims(answer)
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if is_claim_supported(c, context))
    return supported / len(claims)


def compute_answer_relevancy(question: str, answer: str) -> float:
    """Simple LLM-judged relevancy: does the answer actually address the question?"""
    prompt = f"""Question: {question}

Answer: {answer}

On a scale of 0 to 10, how directly does this answer address the question asked 
(ignore correctness, only judge relevance/focus)? Reply with just the number."""
    raw = _judge_call(prompt)
    try:
        return min(10, max(0, float(raw.strip().split()[0]))) / 10
    except (ValueError, IndexError):
        return 0.5  # fallback if the judge gives an unparseable response


if __name__ == "__main__":
    initialize_legal_corpus()

    with open(TEST_SET_PATH, "rb") as f:
        questions = pickle.load(f)[:SAMPLE_SIZE]

    logger.info(f"Running hand-built faithfulness eval on {len(questions)} questions")

    faithfulness_scores, relevancy_scores = [], []

    for i, q in enumerate(questions):
        try:
            chunks = retrieve_legal(q.question, top_k_final=5, use_hnsw=False)
            if not chunks:
                continue

            context = "\n\n".join(c["content"] for c in chunks)
            answer, _ = generate_answer(
                question=q.question, chunks=chunks, history=[],
                temperature=0.2, max_tokens=1024,
            )

            f_score = compute_faithfulness(answer, context)
            r_score = compute_answer_relevancy(q.question, answer)

            faithfulness_scores.append(f_score)
            relevancy_scores.append(r_score)

            logger.info(f"[{i+1}/{len(questions)}] faithfulness={f_score:.2f} relevancy={r_score:.2f}")
        except Exception as e:
            logger.error(f"Question {i+1} failed permanently, skipping: {e}")
            continue

    n = len(faithfulness_scores)
    print(f"\n=== FAITHFULNESS RESULTS (n={n}) ===")
    print(f"Avg Faithfulness:     {sum(faithfulness_scores)/n:.4f}")
    print(f"Avg Answer Relevancy: {sum(relevancy_scores)/n:.4f}")