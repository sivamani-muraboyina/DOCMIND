import pickle
from evaluation.build_test_set import EvalQuestion
from backend.services.legal_corpus import initialize_legal_corpus, find_matching_document, _legal_store

initialize_legal_corpus()

with open("evaluation/test_set.pkl", "rb") as f:
    questions = pickle.load(f)

chunk_lookup = {c.chunk_id: c.file_path for c in _legal_store["chunks"]}

correct = 0
total = len(questions[:40])

for q in questions[:40]:
    matched = find_matching_document(q.question, top_n=1)
    true_files = {chunk_lookup[cid] for cid in q.relevant_chunk_ids if cid in chunk_lookup}

    hit = bool(matched) and matched[0] in true_files
    correct += hit

    status = "OK " if hit else "MISS"
    print(f"{status} | matched={matched} | true={list(true_files)[:1]}")
    print(f"      q: {q.question[:90]}")

print(f"\nMatched correctly: {correct}/{total} ({100*correct/total:.0f}%)")