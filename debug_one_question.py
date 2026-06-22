from backend.services.legal_corpus import initialize_legal_corpus, search_legal_faiss, search_legal_bm25
from backend.services.retriever import retrieve_legal

initialize_legal_corpus()

query = "Is there a non-compete clause in this contract?"

print("=== DENSE ONLY (top 5) ===")
for chunk, score in search_legal_faiss(query, top_k=5, use_hnsw=False):
    print(f"{score:.4f} | {chunk.chunk_id} | {chunk.file_path} | start={chunk.char_start}")

print("\n=== SPARSE ONLY (top 5) ===")
for chunk, score in search_legal_bm25(query, top_k=5):
    print(f"{score:.4f} | {chunk.chunk_id} | {chunk.file_path} | start={chunk.char_start}")

print("\n=== FULL PIPELINE retrieve_legal() output (top 5) ===")
results = retrieve_legal(query, top_k_final=5, use_hnsw=False)
for r in results:
    print(r)