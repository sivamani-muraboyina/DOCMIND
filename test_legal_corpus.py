from backend.services.legal_corpus import initialize_legal_corpus, search_legal_faiss, search_legal_bm25, get_legal_corpus_stats

initialize_legal_corpus()
print(get_legal_corpus_stats())

results = search_legal_faiss("What is the termination notice period?", top_k=3, use_hnsw=False)
for chunk, score in results:
    print(f"\nscore={score:.4f} | file={chunk.file_path}")
    print(chunk.content[:200])