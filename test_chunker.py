from pathlib import Path
from backend.utils.legal_chunker import load_and_chunk_corpus

chunks = load_and_chunk_corpus(Path("data/raw/corpus/cuad"))
print(f"Total chunks: {len(chunks)}")
print(chunks[0].file_path, chunks[0].char_start, chunks[0].char_end)
print(chunks[0].content[:200])