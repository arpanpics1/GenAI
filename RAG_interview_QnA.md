Q1
Your RAG pipeline's retrieval precision is high but recall is poor — users are missing relevant docs. How do you diagnose and fix this without degrading precision?
▾
Diagnose first: compute recall@k across a labeled eval set; check if missed docs exist in the index at all (ingestion gap) vs. exist but aren't ranked high (embedding gap).
Embedding gap: try hybrid search (dense + sparse BM25), increase top-k + re-rank with a cross-encoder, or fine-tune embeddings on domain data.
Chunking gap: long or poorly split chunks bury relevant sentences — experiment with smaller chunks or sentence-window retrieval.
Query gap: add HyDE (hypothetical document embedding) or query expansion via LLM to cover vocabulary mismatch.
Guard precision: use re-ranking as a post-filter so you retrieve wider but still serve narrow — re-ranker becomes your precision gate.
