# Individual Reflection — Lab 18

**Tên:** [LÊ TÚ NAM]- 2A202600109  
**Module phụ trách:** [M1]

---

## 1. Đóng góp kỹ thuật

- Module đã implement:Chunking strategies cho Ingestion Pipeline trong RAG (Retrieval-Augmented Generation).
- Các hàm/class chính đã viết:  Chunk dataclass; chunk_basic (fixed-size); chunk_semantic (dùng SentenceTransformer cosine sim < 0.75); chunk_hierarchical (parent 2000/child 500 tokens, cha-con); chunk_structure_aware (theo header Markdown); load_documents (PDF/TXT/MD từ ./data/); compare_strategies (stats số chunk, độ dài).
- Số tests pass: 13/13

## 2. Kiến thức học được

- Khái niệm mới nhất: Hierarchical chunking - chia parent lớn cho context, child nhỏ cho retrieval chính xác, khớp slide page-4 "retrieve child, return parent".
- Điều bất ngờ nhất: Semantic chunking dùng embedding để nhóm câu tương đồng tự động, nhanh hơn expected với model cache all-MiniLM-L6-v2.
- Kết nối với bài giảng (slide nào):  Trực tiếp implement 3 chunking strategies (Fixed, Semantic, Hierarchical) từ slide page-4, fix "Chunking Mismatch" offline failure slide page-1/3.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất:  Regex split câu/đoạn trong chunk_semantic và chunk_structure_aware fail với text PDF noisy (nhiều và lộn xộn)
- Cách giải quyết: Cách giải quyết: Thêm re.split(r'(?<=[.!?])\s+|\n\n'), lookbehind cho sentence; test với sample PDF
- Thời gian debug:  2 giờ (1h regex, 1h hierarchical parent_id logic).

## 4. Nếu làm lại

Sẽ làm khác điều gì: Thêm overlap 20% cho basic chunk.
Module nào muốn thử tiếp: Retrieval (Hybrid BM25 + Dense, reranking bge-reranker-v2-m3) vì LÀ "Highest ROI"

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 3|
| Code quality | 3|
| Teamwork | 3|
| Problem solving | 3|
