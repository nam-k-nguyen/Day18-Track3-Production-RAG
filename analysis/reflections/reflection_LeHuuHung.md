# Individual Reflection — Lab 18

**Tên:** Lê Hữu Hưng — 2A202600098  
**Module phụ trách:** M2: Hybrid Search

---

## 1. Đóng góp kỹ thuật

- Module đã implement: Hybrid Search Pipeline kết hợp BM25 (Vietnamese) + Dense Vector Search + Reciprocal Rank Fusion (RRF).
- Các hàm/class chính đã viết:
  - `segment_vietnamese()`: tách từ tiếng Việt dùng underthesea word_tokenize
  - `BM25Search`: index và search bằng BM25Okapi trên corpus đã segment
  - `DenseSearch`: index và search bằng SentenceTransformer (BAAI/bge-m3) + Qdrant in-memory
  - `reciprocal_rank_fusion()`: merge kết quả BM25 và Dense theo công thức RRF score = Σ 1/(k + rank + 1)
  - `HybridSearch`: orchestrate BM25 + Dense + RRF thành một pipeline thống nhất
- Số tests pass: 5/5

## 2. Kiến thức học được

- Khái niệm mới nhất: Reciprocal Rank Fusion — cách merge nhiều ranked list mà không cần normalize score, hiệu quả hơn weighted average vì không phụ thuộc vào scale của từng model.
- Điều bất ngờ nhất: underthesea word_tokenize cải thiện đáng kể chất lượng BM25 với tiếng Việt — tiếng Việt là ngôn ngữ có từ ghép nên tách từ đúng quyết định độ chính xác của BM25.
- Kết nối với bài giảng: Trực tiếp implement "Hybrid Search = BM25 + Dense + RRF" từ slide — BM25 tốt cho exact match (mã số thuế, tên điều luật), Dense tốt cho semantic match (câu hỏi paraphrase).

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Qdrant version mismatch — qdrant-client 1.17.1 không tương thích với server 1.13.0; `recreate_collection` và `client.search()` đều bị deprecated/removed.
- Cách giải quyết: Chuyển sang Qdrant in-memory (`QdrantClient(":memory:")`) và đổi `client.search()` sang `client.query_points().points` — loại bỏ hoàn toàn dependency vào Docker server.
- Thời gian debug: ~45 phút (3 lần lỗi Qdrant khác nhau trước khi tìm ra root cause).

## 4. Nếu làm lại

Sẽ làm khác điều gì: Thêm caching embedding để không phải encode lại toàn bộ corpus mỗi lần chạy pipeline; thêm query expansion trước khi search để cải thiện recall cho câu hỏi ngắn.

Module nào muốn thử tiếp: M5 Enrichment — contextual prepend và HyQA rất thú vị, trực tiếp cải thiện chất lượng embedding mà không cần thay model.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
