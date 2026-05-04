# Individual Reflection — Lab 18

**Tên:** Nguyễn Khánh Nam  
**MSSV:** 2A202600172  
**Module phụ trách:** M5 - Enrichment Pipeline

---

## 1. Đóng góp kỹ thuật

- Module đã implement: Enrichment Pipeline cho chunks trước khi embed trong RAG pipeline.
- Các hàm/class chính đã viết:
  - `EnrichedChunk` dataclass — cấu trúc lưu chunk đã làm giàu
  - `summarize_chunk()` — LLM tóm tắt chunk 2-3 câu, fallback extractive (lấy 2 câu đầu)
  - `generate_hypothesis_questions()` — Generate câu hỏi chunk có thể trả lời để bridge vocabulary gap, fallback keyword-based
  - `contextual_prepend()` — Prepend context mô tả chunk nằm ở đâu trong document (Anthropic style)
  - `extract_metadata()` — LLM extract JSON metadata, fallback rule-based category/entity extraction
  - `enrich_chunks()` — Orchestrate toàn bộ enrichment pipeline trên list chunks
  - `_get_openai_client()`, `_llm_call()` — Helper functions cho LLM interaction với graceful fallback khi không có API key
- Số tests pass: /8 (pytest tests/test_m5.py)

## 2. Kiến thức học được

- Khái niệm mới nhất: **Hypothesis Question-Answer (HyQA)** — generate câu hỏi mà chunk có thể trả lời, index cả questions và chunk để bridge vocabulary gap giữa query và document.
- Điều bất ngờ nhất: Enrichment chỉ là **one-time cost** khi indexing, không ảnh hưởng inference latency. Dùng gpt-4o-mini chi phí rất thấp (~$0.001/chunk) nhưng cải thiện đáng kể retrieval quality.
- Kết nối với bài giảng: Enrichment techniques từ Anthropic benchmark — contextual prepend alone giảm 49% retrieval failure. HyQA và auto metadata giúp dense retrieval match tốt hơn khi query dùng từ khác với document.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Thiết kế fallback mechanism cho trường hợp không có OpenAI API key — phải đảm bảo pipeline vẫn chạy được với rule-based extraction (keyword matching, regex entity extraction, extractive summarization).
- Cách giải quyết: Mỗi enrichment function có 2 path — LLM call ưu tiên trước, nếu fail (None hoặc exception) thì fallback về heuristic. Dùng `_llm_call()` helper return None khi không có API key, mỗi function check None rồi chạy fallback logic.
- Thời gian debug: 1.5 giờ (chủ yếu cho extract_metadata JSON parsing — LLM trả về markdown code block cần regex extract, plus fallback category classification dùng keyword matching).

## 4. Nếu làm lại

Sẽ làm khác điều gì: Tích hợp enrichment trực tiếp vào pipeline.py thay vì để riêng, thêm caching cho LLM calls để tránh duplicate work trên chunks giống nhau. Dùng batch API calls thay vì single request per chunk để giảm latency.

Module nào muốn thử tiếp: Retrieval optimization (hybrid BM25 + Dense search với adaptive weighting) vì enrichment chỉ có ý nghĩa khi retrieval quality được cải thiện thực sự.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
