# Group Report — Lab 18: Production RAG

**Nhóm:** C401-C5
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| [Tên 1] | M1: Chunking | ☑ | 8/8 |
| Lê Hữu Hưng | M2: Hybrid Search | ☑ | 5/5 |
| [Tên 3] | M3: Reranking | ☑ | 5/5 |
| [Tên 4] | M4: Evaluation | ☑ | 4/4 |
| [Tên 5] | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | N/A | 0.7184 | — |
| Answer Relevancy | N/A | 0.5232 | — |
| Context Precision | N/A | 0.7184 | — |
| Context Recall | N/A | 0.7167 | — |

> Pipeline chạy trên 2 tài liệu tiếng Việt: BCTC (tờ khai thuế GTGT Q4/2024) và Nghị định 13/2023 về bảo vệ dữ liệu cá nhân. Tổng 284 chunks sau M1, enriched bằng contextual prepend (M5), indexed bằng BM25 + bge-m3 dense (M2), reranked bằng bge-reranker (M3). Thời gian chạy: 737 giây.

## Key Findings

1. **Biggest improvement:** Hybrid search (BM25 + Dense + RRF) kết hợp với cross-encoder reranking giúp context_precision đạt 0.7184 — tốt hơn so với BM25 thuần hoặc dense thuần
2. **Biggest challenge:** Qdrant version mismatch (client 1.17 vs server 1.13) và Vietnamese font encoding trong PDF khiến mất nhiều thời gian setup. Giải pháp: dùng Qdrant in-memory và GPT-4o Vision OCR
3. **Surprise finding:** answer_relevancy thấp (0.5232) dù faithfulness cao — nguyên nhân là system prompt quá chặt khiến LLM trả lời "Không tìm thấy" với câu hỏi định nghĩa, dẫn đến 3/10 câu có answer_relevancy = 0.0

## Presentation Notes (5 phút)

1. **RAGAS scores:** Faithfulness 0.72, Context Precision 0.72, Context Recall 0.72 — cả 3 metric gần ngưỡng 0.75; answer_relevancy 0.52 do prompt issue
2. **Biggest win:** M5 Contextual Enrichment — prepend document summary vào mỗi chunk giúp LLM hiểu context tốt hơn, cải thiện faithfulness
3. **Case study:** "Các hành vi bị cấm về dữ liệu cá nhân?" — context_recall 0.48 vì Điều 8 bị chunk thành mảnh nhỏ, reranker chỉ lấy top-3 → fix: tăng RERANK_TOP_K
4. **Next optimization:** Tăng RERANK_TOP_K từ 3 lên 5 và thêm query expansion cho câu hỏi định nghĩa → ước tính đẩy faithfulness và context_recall lên ≥ 0.75
