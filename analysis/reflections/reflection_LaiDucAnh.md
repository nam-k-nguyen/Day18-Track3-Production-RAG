# Individual Reflection — Lab 18

**Tên:** Lại Đức Anh  
**Module phụ trách:** M4: Evaluation (RAGAS)

---

## 1. Đóng góp kỹ thuật

- Module đã implement: RAGAS Evaluation Pipeline — đánh giá chất lượng RAG bằng 4 metrics + failure analysis tự động.
- Các hàm/class chính đã viết:
  - `EvalResult`: dataclass lưu kết quả đánh giá cho từng câu hỏi (faithfulness, answer_relevancy, context_precision, context_recall)
  - `evaluate_ragas()`: chạy RAGAS evaluate với fallback lexical (token overlap) khi thư viện không khả dụng
  - `_token_overlap()`: hàm helper tính độ trùng token giữa 2 chuỗi (lexical fallback metric)
  - `failure_analysis()`: phân tích bottom-N câu hỏi tệ nhất, dùng decision tree để chẩn đoán lỗi và đề xuất fix
  - `save_report()`: lưu báo cáo aggregate + failure list ra JSON
  - `load_test_set()`: load test set từ JSON
- Số tests pass: 4/4

## 2. Kiến thức học được

- Khái niệm mới nhất: RAGAS evaluation framework — đánh giá RAG theo 4 chiều độc lập: faithfulness (LLM có hallucinate không), answer_relevancy (câu trả lời có đúng với câu hỏi không), context_precision (context được lấy có liên quan không), context_recall (context có đủ để trả lời không). Mỗi metric chỉ ra một điểm yếu khác nhau trong pipeline.
- Điều bất ngờ nhất: Failure analysis bằng decision tree đơn giản (so sánh worst_metric và threshold) lại rất hữu ích — chỉ cần 4 rule là xác định được root cause cho phần lớn lỗi, không cần model phức tạp.
- Kết nối với bài giảng: Trực tiếp implement "Offline Evaluation" từ slide — 4 RAGAS metrics tương ứng với 4 điểm kiểm tra trong RAG pipeline (generation, retrieval precision, retrieval recall). Failure analysis thực hiện "Error Tree" từ slide page-12.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: RAGAS library yêu cầu LLM để tính faithfulness và answer_relevancy — cần OpenAI API key, nếu không có thì evaluate() raise exception và toàn bộ evaluation bị skip.
- Cách giải quyết: Implement lexical fallback dùng token overlap — khi RAGAS raise exception, tự động chuyển sang tính metrics bằng Jaccard-style token overlap. Đảm bảo pipeline luôn chạy được dù không có API key.
- Thời gian debug: ~1 giờ (30 phút tìm hiểu RAGAS API thay đổi giữa các version, 30 phút viết và test fallback logic).

## 4. Nếu làm lại

Sẽ làm khác điều gì: Thêm per-metric threshold alerting — thay vì chỉ in điểm thấp, tự động flag khi bất kỳ metric nào dưới ngưỡng 0.75 và gợi ý module cụ thể cần cải thiện (M1 nếu context_recall thấp, M3 nếu context_precision thấp). Cũng sẽ thêm comparison giữa naive baseline và production pipeline để thấy rõ delta cải thiện.

Module nào muốn thử tiếp: M5 Enrichment — HyQA (Hypothetical Question Answering) rất thú vị vì cải thiện embedding mà không cần thay model; kỹ thuật này ảnh hưởng trực tiếp đến context_precision và context_recall mà M4 đang đo.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
