# Failure Analysis — Lab 18: Production RAG

**Nhóm:** C401-C5  
**Thành viên:** [Lê Tú Nam → M1] · [Lê Hữu Hưng → M2] · [Đỗ Minh Phúc → M3] · [Lại Đức Anh → M4] · [Nguyễn Khánh Nam → M5]

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | N/A | 0.7184 | — |
| Answer Relevancy | N/A | 0.5232 | — |
| Context Precision | N/A | 0.7184 | — |
| Context Recall | N/A | 0.7167 | — |

> Naive baseline không chạy do tập trung vào production pipeline.

---

## Bottom-5 Failures

### #1
- **Question:** Nếu thuế đầu vào lớn hơn thuế đầu ra thì xử lý thế nào?
- **Expected:** Chênh lệch được khấu trừ kỳ sau hoặc hoàn thuế (chỉ tiêu [41], [42], [43] trong mẫu 01/GTGT)
- **Got:** Câu trả lời không liên quan đến câu hỏi
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:** Output sai → Context có liên quan? (một phần) → Prompt quá chặt → LLM trả lời "Không tìm thấy" thay vì suy luận từ bảng số liệu
- **Root cause:** System prompt "Trả lời CHỈ dựa trên context" khiến LLM từ chối suy luận khi context chứa số liệu nhưng không có câu trả lời rõ ràng
- **Suggested fix:** Thêm hướng dẫn "nếu context có số liệu liên quan, hãy suy luận và trả lời dựa trên đó"

### #2
- **Question:** Theo nghị định 13, dữ liệu cá nhân là gì?
- **Expected:** Thông tin gắn liền với một con người cụ thể, bao gồm dữ liệu cơ bản và dữ liệu nhạy cảm (Điều 2, NĐ 13/2023)
- **Got:** Câu trả lời không khớp với câu hỏi
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:** Output sai → Context đúng? (không, chunk sai được retrieve) → Query "dữ liệu cá nhân là gì" quá ngắn → BM25 không segment đúng → Retrieve chunk sai
- **Root cause:** Query ngắn, từ khóa phổ biến → BM25 trả về chunk không liên quan → LLM không có context để trả lời
- **Suggested fix:** Query expansion hoặc tăng DENSE_TOP_K để hybrid search lấy được chunk định nghĩa từ Điều 2

### #3
- **Question:** Thuế GTGT đầu ra là gì?
- **Expected:** Thuế tính trên hàng hóa/dịch vụ bán ra, chỉ tiêu [28] = 344.675.400 đồng trong kỳ Q4/2024
- **Got:** Câu trả lời không liên quan
- **Worst metric:** answer_relevancy = 0.0
- **Error Tree:** Output sai → Context có bảng số liệu GTGT? (có) → LLM hiểu câu hỏi định nghĩa hay số liệu? → Mơ hồ → Trả lời sai hướng
- **Root cause:** Câu hỏi vừa hỏi định nghĩa vừa có thể hỏi số liệu; LLM không xác định được intent → trả lời không đúng chiều
- **Suggested fix:** Cải thiện prompt: phân biệt câu hỏi định nghĩa vs câu hỏi số liệu

### #4
- **Question:** Các hành vi bị cấm liên quan đến dữ liệu cá nhân là gì?
- **Expected:** Các hành vi cấm theo Điều 8 NĐ 13/2023 (thu thập, xử lý trái phép, mua bán dữ liệu...)
- **Got:** Trả lời thiếu nhiều hành vi bị cấm
- **Worst metric:** context_recall = 0.4815
- **Error Tree:** Output thiếu → Context thiếu → Chunking cắt Điều 8 thành nhiều chunk nhỏ → Reranker chỉ lấy top-3 → Bỏ sót các mục con
- **Root cause:** Hierarchical chunking cắt nhỏ danh sách các hành vi bị cấm; RERANK_TOP_K=3 không đủ để cover toàn bộ Điều 8
- **Suggested fix:** Tăng RERANK_TOP_K từ 3 lên 5-7, hoặc dùng parent chunk khi câu hỏi yêu cầu danh sách đầy đủ

### #5
- **Question:** Doanh thu chịu thuế GTGT được xác định như thế nào?
- **Expected:** Dựa vào chỉ tiêu [32] = 3.703.695.610 đồng, là giá trị hàng hóa/dịch vụ bán ra chịu thuế 10%
- **Got:** Câu trả lời có thông tin ngoài context (hallucination)
- **Worst metric:** faithfulness = 0.6667
- **Error Tree:** Output không trung thực → LLM thêm thông tin ngoài context → Temperature cao + prompt không đủ chặt → Hallucinate
- **Root cause:** LLM bổ sung kiến thức nền về thuế GTGT không có trong document, vi phạm ràng buộc "chỉ dựa trên context"
- **Suggested fix:** Thêm ví dụ few-shot trong prompt, nhấn mạnh "không được thêm thông tin ngoài context"

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Các hành vi bị cấm liên quan đến dữ liệu cá nhân là gì?" (context_recall = 0.4815)

**Error Tree walkthrough:**
1. Output đúng? → **Không** — thiếu nhiều hành vi bị cấm
2. Context đúng? → **Một phần** — chỉ retrieve được 3 chunk, bỏ sót phần lớn Điều 8
3. Chunking OK? → **Không** — Điều 8 bị cắt thành nhiều chunk nhỏ 256 token, reranker chỉ giữ top-3
4. Fix ở bước: **Chunking** (tăng child_size) + **RERANK_TOP_K** (tăng lên 5)

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tăng `RERANK_TOP_K` từ 3 lên 5 → cải thiện context_recall ngay lập tức
- Thêm query expansion: với câu hỏi "là gì" → tự động thêm "định nghĩa", "khái niệm" vào query
- Cải thiện system prompt: phân biệt câu hỏi định nghĩa vs câu hỏi số liệu
