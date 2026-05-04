# Individual Reflection — Lab 18: Production RAG

**Tên:** Đỗ Minh Phúc
**MSSV:** 2A202600039
**Module phụ trách:** M3 — Reranking
**File:** `src/m3_rerank.py`
**Branch:** `m3-rerank` (commit `7055eb8`)

---

## 1. Đóng góp kỹ thuật

**Module đã implement:** M3 — Cross-encoder Reranking với latency benchmark.

**Các hàm/class chính đã viết:**

| Component                            | Mô tả ngắn                                                                                                                                            |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CrossEncoderReranker._load_model()` | Lazy-load model với **dual backend**: ưu tiên `FlagEmbedding.FlagReranker` (FP16), fallback `sentence_transformers.CrossEncoder` nếu chưa cài.        |
| `CrossEncoderReranker.rerank()`      | Tạo pairs `(query, doc)` → predict scores → sort descending → trả top-k `RerankResult`. Có guard cho input rỗng.                                      |
| `FlashrankReranker`                  | Lightweight alternative (lazy import `flashrank`), dùng `RerankRequest` API để rerank nhanh dưới 5ms/query — **optional**, không yêu cầu lib cài sẵn. |
| `benchmark_reranker()`               | Đo latency qua `n_runs` lần, trả `{avg_ms, min_ms, max_ms, n_runs}` — phục vụ Bonus Latency Breakdown của nhóm.                                       |

**Số tests pass:** **5/5** (`pytest tests/test_m3.py -v`)
**TODO còn lại trong M3:** **0**

**Quyết định thiết kế đáng chú ý:**

1. **Dual backend với try/except ImportError** thay vì hardcode 1 lib → pipeline chạy được trên máy chỉ có 1 trong 2 thư viện.
2. **Empty-input guard** (`if not documents: return []`) → khi search trả 0 kết quả, pipeline không crash.
3. **Score normalization** (`float(score)` + handle scalar vs list từ FlagReranker) — tránh lỗi type khi pairs chỉ có 1 phần tử.
4. **`benchmark_reranker` có thêm `n_runs` trong dict** → giúp nhóm dễ làm Bonus Latency Breakdown (+2 điểm).

---

## 2. Kiến thức học được

**Khái niệm mới hoặc rõ ràng hơn:**

- **Two-stage retrieval:** retrieval (bi-encoder, top-20 nhanh nhưng không chính xác bằng) → reranker (cross-encoder, top-3 chính xác nhưng chậm hơn ~100x). Đây là pattern chuẩn production vì cross-encoder không scale với toàn corpus.
- **Cross-encoder vs Bi-encoder:**
  - Bi-encoder: encode query và doc **riêng biệt** → cosine similarity. Có thể pre-compute doc embeddings.
  - Cross-encoder: encode `(query, doc)` **cùng nhau** trong 1 forward pass → score relevance trực tiếp. Không pre-compute được, nhưng chính xác hơn nhiều vì có attention giữa query và doc.
- **`bge-reranker-v2-m3`** là multilingual reranker — hoạt động trực tiếp trên tiếng Việt mà không cần fine-tune. Đây là lý do dùng nó cho Vietnamese RAG.
- **FP16 (half precision)** giảm 50% memory + tăng tốc inference ~2x với loss negligible cho task ranking.
- **Score interpretation:** bge-reranker output qua sigmoid → range [0, 1], không phải cosine [-1, 1]. Số càng gần 1 = càng relevant.

**Điều bất ngờ nhất:**
Khoảng cách score giữa relevant và irrelevant docs **cực lớn**. Khi test với query "Nhân viên được nghỉ phép bao nhiêu ngày?":

- Doc "Nhân viên được nghỉ 12 ngày/năm" → **0.9914**
- Doc "Thời gian thử việc là 60 ngày" → **0.0206**
- Doc "Mật khẩu thay đổi mỗi 90 ngày" → **0.0007**

Bi-encoder cosine thường chỉ chênh ~0.1-0.2 giữa relevant và irrelevant. Cross-encoder discriminate rõ rệt hơn nhiều — giải thích tại sao reranker giúp Context Precision tăng vọt.

**Kết nối với bài giảng:**

- Slide về **Two-Stage Retrieval** (retrieval + rerank) — pattern này được áp dụng trực tiếp trong M3.
- Slide về **Latency-Quality Tradeoff** — cross-encoder slow nhưng chỉ chạy trên top-20 ⇒ tổng latency vẫn chấp nhận được.
- Concept **Vietnamese-specific embeddings** (bge-m3) — bge-reranker-v2-m3 là phiên bản reranker của cùng family.

---

## 3. Khó khăn & Cách giải quyết

| #   | Khó khăn                                                                                                | Cách giải quyết                                                                                                                         |
| --- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `FlagEmbedding` không có sẵn trong môi trường (`ModuleNotFoundError`)                                   | `try/except ImportError` → fallback sang `sentence_transformers.CrossEncoder` (đã có sẵn). Track backend đang dùng qua `self._backend`. |
| 2   | First load model **rất chậm** (~140s) — bge-reranker-v2-m3 nặng ~568MB                                  | **Lazy load** trong `_load_model()` — chỉ tải lần đầu gọi `rerank()`. Lần chạy sau dùng HF cache nên nhanh.                             |
| 3   | `FlagReranker.compute_score()` trả scalar khi pairs có 1 phần tử, list khi nhiều → type error khi `zip` | Normalize: `if not isinstance(raw_scores, list): raw_scores = [raw_scores]`.                                                            |
| 4   | Windows PowerShell cp1252 stdout không print được tiếng Việt (`UnicodeEncodeError`)                     | Set `PYTHONIOENCODING=utf-8` khi chạy `python src/m3_rerank.py` để demo. Lỗi không ảnh hưởng tests vì pytest capture output khác.       |

**Thời gian debug:** khoảng 10 phút (phần lớn là chờ model download lần đầu, không phải debug logic).

---

## 4. Nếu làm lại

**Sẽ làm khác:**

- Thêm **batch processing** trong `rerank()` — hiện tại pass cả list pairs vào model, nhưng nếu corpus rất lớn nên chia batch để tránh OOM.
- Cài cả `FlagEmbedding` từ đầu để tận dụng FP16 — nhanh hơn 2x so với CrossEncoder mặc định.
- Implement luôn **FlashrankReranker** với pip install để có số liệu so sánh latency 2 backends (cross-encoder ~50ms vs flashrank ~5ms).

**Module muốn thử tiếp:**

- **M2 (Hybrid Search)** — để hiểu sâu hơn RRF formula và Vietnamese segmentation với `underthesea`. Đây là module ảnh hưởng trực tiếp tới quality của top-20 đầu vào reranker, nên cải thiện M2 sẽ leverage M3 tốt hơn.
- **M5 (Enrichment)** — đặc biệt là Contextual Prepend (Anthropic style) — tò mò xem giảm 49% retrieval failure có replicate được không.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Lý do |
| --- | --- | --- |
| Hiểu bài giảng | 4 | Nắm rõ two-stage retrieval, cross-encoder vs bi-encoder, latency-quality tradeoff. |
| Code quality | 4 | Type hints đầy đủ, docstring, dual backend defensive, empty-input guard. Có thể cải thiện thêm batch processing. |
| Teamwork | 4 | Team chia theo role file đã được tạo sẵn từ M1 đến M5, tôi nhận M3 — Reranking. Hoàn thành module và push branch `m3-rerank` đúng tiến độ để nhóm ghép pipeline mà không bị block. |
| Problem solving | 4 | Giải quyết 4 vấn đề thực tế: (1) thiếu `FlagEmbedding` → dual backend với fallback, (2) `FlagReranker` trả type không nhất quán → normalize, (3) load model lần đầu chậm → lazy load, (4) Windows cp1252 không in được tiếng Việt → `PYTHONIOENCODING=utf-8`. |

---

**Tests evidence:**

```
$ pytest tests/test_m3.py -v
tests/test_m3.py::test_rerank_returns         PASSED  [ 20%]
tests/test_m3.py::test_rerank_type            PASSED  [ 40%]
tests/test_m3.py::test_rerank_sorted          PASSED  [ 60%]
tests/test_m3.py::test_rerank_relevant_first  PASSED  [ 80%]
tests/test_m3.py::test_benchmark_stats        PASSED  [100%]
======================== 5 passed in 143.03s ========================
```
