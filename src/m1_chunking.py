import re
import os
import base64
import time
import numpy as np
from io import BytesIO
from dataclasses import dataclass, field
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ─── CONFIG SAFE IMPORT ─────────────────────────────────

try:
    from config import (
        SEMANTIC_THRESHOLD,
        HIERARCHICAL_PARENT_SIZE,
        HIERARCHICAL_CHILD_SIZE,
        OPENROUTER_API_KEY,         # thêm vào config.py nếu có
        OCR_MODEL,                  # optional
        OCR_MIN_TEXT_LENGTH,        # ngưỡng fallback OCR
    )
except ImportError:
    SEMANTIC_THRESHOLD       = 0.75
    HIERARCHICAL_PARENT_SIZE = 2000
    HIERARCHICAL_CHILD_SIZE  = 500
    OPENROUTER_API_KEY       = os.getenv("OPENROUTER_API_KEY", "")
    OCR_MODEL                = "qwen/qwen2.5-vl-72b-instruct"
    OCR_MIN_TEXT_LENGTH      = 200  # ký tự/trang — dưới ngưỡng này → dùng OCR


# ─── CORE DATA STRUCTURE ────────────────────────────────

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


# ─── OCR CLIENT (lazy init) ─────────────────────────────

_OCR_CLIENT = None

def get_ocr_client():
    global _OCR_CLIENT
    if _OCR_CLIENT is None:
        try:
            from openai import OpenAI
            _OCR_CLIENT = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
        except ImportError:
            raise ImportError("Cần cài: pip install openai")
    return _OCR_CLIENT


# ─── OCR HELPERS ────────────────────────────────────────

def _pil_to_base64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _ocr_single_page(img, page_num: int, delay: float = 0.5) -> str:
    """Gửi 1 trang ảnh lên Qwen OCR, trả về text."""
    client = get_ocr_client()
    b64    = _pil_to_base64(img)
    prompt = (
        "Trích xuất TOÀN BỘ nội dung văn bản trong ảnh này. "
        "Giữ nguyên cấu trúc bảng biểu, số liệu, tiêu đề. "
        "Chỉ xuất text thuần, không thêm giải thích."
    )
    try:
        response = client.chat.completions.create(
            model=OCR_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    },
                    {"type": "text", "text": prompt},
                ]
            }],
            extra_headers={
                "HTTP-Referer": "https://github.com",
                "X-Title": "RAG-OCR-Pipeline",
            },
            max_tokens=4096,
        )
        time.sleep(delay)
        return response.choices[0].message.content
    except Exception as e:
        return f"[OCR ERROR page {page_num}: {e}]"


def _read_pdf_ocr(path: str, dpi: int = 200, verbose: bool = True) -> str:
    """
    Đọc PDF bằng PyMuPDF → render từng trang thành ảnh → OCR qua Qwen.
    Dùng khi PDF là bản scan hoặc pypdf extract được quá ít text.
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError:
        raise ImportError("Cần cài: pip install pymupdf pillow")

    doc    = fitz.open(path)
    pages  = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)

    doc.close()

    if verbose:
        print(f"  📄 OCR {len(pages)} trang từ: {Path(path).name}")

    extracted = []
    for i, img in enumerate(pages):
        if verbose:
            print(f"  🔍 Trang {i+1}/{len(pages)}...", end=" ", flush=True)
        text = _ocr_single_page(img, page_num=i + 1)
        extracted.append(text)
        if verbose:
            print("✅")

    return "\n\n".join(extracted)


# ─── DOCUMENT LOADER ────────────────────────────────────

def _read_pdf(path: str, use_ocr: bool = False,
              ocr_fallback: bool = True, verbose: bool = True) -> str:
    """
    Đọc PDF với 3 chế độ:
      - use_ocr=True              → luôn dùng OCR
      - ocr_fallback=True (mặc định) → thử pypdf trước, fallback OCR nếu text ít
      - use_ocr=False, ocr_fallback=False → chỉ dùng pypdf
    """
    # ── Fast path: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages  = [page.extract_text() or "" for page in reader.pages]
        pypdf_text = "\n\n".join(pages).strip()
    except Exception:
        pypdf_text = ""

    # ── Quyết định có dùng OCR không
    avg_chars_per_page = (
        len(pypdf_text) / max(len(pages), 1) if pypdf_text else 0
    )

    if use_ocr:
        # Luôn OCR
        if verbose:
            print(f"  ⚡ Chế độ OCR bắt buộc")
        return _read_pdf_ocr(path, verbose=verbose)

    if ocr_fallback and avg_chars_per_page < OCR_MIN_TEXT_LENGTH:
        # pypdf text quá ít → PDF scan → fallback OCR
        if verbose:
            print(
                f"  ⚠️  pypdf chỉ lấy được ~{int(avg_chars_per_page)} ký tự/trang "
                f"(ngưỡng: {OCR_MIN_TEXT_LENGTH}) → chuyển sang OCR"
            )
        return _read_pdf_ocr(path, verbose=verbose)

    # pypdf đủ tốt
    if verbose:
        print(f"  ✅ pypdf OK (~{int(avg_chars_per_page)} ký tự/trang)")
    return pypdf_text


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(
    paths: list[str] | None = None,
    use_ocr: bool = False,
    ocr_fallback: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Load documents từ file paths hoặc auto-scan ./data/ folder.
    Supports: .pdf, .txt, .md

    Args:
        paths:        Danh sách đường dẫn file. None = auto-scan ./data/
        use_ocr:      True → luôn dùng OCR cho PDF
        ocr_fallback: True → tự động OCR khi pypdf lấy được ít text (mặc định)
        verbose:      In tiến trình ra console
    """
    if not paths:
        data_dir = Path(__file__).parent.parent / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"data/ folder not found at: {data_dir}")
        paths = [
            str(p) for p in sorted(data_dir.iterdir())
            if p.suffix.lower() in {".pdf", ".txt", ".md"}
        ]
        if not paths:
            raise FileNotFoundError(f"No .pdf/.txt/.md files found in: {data_dir}")

    documents = []
    for path in paths:
        ext = Path(path).suffix.lower()
        if verbose:
            print(f"\n📂 Loading: {Path(path).name}")
        try:
            if ext == ".pdf":
                text = _read_pdf(
                    path,
                    use_ocr=use_ocr,
                    ocr_fallback=ocr_fallback,
                    verbose=verbose,
                )
            elif ext in {".txt", ".md"}:
                text = _read_txt(path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": path,
                        "filename": Path(path).name,
                    }
                })
        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

    return documents


# ─── BASELINE CHUNKING ──────────────────────────────────

def chunk_basic(text: str, chunk_size: int = 500, metadata=None) -> list[Chunk]:
    metadata = metadata or {}
    paragraphs = [p.strip() for p in re.split(r'\n\n|\n', text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── MODEL CACHE ────────────────────────────────────────

_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# ─── Strategy 1: Semantic ───────────────────────────────

def chunk_semantic(text: str, threshold: float = None,
                   metadata: dict | None = None) -> list[Chunk]:
    metadata  = metadata or {}
    threshold = threshold or SEMANTIC_THRESHOLD
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    model      = get_model()
    embeddings = model.encode(sentences)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    chunks, current_group = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))
    return chunks


# ─── Strategy 2: Hierarchical ───────────────────────────

def chunk_hierarchical(text: str,
                       parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None):
    metadata   = metadata or {}
    paragraphs = [p.strip() for p in re.split(r'\n\n|\n', text) if p.strip()]
    parents, children = [], []
    current, p_index  = "", 0

    for para in paragraphs:
        if len(current) + len(para) > parent_size and current:
            pid = f"parent_{p_index}"
            parents.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current  = ""
            p_index += 1
        current += para + "\n\n"

    if current.strip():
        pid = f"parent_{p_index}"
        parents.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    for parent in parents:
        pid  = parent.metadata["parent_id"]
        text = parent.text
        for i in range(0, len(text), child_size):
            child_text = text[i:i + child_size]
            if child_text.strip():
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid
                ))
    return parents, children


# ─── Strategy 3: Structure-Aware ────────────────────────

def chunk_structure_aware(text: str, metadata: dict | None = None):
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    chunks, current_header, current_content = [], "", ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
            current_header  = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_content.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))
    return chunks


# ─── Stats & Compare ────────────────────────────────────

def _compute_stats(chunks: list[Chunk]):
    if not chunks:
        return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    lengths = [len(c.text) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": int(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def compare_strategies(documents: list[dict]):
    results = {
        "basic": [], "semantic": [],
        "hierarchical_parents": [], "hierarchical_children": [],
        "structure": []
    }
    for doc in documents:
        text     = doc["text"]
        metadata = doc.get("metadata", {})
        results["basic"].extend(chunk_basic(text, metadata=metadata))
        results["semantic"].extend(chunk_semantic(text, metadata=metadata))
        parents, children = chunk_hierarchical(text, metadata=metadata)
        results["hierarchical_parents"].extend(parents)
        results["hierarchical_children"].extend(children)
        results["structure"].extend(chunk_structure_aware(text, metadata=metadata))

    return {
        "basic":    _compute_stats(results["basic"]),
        "semantic": _compute_stats(results["semantic"]),
        "hierarchical": {
            "parents":  _compute_stats(results["hierarchical_parents"]),
            "children": _compute_stats(results["hierarchical_children"]),
        },
        "structure": _compute_stats(results["structure"]),
    }
