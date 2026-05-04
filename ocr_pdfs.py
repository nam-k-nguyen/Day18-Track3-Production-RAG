"""Process PDFs to Markdown: direct extract for text PDFs, GPT-4o for table-heavy PDFs."""

import base64
import re
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def extract_pdf(pdf_path: str, output_md: str) -> None:
    """Direct text extraction for digital PDFs (no tables). Fast, free, accurate."""
    doc = fitz.open(pdf_path)
    n = len(doc)
    print(f"Extracting {Path(pdf_path).name} ({n} pages)...")

    all_pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        text = _to_markdown(text)
        all_pages.append(text)
        print(f"  Page {i+1}/{n}: {len(text)} chars")

    doc.close()

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_pages))
    print(f"  Saved -> {output_md}\n")


def extract_pdf_with_tables(pdf_path: str, output_md: str) -> None:
    """Extract digital PDF using PyMuPDF table detection. Best for complex table layouts."""
    doc = fitz.open(pdf_path)
    n = len(doc)
    print(f"Extracting (with tables) {Path(pdf_path).name} ({n} pages)...")

    all_pages = []
    for i, page in enumerate(doc):
        blocks = []
        # Find tables on this page
        tables = page.find_tables()
        table_bboxes = []

        for table in tables:
            table_bboxes.append(table.bbox)
            md_rows = []
            for r, row in enumerate(table.extract()):
                cells = [str(c).strip() if c else "" for c in row]
                md_rows.append("| " + " | ".join(cells) + " |")
                if r == 0:
                    md_rows.append("|" + "|".join(["---"] * len(cells)) + "|")
            blocks.append((table.bbox[1], "\n".join(md_rows)))

        # Get text blocks not inside any table
        for block in page.get_text("blocks"):
            bx0, by0, bx1, by1, text, *_ = block
            in_table = any(
                bx0 >= tb[0] - 2 and by0 >= tb[1] - 2 and bx1 <= tb[2] + 2 and by1 <= tb[3] + 2
                for tb in table_bboxes
            )
            if not in_table and text.strip():
                blocks.append((by0, _to_markdown(text.strip())))

        blocks.sort(key=lambda x: x[0])
        page_md = "\n\n".join(b[1] for b in blocks)
        all_pages.append(page_md)
        print(f"  Page {i+1}/{n}: {len(page_md)} chars")

    doc.close()

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_pages))
    print(f"  Saved -> {output_md}\n")


def _to_markdown(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append("")
            continue
        if re.match(r"^Chương\s+[IVXLCDM\d]+", s):
            out.append(f"# {s}")
        elif re.match(r"^Điều\s+\d+[a-z]?\.", s):
            out.append(f"## {s}")
        elif re.match(r"^Mục\s+\d+", s):
            out.append(f"### {s}")
        else:
            out.append(s)
    return "\n".join(out)


def ocr_pdf(pdf_path: str, output_md: str, dpi: int = 250, model: str = "gpt-4o-mini") -> None:
    """GPT-4o Vision OCR — works for both scanned images and font-encoded PDFs."""
    doc = fitz.open(pdf_path)
    n = len(doc)
    print(f"OCR {Path(pdf_path).name} ({n} pages) via {model}...")

    all_text = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()

        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Trích xuất toàn bộ văn bản trong ảnh này dưới dạng Markdown. Dùng # ## cho tiêu đề, bảng Markdown (| col |) cho bảng số liệu, giữ nguyên cấu trúc. Không bọc trong code block, không thêm bình luận.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }],
            max_tokens=4096,
        )
        page_text = resp.choices[0].message.content.strip()
        all_text.append(page_text)
        print(f"  Page {i+1}/{n}: {len(page_text)} chars")

    doc.close()

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_text))
    print(f"  Saved -> {output_md}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", choices=["bctc", "nd13", "all"], default="all")
    args = parser.parse_args()

    data_dir = Path("data")
    if args.file in ("bctc", "all"):
        # BCTC: 2-page scan with complex tables → gpt-4o for best table accuracy
        ocr_pdf(str(data_dir / "BCTC.pdf"),
                str(data_dir / "BCTC.md"),
                model="gpt-4o")
    if args.file in ("nd13", "all"):
        # Nghị định: 39 pages, font encoding issue → gpt-4o-mini (cheaper, text only)
        ocr_pdf(str(data_dir / "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf"),
                str(data_dir / "nd13_2023.md"),
                model="gpt-4o-mini")
    print("Done. Run pipeline.py next.")
