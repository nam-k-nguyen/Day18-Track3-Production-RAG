"""OCR scanned PDFs using GPT-4o Vision via PyMuPDF page rendering."""

import base64
import os
import sys
from pathlib import Path

import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def ocr_pdf(pdf_path: str, output_txt: str, dpi: int = 150) -> None:
    doc = fitz.open(pdf_path)
    n = len(doc)
    print(f"Processing {Path(pdf_path).name} ({n} pages)...")

    all_text = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
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

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    print(f"  Saved -> {output_txt}\n")


if __name__ == "__main__":
    data_dir = Path("data")
    ocr_pdf(str(data_dir / "BCTC.pdf"),
            str(data_dir / "BCTC.md"))
    ocr_pdf(str(data_dir / "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf"),
            str(data_dir / "nd13_2023.md"))
    print("Done. Run pipeline.py next.")
