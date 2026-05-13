"""source_text_reader 单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.source_text_reader import read_source_text


def test_read_source_text_utf8_txt(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("你好\nline2", encoding="utf-8")
    assert read_source_text(f) == "你好\nline2"


def test_read_pdf_text_mocked(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-fake")

    page1 = MagicMock()
    page1.extract_text.return_value = "第一段"
    page2 = MagicMock()
    page2.extract_text.return_value = "第二段"
    reader = MagicMock()
    reader.pages = [page1, page2]

    with patch("pypdf.PdfReader", return_value=reader):
        out = read_source_text(pdf)

    assert "第一段" in out and "第二段" in out
    page1.extract_text.assert_called_once()
    page2.extract_text.assert_called_once()
