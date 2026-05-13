"""从支持的源文件路径读取纯文本（供分割与向量化）。"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from app.config import config


def read_source_text(path: Path) -> str:
    """按扩展名读取文件为 UTF-8 文本。支持 .txt / .md（UTF-8）与 .pdf（pypdf 抽取）。"""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_text(path)
    return path.read_text(encoding="utf-8")


def read_pdf_text(path: Path) -> str:
    """用 pypdf 抽取 PDF 全文（多页拼接）。扫描件/图片型 PDF 可能几乎无字。"""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as e:
        raise RuntimeError("缺少依赖 pypdf，无法解析 PDF。请安装: pip install pypdf") from e

    try:
        reader = PdfReader(str(path), strict=False)
    except PdfReadError as e:
        raise ValueError(f"无法解析 PDF: {path}: {e}") from e

    max_pages = max(1, config.pdf_max_pages)
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            logger.warning("PDF 页数超过 pdf_max_pages=%s，已截断: %s", max_pages, path)
            break
        try:
            t = page.extract_text()
        except Exception as ex:  # noqa: BLE001
            logger.warning("PDF 第 %s 页抽取失败: %s", i + 1, ex)
            t = ""
        parts.append(t.strip() if t else "")

    out = "\n\n".join(parts).strip()
    lim = max(1024, config.pdf_max_extract_chars)
    if len(out) > lim:
        logger.warning("PDF 抽取文本超过 pdf_max_extract_chars=%s，已截断: %s", lim, path)
        out = out[:lim]
    if not out:
        logger.warning("PDF 未抽取到可见文本（可能为扫描件）: %s", path)
    return out
