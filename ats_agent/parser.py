"""Extract text (and parsability signals) from PDF, DOCX, and plain-text resumes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedResume:
    text: str
    file_type: str
    pages: int = 1
    # Parsability signals that real ATS parsers tend to struggle with.
    table_count: int = 0
    image_count: int = 0
    header_footer_text: str = ""
    multi_column_suspected: bool = False
    warnings: list[str] = field(default_factory=list)


def parse_resume(path: str | Path) -> ParsedResume:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix in {".txt", ".md", ""}:
        return ParsedResume(text=path.read_text(encoding="utf-8", errors="replace"), file_type="txt")
    raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, DOCX, or TXT.")


def _parse_pdf(path: Path) -> ParsedResume:
    import pdfplumber

    texts: list[str] = []
    tables = images = 0
    multi_col = False
    with pdfplumber.open(path) as pdf:
        pages = len(pdf.pages)
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
            tables += len(page.find_tables())
            images += len(page.images)
            multi_col = multi_col or _looks_multi_column(page)
    text = "\n".join(texts).strip()
    parsed = ParsedResume(
        text=text, file_type="pdf", pages=pages, table_count=tables,
        image_count=images, multi_column_suspected=multi_col,
    )
    if len(text) < 200:
        parsed.warnings.append(
            "Very little text could be extracted. The PDF may be scanned or image-based, "
            "which most ATS systems cannot read."
        )
    return parsed


def _looks_multi_column(page) -> bool:
    """Heuristic: many words start near the middle of the page on lines that also have words on the left."""
    words = page.extract_words() or []
    if len(words) < 40:
        return False
    mid = page.width / 2
    rows: dict[int, list[float]] = {}
    for w in words:
        rows.setdefault(round(w["top"] / 3), []).append(w["x0"])
    split_rows = 0
    for xs in rows.values():
        xs.sort()
        left = [x for x in xs if x < mid * 0.8]
        right = [x for x in xs if x > mid * 1.05]
        if left and right and (min(right) - max(left)) > page.width * 0.15:
            split_rows += 1
    return split_rows > len(rows) * 0.35


def _parse_docx(path: Path) -> ParsedResume:
    import docx

    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:  # include table text so scoring still sees it
        for row in table.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    hf: list[str] = []
    for section in doc.sections:
        for container in (section.header, section.footer):
            hf.extend(p.text for p in container.paragraphs if p.text.strip())
    images = len(doc.inline_shapes)
    multi_col = False
    for section in doc.sections:
        cols = section._sectPr.xpath("./w:cols")
        if cols and int(cols[0].get(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}num", "1"
        ) or 1) > 1:
            multi_col = True
    text = "\n".join(parts).strip()
    words = len(text.split())
    return ParsedResume(
        text=text, file_type="docx", pages=max(1, round(words / 500)),
        table_count=len(doc.tables), image_count=images,
        header_footer_text="\n".join(hf), multi_column_suspected=multi_col,
    )
