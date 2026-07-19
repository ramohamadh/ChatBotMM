"""
Document ingestion module.
Supports loading and extracting text from PDF, DOCX, TXT, and MD files.
"""

import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

import docx
import pypdf

logger = logging.getLogger(__name__)

# --- visual-order RTL repair (for pdfplumber table cells) --------------------
# pdfplumber/pdfminer return Persian glyphs in *visual* order (left-to-right
# on screen = reversed logical order) using Arabic presentation-form
# codepoints. NFKC restores the base letters; reversing restores the logical
# order; Latin/digit runs must then be re-reversed back.
_RTL_CHARS = re.compile(r"[֐-ࣿﭐ-ﻼ]")
_LTR_RUN = re.compile(r"[A-Za-z0-9_@#./\\:%+*=-]+")
_BRACKET_SWAP = str.maketrans("()[]{}<>", ")(][}{><")


def _fix_visual_rtl(text: str) -> str:
    """Convert visually-ordered RTL text (pdfminer output) to logical order."""
    if not text or not _RTL_CHARS.search(text):
        return text  # pure LTR — already correct
    text = unicodedata.normalize("NFKC", text)
    reversed_text = text[::-1].translate(_BRACKET_SWAP)
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1].translate(_BRACKET_SWAP), reversed_text)


class DocumentIngester:
    """Handles ingestion of various document formats."""

    def __init__(self, supported_extensions: set | None = None):
        """
        Initialize the document ingester.

        Args:
            supported_extensions: Set of file extensions to support.
                                 Defaults to {'.pdf', '.docx', '.txt', '.md'}
        """
        self.supported_extensions = supported_extensions or {".pdf", ".docx", ".txt", ".md"}

    def is_supported(self, file_path: Path) -> bool:
        """Check if a file format is supported."""
        return file_path.suffix.lower() in self.supported_extensions

    def load_document(self, file_path: Path) -> list[tuple[str, dict]]:
        """
        Load a document and extract text with metadata.

        Args:
            file_path: Path to the document file

        Returns:
            List of (text, metadata_dict) tuples. For PDFs this will contain one entry
            per page, while other formats will typically return a single entry.

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_supported(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loading document: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix in {".txt", ".md"}:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_pdf(self, file_path: Path) -> list[tuple[str, dict]]:
        """Load text from a PDF file, returning one entry per page."""
        page_entries: list[tuple[str, dict]] = []

        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception as page_error:
                        logger.warning(f"Failed to extract text from {file_path} page {page_num + 1}: {page_error}")
                        page_text = ""

                    page_text = page_text.strip()
                    if not page_text:
                        continue

                    page_metadata = {
                        "filename": file_path.name,
                        "file_path": str(file_path),
                        "file_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": total_pages,
                        "source_id": f"{file_path.name}::page_{page_num + 1}"
                    }

                    page_entries.append((page_text, page_metadata))

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise

        # Headers/footers repeat on most pages and poison every chunk with the
        # same noise; drop them before chunking.
        page_entries = self._strip_repeated_lines(page_entries)

        # Tables: each row becomes a self-contained sentence, which retrieves
        # and reads far better than the flattened table text.
        page_entries.extend(self._extract_tables(file_path))

        if not page_entries:
            logger.warning(f"No text extracted from PDF: {file_path}")

        return page_entries

    @staticmethod
    def _strip_repeated_lines(
        page_entries: list[tuple[str, dict]], min_ratio: float = 0.6
    ) -> list[tuple[str, dict]]:
        """Drop header/footer lines that repeat on most pages.

        Lines are compared with digits collapsed so "صفحه 12 از 143" and
        "صفحه 13 از 143" count as the same repeated line.
        """
        if len(page_entries) < 4:
            return page_entries

        def normalize(line: str) -> str:
            return re.sub(r"\d+", "#", line.strip())

        appearances: Counter = Counter()
        for text, _ in page_entries:
            for line in {normalize(ln) for ln in text.splitlines() if ln.strip()}:
                appearances[line] += 1

        threshold = max(3, int(len(page_entries) * min_ratio))
        repeated = {line for line, count in appearances.items() if count >= threshold}
        if not repeated:
            return page_entries

        cleaned: list[tuple[str, dict]] = []
        for text, metadata in page_entries:
            kept = [ln for ln in text.splitlines() if normalize(ln) not in repeated]
            new_text = "\n".join(kept).strip()
            if new_text:
                cleaned.append((new_text, metadata))
        logger.info(f"Stripped {len(repeated)} repeated header/footer line(s)")
        return cleaned

    def _extract_tables(self, file_path: Path) -> list[tuple[str, dict]]:
        """Extract tables via pdfplumber; each row becomes one sentence.

        Returns one (text, metadata) entry per page that has tables. Skipped
        silently if pdfplumber is unavailable.
        """
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed — table extraction skipped")
            return []

        entries: list[tuple[str, dict]] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    sentences: list[str] = []
                    for table in page.extract_tables() or []:
                        sentences.extend(self._table_to_sentences(table))
                    if not sentences:
                        continue
                    entries.append(
                        (
                            "\n".join(sentences),
                            {
                                "filename": file_path.name,
                                "file_path": str(file_path),
                                "file_type": "pdf",
                                "page_number": page_num,
                                "total_pages": total_pages,
                                "content_type": "table",
                                "source_id": f"{file_path.name}::page_{page_num}::tables",
                            },
                        )
                    )
        except Exception as e:  # noqa: BLE001 - tables are an enhancement, not critical
            logger.warning(f"Table extraction failed for {file_path}: {e}")
            return []

        if entries:
            logger.info(f"Extracted tables from {len(entries)} page(s) of {file_path.name}")
        return entries

    @staticmethod
    def _table_to_sentences(table: list[list]) -> list[str]:
        """Turn one extracted table into self-contained row sentences.

        With a usable header row, each data row becomes
        "header1: cell1؛ header2: cell2؛ …" so a row can answer a question on
        its own, without the surrounding table.
        """
        rows = [
            [_fix_visual_rtl((cell or "").replace("\n", " ").strip()) for cell in row]
            for row in (table or [])
        ]
        rows = [row for row in rows if sum(1 for cell in row if cell) >= 2]
        if len(rows) < 2:
            return []

        header = rows[0]
        use_header = sum(1 for cell in header if cell) >= 2

        sentences = []
        for row in rows[1:]:
            if use_header:
                parts = [f"{h}: {c}" for h, c in zip(header, row, strict=False) if h and c]
            else:
                parts = [c for c in row if c]
            if not parts:
                continue
            sentence = "؛ ".join(parts)
            # Skip decorative/garbled rows: mostly single-character tokens.
            tokens = sentence.split()
            if tokens and sum(1 for t in tokens if len(t) == 1) / len(tokens) > 0.5:
                continue
            sentences.append(sentence)
        return sentences

    def _load_docx(self, file_path: Path) -> list[tuple[str, dict]]:
        """Load text from a DOCX file."""
        try:
            doc = docx.Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()

            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": "docx",
                "paragraph_count": len(paragraphs)
            }

        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise

        if not full_text:
            logger.warning(f"No text extracted from DOCX: {file_path}")
            return []

        if not full_text.strip():
            logger.warning(f"No text extracted from file: {file_path}")
            return []

        return [(full_text, metadata)]

    def _load_text(self, file_path: Path) -> list[tuple[str, dict]]:
        """Load text from a TXT or MD file."""
        try:
            with open(file_path, encoding="utf-8") as file:
                full_text = file.read()

            # Count lines and sections (markdown headers)
            lines = full_text.split("\n")
            section_count = len([line for line in lines if line.strip().startswith("#")])

            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "line_count": len(lines),
                "section_count": section_count if file_path.suffix == ".md" else 0
            }

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise

        return [(full_text, metadata)]

    def load_directory(self, directory: Path) -> list[tuple[str, dict]]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to directory containing documents

        Returns:
            List of (text, metadata) tuples for each document
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        documents: list[tuple[str, dict]] = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and self.is_supported(file_path):
                try:
                    entries = self.load_document(file_path)
                    documents.extend(entries)
                    logger.info(f"Successfully loaded: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} document sections from {directory}")
        return documents

