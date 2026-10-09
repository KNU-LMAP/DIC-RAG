"""Document loading and chunking.

Only text and table content is indexed (images are not extracted).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ContentChunk:
    """A retrieved/indexed unit of the knowledge base (text or table)."""

    id: str
    content_type: str  # 'text' or 'table'
    content: Union[str, pd.DataFrame]
    page_number: int
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    description: Optional[str] = None  # text used to embed table chunks
    role_dist: Optional[Dict] = None  # sentence counts per role before compression
    role_dist_kept: Optional[Dict] = None  # sentence counts per role kept after compression

    def __post_init__(self):
        if self.content_type not in ("text", "table"):
            raise ValueError("content_type must be 'text' or 'table'")


def chunk_text_as_str(content: Union[str, pd.DataFrame]) -> str:
    """Plain-text view of a chunk's content."""
    if isinstance(content, str):
        return content
    if isinstance(content, pd.DataFrame):
        return content.to_string()
    return str(content)


class DocumentProcessor:
    """PDF -> list of ContentChunk (paragraph/sentence-aware chunking with overlap)."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, verbose: bool = False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose

    # ------------------------------------------------------------------ PDF
    def process_pdf(self, pdf_paths: List[str]) -> List[ContentChunk]:
        import pdfplumber

        chunks: List[ContentChunk] = []
        chunk_id = 0
        for pdf_path in pdf_paths:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    if self.verbose:
                        print(f"  processing page {page_num}/{total_pages} of {pdf_path}")

                    # tables first
                    for table_idx, table in enumerate(page.extract_tables() or []):
                        if table and len(table) > 1:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df = df.dropna(how="all").fillna("")
                                chunks.append(
                                    ContentChunk(
                                        id=f"chunk_{chunk_id}",
                                        content_type="table",
                                        content=df,
                                        page_number=page_num,
                                        metadata={
                                            "source": pdf_path,
                                            "table_index": table_idx,
                                            "rows": len(df),
                                            "columns": len(df.columns),
                                        },
                                        description=self._table_to_text(df),
                                    )
                                )
                                chunk_id += 1
                            except Exception as e:  # malformed table
                                print(f"  warning: could not parse table on page {page_num}: {e}")

                    # then text
                    text = page.extract_text()
                    if text:
                        for piece in self.chunk_text(text):
                            chunks.append(
                                ContentChunk(
                                    id=f"chunk_{chunk_id}",
                                    content_type="text",
                                    content=piece,
                                    page_number=page_num,
                                    metadata={"source": pdf_path},
                                )
                            )
                            chunk_id += 1
        print(f"extracted {len(chunks)} chunks from {len(pdf_paths)} PDF(s)")
        return chunks

    # ------------------------------------------------------------ chunking
    def chunk_text(self, text: str) -> List[str]:
        """Paragraph-aware chunking (splits oversized paragraphs into sentences)."""
        chunks: List[str] = []
        current = ""
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) > self.chunk_size:
                for sent in self._split_sentences(para):
                    if len(current) + len(sent) + 1 <= self.chunk_size:
                        current += sent + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent + " "
            else:
                if len(current) + len(para) + 2 <= self.chunk_size:
                    current += para + "\n\n"
                else:
                    if current:
                        chunks.append(current.strip())
                    current = para + "\n\n"
        if current:
            chunks.append(current.strip())

        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        pattern = r'(?<=[.!?。])\s+(?=[A-Z가-힣"\'])|(?<=[.!?。])\s*$'
        return [s.strip() for s in re.split(pattern, text) if s.strip()]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            tail = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            out.append(tail + " " + chunks[i])
        return out

    @staticmethod
    def _table_to_text(df: pd.DataFrame) -> str:
        """Short textual summary of a table, used only to embed table chunks."""
        parts = [
            f"Table with {len(df)} rows and {len(df.columns)} columns",
            "Columns: " + ", ".join(str(c) for c in df.columns),
            "\nFirst 2 rows:",
        ]
        for _, row in df.head(2).iterrows():
            row_text = " | ".join(f"{c}: {v}" for c, v in row.items() if v)
            if row_text:
                parts.append(row_text)
        if len(df) > 3:
            parts.append(f"\n... ({len(df) - 3} more rows) ...\n")
            parts.append("Last row:")
            row_text = " | ".join(f"{c}: {v}" for c, v in df.iloc[-1].items() if v)
            if row_text:
                parts.append(row_text)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            parts.append("\nNumeric statistics:")
            for col in numeric_cols[:3]:
                try:
                    parts.append(
                        f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
                    )
                except Exception:
                    pass
        return "\n".join(parts)
