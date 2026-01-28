from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


_LLMWARE_DEFAULTS = {
    "doc_ID": 0,
    "master_index2": 0,
    "coords_x": 0,
    "coords_y": 0,
    "coords_cx": 0,
    "coords_cy": 0,
    "author_or_speaker": "",
    "modified_date": "",
    "created_date": "",
    "external_files": "",
    "header_text": "",
    "user_tags": "",
    "special_field1": "",
    "special_field2": "",
    "special_field3": "",
    "graph_status": "false",
    "dialog": "false",
    "embedding_flags": {},
}


class Parser(ABC):
    parser_name: str = "base"

    @abstractmethod
    def parse_pdf(self, file: str) -> list[dict]:
        """
        Parse PDF and return llmware-compatible blocks.
        Subclasses implement this using create_block() helper.
        """
        pass

    def create_block(
        self,
        text: str,
        page_num: int,
        block_idx: int,
        file_source: str,
        is_table: bool = False,
    ) -> dict:
        """
        Create an llmware-compatible block.
        
        Args:
            text:        The extracted text content
            page_num:    Page number (1-indexed)
            block_idx:   Chunk number within document (0, 1, 2, ...)
            file_source: Original filename (e.g. "report.pdf")
            is_table:    Set True if this block is a table
        """
        return {
            "text": text,
            "master_index": page_num,
            "block_ID": block_idx,
            "file_source": file_source,
            "content_type": "table" if is_table else "text",
            "table": text if is_table else "",
            "text_search": text,
            "file_type": "pdf",
            "creator_tool": self.parser_name,
            "added_to_collection": datetime.now().isoformat(),
            # llmware defaults
            **_LLMWARE_DEFAULTS,
        }
