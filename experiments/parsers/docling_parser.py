from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .base import Parser


class DoclingParser(Parser):
    parser_name = "docling"

    def __init__(self):
        super().__init__()
        self.pipeline_options = PdfPipelineOptions()

    def parse_pdf(self, file: str) -> list[dict]:
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
        self.pipeline_options.ocr_options = ocr_options

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )

        doc = converter.convert(file).document
        file_name = Path(file).name
        blocks = []

        for item in doc.iterate_items():
            element = item[0]
            page_num = self._get_page_num(element)
            is_table = type(element).__name__ == "TableItem"

            if is_table:
                text = element.export_to_markdown(doc) if hasattr(element, 'export_to_markdown') else str(element)
            elif hasattr(element, 'text'):
                text = element.text
            else:
                continue

            if text and text.strip():
                blocks.append(self.create_block(
                    text=text.strip(),
                    page_num=page_num,
                    block_idx=len(blocks),
                    file_source=file_name,
                    is_table=is_table,
                ))

        return blocks

    def _get_page_num(self, element) -> int:
        """Extract page number from element (1-indexed)."""
        try:
            if hasattr(element, 'prov') and element.prov:
                prov = element.prov[0] if isinstance(element.prov, list) else element.prov
                if hasattr(prov, 'page_no'):
                    return prov.page_no
        except (IndexError, AttributeError):
            pass
        return 1

if __name__ == "__main__":
    file_path = '/Users/itorky/Desktop/Phd Proposal/PhD Presentation (2nd Follow Up).pdf'
    
    parser = DoclingParser()
    blocks = parser.parse_pdf(file_path)
    
    print(blocks)
