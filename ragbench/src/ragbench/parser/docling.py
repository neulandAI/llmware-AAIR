from .base import Parser


class DoclingParser(Parser):
    def __init__(self):
        super().__init__()
        # implement the setup for docling here.
        # the parsers will be initialized only once for a run.

    def parse_pdf(self, sample: dict):
        pass

        # implement the parsing logic here.

    def parse_txt(self, sample: dict):
        pass
