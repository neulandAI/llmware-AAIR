from .base import Parser


class LlamaIndexParser(Parser):
    def __init__(self):
        super().__init__()
        # implement the setup for llamaindex parser here.
        # the parsers will be initialized only once for a run.

    def parse_pdf(self, sample: dict):
        pass

        # implement the parsing logic here.
