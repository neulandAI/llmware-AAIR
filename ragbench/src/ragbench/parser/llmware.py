from .base import Parser


class LlmwareParser(Parser):
    def __init__(self):
        super().__init__()
        # implement the setup for llmware parser here.
        # the parsers will be initialized only once for a run.

    def parse_pdf(self, sample: dict):
        pass

        # implement the parsing logic here.
