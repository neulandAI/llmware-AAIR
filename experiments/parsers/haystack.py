from .base import Parser


class HaystackParser(Parser):
    def __init__(self):
        super().__init__()
        # implement the setup for haystack here.
        # the parsers will be initialized only once for a run.

    def parse_pdf(self, sample: dict):
        pass

        # implement the parsing logic here.
