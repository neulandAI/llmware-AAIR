"""
This is model based. Maybe if we have time we can get these methods
as well.
"""

from .base import Parser


class DolphinParser(Parser):
    def __init__(self):
        super().__init__()
        # implement the setup for dolphin here.
        # the parsers will be initialized only once for a run.

    def parse_pdf(self, sample: dict):
        pass

        # implement the parsing logic here.
