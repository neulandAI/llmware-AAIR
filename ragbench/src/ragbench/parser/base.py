from abc import ABC, abstractmethod
from pathlib import Path


class Parser(ABC):
    def parse_pdf(self, sample: dict):
        pass

    @abstractmethod
    def parse_one(self, sample: dict):
        extension = Path(sample["file_path"]).suffix[1:]
        method_nm = f"parse_{extension}"

        assert hasattr(self, method_nm), f"Parser for {extension} not implemented."
        method = getattr(self, method_nm)
        assert callable(method), f"Method {method_nm} is not callable."

        return method(sample)
