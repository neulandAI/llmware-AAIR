import itertools

from ragbench.runner import Runner
from ragbench.parser import
from ragbench.encoder import



def setup():
    pass


def main():
    setup()
    sweep_config = {
        parsers: [
            RagFlowParser(),
            UnstructuredParser(),
        ],
        encoders: [
            QwenEmbedding4B(),
            QwenEmbedding8B(),
            Voyage3Large(),
            OctenEmbedding4B(),
            OctenEmbedding8B(),
        ],
    }

    sweep_attrib = sweep_config.keys()
    sweep_values = sweep_config.values()

    for current_values in itertools.product(*sweep_values):
        current_params = dict(zip(sweep_attrib, current_values))
        print(f"Running for config:\n{current_params}")

        evaluate(**current_params)


if __name__ == "__main__":
    main()
