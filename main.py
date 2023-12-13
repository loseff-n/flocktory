import argparse
import data_process

DATA_PATH = "data/"
CHUNK_SIZE = 1000


def parse_visits():
    # for dataset_type in ["train", "val", "test"]:
    for dataset_type in ["val"]:
        data_process.get_visits_dataset(DATA_PATH, CHUNK_SIZE, dataset_type=dataset_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode', 
        choices=["parse_visits"], 
        required=False, default="parse_visits"
    )
    args = parser.parse_args()
    if args.mode == "parse_visits":
        parse_visits()