import argparse

import data_process
import train

DATA_PATH = "data/"
CHUNK_SIZE = 1000


def parse_visits():
    # for dataset_type in ["train", "val", "test"]:
    for dataset_type in ["test"]:
        data_process.get_visits_dataset(DATA_PATH, CHUNK_SIZE, dataset_type=dataset_type)

def parse_orders():
    for dataset_type in ["train", "val", "test"]:
        data_process.get_orders_dataset(DATA_PATH, CHUNK_SIZE, dataset_type=dataset_type)

def catboost_fit_visits():
    model = train.catboost_fit(DATA_PATH, ["site-id"], "visits")
    df_val = data_process.read_visits_data(DATA_PATH, dataset_type="val")
    metrics = train.evaluate_model(model, df_val)

def catboost_fit_orders():
    model = train.catboost_fit(DATA_PATH, ["id", "brand-id", "site-id"], "orders")
    df_val = data_process.read_orders_data(DATA_PATH, dataset_type="val")
    metrics = train.evaluate_model(model, df_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode', 
        choices=["parse_visits", "parse_orders", 
                 "catboost_fit_visits"], 
        required=False, default="catboost_fit_orders"
    )
    args = parser.parse_args()
    if args.mode == "parse_visits":
        parse_visits()
    elif args.mode == "parse_orders":
        parse_orders()
    elif args.mode == "catboost_fit_visits":
        catboost_fit_visits()
    elif args.mode == "catboost_fit_orders":
        catboost_fit_orders()