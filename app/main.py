import argparse

import data_process
import train

DATA_PATH = "data/"
CHUNK_SIZE = 1000

def parse_visits():
    for dataset_type in ["train", "val", "test"]:
        data_process.get_visits_dataset(DATA_PATH, CHUNK_SIZE, dataset_type=dataset_type)

def parse_orders():
    for dataset_type in ["train", "val", "test"]:
        data_process.get_orders_dataset(DATA_PATH, CHUNK_SIZE, dataset_type=dataset_type)

def parse_meta():
    for dataset_type in ["train", "val", "test"]:
        data_process.get_meta_dataset(DATA_PATH, CHUNK_SIZE, dataset_type=dataset_type)

def parse_accepted():
    for dataset_type in ["train", "val", "test"]:
        data_process.get_accepted_dataset(DATA_PATH, CHUNK_SIZE, dataset_type=dataset_type)

def catboost_fit_visits():
    model = train.catboost_fit(DATA_PATH, ["site-id"], "visits")
    df_val = data_process.read_visits_data(DATA_PATH, dataset_type="val")
    metrics = train.calc_metrics(model, df_val)

def catboost_fit_orders():
    model = train.catboost_fit(DATA_PATH, ["id", "brand-id", "site-id"], "orders")
    # model = train.catboost_fit(DATA_PATH, ["id", "brand-id", "site-id", "general-category-path"], "orders")
    df_val = data_process.read_orders_data(DATA_PATH, dataset_type="val")
    metrics = train.calc_metrics(model, df_val)

def catboost_fit_orders_scores():
    # model = train.catboost_fit(DATA_PATH, ["id", "brand-id", "site-id"], "orders")
    model = train.catboost_fit(DATA_PATH, [], "orders")
    df_val = data_process.read_orders_data_scores_val(DATA_PATH)
    metrics = train.calc_metrics(model, df_val)

def catboost_fit_meta():
    model = train.catboost_fit(DATA_PATH, ["site-id"], "meta")
    df_val = data_process.read_meta_data(DATA_PATH, dataset_type="val")
    metrics = train.calc_metrics(model, df_val)

def logreg_fit_meta():
    model = train.logreg_fit(DATA_PATH)
    df_val = data_process.get_logreg_val_data_meta(DATA_PATH)
    df_val = df_val[["sex_score_meta", "target"]]
    metrics = train.calc_metrics(model, df_val)

def logreg_fit_both():
    model = train.logreg_fit(DATA_PATH, two_feats="both")
    df_val = data_process.get_logreg_val_data_both(DATA_PATH)
    df_val = df_val[["sex_score_meta", "sex_score_accepted", "target"]]
    metrics = train.calc_metrics(model, df_val)

def logreg_fit_brands():
    model = train.logreg_fit(DATA_PATH, two_feats="brands")
    df_val = data_process.get_logreg_val_data_brands(DATA_PATH)
    df_val = df_val[["sex_score_meta", "sex_score_brands", "target"]]
    metrics = train.calc_metrics(model, df_val)

def logreg_predict():
    model = train.logreg_predict(DATA_PATH)

def switch(mode):
    if mode == "parse_visits":
        parse_visits()
    elif mode == "parse_orders":
        parse_orders()
    elif mode == "parse_meta":
        parse_meta()
    elif mode == "parse_accepted":
        parse_accepted()
    elif mode == "catboost_fit_visits":
        catboost_fit_visits()
    elif mode == "catboost_fit_orders":
        catboost_fit_orders()
    elif mode == "catboost_fit_orders_scores":
        catboost_fit_orders_scores()
    elif mode == "catboost_fit_meta":
        catboost_fit_meta()
    elif mode == "logreg_fit_meta":
        logreg_fit_meta()
    elif mode == "logreg_fit_both":
        logreg_fit_both()
    elif mode == "logreg_fit_brands":
        logreg_fit_brands()
    elif mode == "logreg_predict":
        logreg_predict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode', 
        choices=[
                # parse modes
                "parse_visits", "parse_orders", "parse_meta", "parse_accepted",
                # experiment fit modes
                 "catboost_fit_visits", "catboost_fit_orders", "catboost_fit_orders_scores",
                 "catboost_fit_meta", "logreg_fit_meta", "logreg_fit_both", "logreg_fit_brands",
                # predict modes
                 "logreg_predict"
                 ], 
        required=False, default="catboost_fit_orders"
    )
    args = parser.parse_args()
    switch(args.mode)
