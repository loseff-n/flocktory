import json
import os
import pandas as pd
from tqdm import tqdm

from utils import logger, timing

@timing
def read_json(data_path, dataset_type):
    with open(data_path+f"{dataset_type}.json", "rb") as f:
        json_object = json.load(f)

    return json_object

@timing
def parse_visits(json_object):
    # parse json
    df = pd.DataFrame()
    for j in range(len(list(json_object.keys()))):
        key = list(json_object.keys())[j]
        dfl1 = pd.DataFrame()
        if "visits" in json_object[key]["features"]:
            for i in range(len(json_object[key]["features"]["visits"])):
                n = len(pd.json_normalize(json_object[key]["features"]["visits"][i]["visits"]))
                dfl2 = pd.concat([
                        pd.concat(
                            [pd.json_normalize(json_object[key]["features"]["visits"][i])] * n, 
                            ignore_index=True
                        )[["site-id", "first-seen", "last-seen"]],
                        pd.json_normalize(json_object[key]["features"]["visits"][i]["visits"])
                    ], axis=1)
                dfl1 = pd.concat([dfl1, dfl2]).reset_index(drop=True)
        dfl1["user"] = [key] * len(dfl1)
        if "target" in json_object[key].keys():
            dfl1["target"] = json_object[key]["target"]
        df = pd.concat([df, dfl1]).reset_index(drop=True)

    df = df.explode("visited-items", ignore_index=True)
    df = df.explode("visited-general-categories", ignore_index=True)
    df = df.explode("visited-universal-brands", ignore_index=True)
    
    return df 

@timing
def parse_orders(json_object):
    # parse json
    df = pd.DataFrame()
    for j in range(len(list(json_object.keys()))):
        key = list(json_object.keys())[j]
        if "orders" in json_object[key]["features"].keys():
            dfl2 = pd.DataFrame()
            for n in range(len(json_object[key]["features"]["orders"])):
                l = json_object[key]["features"]["orders"][n]
                dfl1 = pd.DataFrame()
                for i in range(len(l["orders"])):
                    k = l["orders"][i]
                    dfl0 = pd.DataFrame(k["items"])
                    dfl0["created-at"] = k["created-at"]
                    dfl1 = pd.concat([dfl1, dfl0]).reset_index(drop=True)
                dfl1["site-id"] = l["site-id"]
                dfl2 = pd.concat([dfl2, dfl1]).reset_index(drop=True)
                dfl2["user"] = key
                if "target" in json_object[key]:
                    dfl2["target"] = json_object[key]["target"]
            df = pd.concat([df, dfl2])

    return df


def iterate_json(json_object, chunk_size):
    items = list(json_object.items())
    for i in range(0, len(items), chunk_size):
        yield {k:v for k,v in items[i:i+chunk_size]}

@timing
def get_visits_dataset(data_path, chunk_size, dataset_type="train"):
    logger.info(f"reading {dataset_type} json...")
    # read as json
    json_object = read_json(data_path, dataset_type)
    # json_object = dict(list(json_object.items())[:50])

    logger.info(f"parsing visits started...")
    for idx, chunk in enumerate(iterate_json(json_object, chunk_size)):
        # get chunk idx
        chunk_users = sorted([int(x[5:]) for x in list(chunk.keys())])
        chunk_users = f"{chunk_users[0]}_{chunk_users[-1]}"

        # parse
        df = parse_visits(chunk)

        # save as parquet by chunks
        df.to_parquet(
            data_path+f"visits/{dataset_type}/{idx}_{chunk_users}_{dataset_type}.parquet.gzip",
            compression="gzip",
            index=False
        )
        logger.info(f"{(idx+1)*chunk_size}/{len(json_object)}")
    logger.info(f"...collecting visits dataset completed")

def get_orders_dataset(data_path, chunk_size, dataset_type="train"):
    logger.info(f"reading {dataset_type} json...")
    # read as json
    json_object = read_json(data_path, dataset_type)
    # json_object = dict(list(json_object.items())[:50])

    logger.info(f"parsing orders started...")
    for idx, chunk in enumerate(iterate_json(json_object, chunk_size)):
        # get chunk idx
        chunk_users = sorted([int(x[5:]) for x in list(chunk.keys())])
        chunk_users = f"{chunk_users[0]}_{chunk_users[-1]}"

        # parse
        df = parse_orders(chunk)
        # save as parquet by chunks
        df.to_parquet(
            data_path+f"orders/{dataset_type}/{idx}_{chunk_users}_{dataset_type}.parquet.gzip",
            compression="gzip",
            index=False
        )
        logger.info(f"{(idx+1)*chunk_size}/{len(json_object)}")
    logger.info(f"...collecting orders dataset completed")

@timing
def read_visits_data(data_path, dataset_type="train"):
    logger.info(f"reading {dataset_type} dataset...")
    clms = ["site-id", "session-duration", "pages-count", "target"]
    df = pd.DataFrame()
    for f in os.listdir(data_path+f"visits/{dataset_type}/"):
        dfs = pd.read_parquet(data_path+f"visits/{dataset_type}/{f}")
        dfs = dfs[clms]
        dfs = dfs.drop_duplicates().reset_index(drop=True)
        df = pd.concat([df,dfs]).reset_index(drop=True)

    # categorical features to str
    df["site-id"] = df["site-id"].astype(int).astype(str)

    # convert target to int
    df.target = (df.target == "male").astype(int)

    return df

@timing
def read_orders_data(data_path, dataset_type="train"):
    logger.info(f"reading {dataset_type} dataset...")
    clms = ["id", "count", "brand-id", "site-id", "target"]
    df = pd.DataFrame()
    for f in os.listdir(data_path+f"orders/{dataset_type}/"):
        dfs = pd.read_parquet(data_path+f"orders/{dataset_type}/{f}")
        dfs = dfs[clms]
        # dfs = dfs.drop_duplicates().reset_index(drop=True)
        df = pd.concat([df,dfs]).reset_index(drop=True)

    # categorical features to str
    df[["brand-id", "site-id"]] = df[["brand-id", "site-id"]].fillna(-1)
    df[["brand-id", "site-id"]] = df[["brand-id", "site-id"]].astype(int).astype(str)

    # convert target to int
    df.target = (df.target == "male").astype(int)

    return df
