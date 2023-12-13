import json
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

def iterate_json(json_object, chunk_size):
    items = list(json_object.items())
    for i in range(0, len(items), chunk_size):
        yield {k:v for k,v in items[i:i+chunk_size]}

#timing
def get_visits_dataset(data_path, chunk_size, dataset_type="train"):
    logger.info(f"reading {dataset_type} json...")
    # read as json
    json_object = read_json(data_path, dataset_type)
    # json_object = dict(list(json_object.items())[:50])

    for idx, chunk in enumerate(iterate_json(json_object, chunk_size)):
        # get chunk idx
        chunk_users = sorted([int(x[5:]) for x in list(chunk.keys())])
        chunk_users = f"{chunk_users[0]}_{chunk_users[-1]}"

        df = parse_visits(chunk)

        # save as parquet by chunks
        df.to_parquet(
            data_path+f"{dataset_type}/{idx}_{chunk_users}_{dataset_type}.parquet.gzip",
            compression="gzip",
            index=False
        )
        logger.info(f"{(idx+1)*chunk_size}/{len(json_object)}")