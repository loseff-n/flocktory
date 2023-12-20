import json
import os
import pandas as pd
import pickle
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

@timing
def parse_meta(json_object):
    # parse json
    df = pd.DataFrame()
    for j in range(len(list(json_object.keys()))):
        key = list(json_object.keys())[j]
        if "site-meta" in json_object[key]["features"]:
            dfl1 = pd.DataFrame(json_object[key]["features"]['site-meta'])
            dfl1["user"] = key
            if "target" in json_object[key]:
                dfl1["target"] = json_object[key]["target"]
            df = pd.concat([df, dfl1]).reset_index(drop=True)

    return df

@timing
def parse_accepted(json_object):
    df = pd.DataFrame()
    for j in range(len(list(json_object.keys()))):
        key = list(json_object.keys())[j]
        if "exchange-sessions" in json_object[key]["features"]:
            if "accepted-site-id" in pd.DataFrame(json_object[key]["features"]['exchange-sessions']).columns:
                dfl1 = pd.DataFrame(json_object[key]["features"]['exchange-sessions'])["accepted-site-id"].dropna().to_frame()
                dfl1["user"] = key
                if "target" in json_object[key]:
                    dfl1["target"] = json_object[key]["target"]
                df = pd.concat([df, dfl1]).reset_index(drop=True)

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

@timing
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

def get_meta_dataset(data_path, chunk_size, dataset_type="train"):
    logger.info(f"reading {dataset_type} json...")
    # read as json
    json_object = read_json(data_path, dataset_type)

    logger.info(f"parsing orders started...")
    for idx, chunk in enumerate(iterate_json(json_object, chunk_size)):
        # get chunk idx
        chunk_users = sorted([int(x[5:]) for x in list(chunk.keys())])
        chunk_users = f"{chunk_users[0]}_{chunk_users[-1]}"

        df = parse_meta(chunk)
        # save as parquet by chunks
        df.to_parquet(
            data_path+f"meta/{dataset_type}/{idx}_{chunk_users}_{dataset_type}.parquet.gzip",
            compression="gzip",
            index=False
        )
        logger.info(f"{(idx+1)*chunk_size}/{len(json_object)}")
    logger.info(f"...collecting meta dataset completed")

def get_accepted_dataset(data_path, chunk_size, dataset_type="train"):
    logger.info(f"reading {dataset_type} json...")
    # read as json
    json_object = read_json(data_path, dataset_type)

    logger.info(f"parsing orders started...")
    for idx, chunk in enumerate(iterate_json(json_object, chunk_size)):
        # get chunk idx
        chunk_users = sorted([int(x[5:]) for x in list(chunk.keys())])
        chunk_users = f"{chunk_users[0]}_{chunk_users[-1]}"

        df = parse_accepted(chunk)
        # save as parquet by chunks
        df.to_parquet(
            data_path+f"accepted/{dataset_type}/{idx}_{chunk_users}_{dataset_type}.parquet.gzip",
            compression="gzip",
            index=False
        )
        logger.info(f"{(idx+1)*chunk_size}/{len(json_object)}")
    logger.info(f"...collecting meta dataset completed")

@timing # TODO update with read_data()
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

@timing # TODO update with read_data()
def read_orders_data(data_path, dataset_type="train"):
    logger.info(f"reading {dataset_type} dataset...")
    # clms = ["id", "count", "brand-id", "site-id", "target"]
    clms = ["id", "count", "brand-id", "site-id","general-category-path", "target"]
    df = pd.DataFrame()
    for f in os.listdir(data_path+f"orders/{dataset_type}/"):
        dfs = pd.read_parquet(data_path+f"orders/{dataset_type}/{f}")
        dfs = dfs[clms]
        # dfs = dfs.drop_duplicates().reset_index(drop=True)
        df = pd.concat([df,dfs]).reset_index(drop=True)

    # categorical features to str
    df[["brand-id", "site-id"]] = df[["brand-id", "site-id"]].fillna(-1)
    df[["brand-id", "site-id"]] = df[["brand-id", "site-id"]].astype(int).astype(str)

    if False: # adding sex score
        with open("models/sex_dict_meta.pkl", "rb") as f:
            sex_dict_meta = pickle.load(f)
        df["sex_score"] = df["site-id"].map(sex_dict_meta)
        logger.info(df.sex_score.isna().sum())
    if True:
        df = df.explode("general-category-path").reset_index(drop=True)
        df["general-category-path"] = df["general-category-path"].fillna(-1)
        df["general-category-path"] = df["general-category-path"].astype(str)

        with open("../Downloads/general_categories_mapping_corr.json", "rb") as f:
            d = json.load(f)
        df["general-category-path"] = df["general-category-path"].map({list(item.keys())[0]: item['guess'] for item in d})
        df["general-category-path"] = df["general-category-path"].fillna("-1")
        logger.info(f"AAAAAAA {df['general-category-path'].isna().sum()}")
    # convert target to int
    df.target = (df.target == "male").astype(int)

    return df

@timing # TODO update with read_data()
def read_meta_data(data_path, dataset_type="train"):
    logger.info(f"reading {dataset_type} dataset...")
    df = pd.DataFrame()
    for f in os.listdir(data_path+f"meta/{dataset_type}/"):
        dfs = pd.read_parquet(data_path+f"meta/{dataset_type}/{f}")
        # dfs = dfs[clms]
        # dfs = dfs.drop_duplicates().reset_index(drop=True)
        df = pd.concat([df,dfs]).reset_index(drop=True)
    
    df[["recency", "frequency", "monetary"]] = df[["recency", "frequency", "monetary"]].fillna(-1)
    # drop unused columns
    df = df.drop("user", axis=1)

    # convert target to int
    df.target = (df.target == "male").astype(int)

    return df

@timing
def read_data(data_path, dataset_source, dataset_type="train"):
    df = pd.DataFrame()
    for f in os.listdir(data_path+f"{dataset_source}/{dataset_type}/"):
        dfs = pd.read_parquet(data_path+f"{dataset_source}/{dataset_type}/{f}")
        df = pd.concat([df,dfs]).reset_index(drop=True)

    return df


@timing
def get_sex_score_dict(df):
    df_encoded = pd.get_dummies(df, columns=['target'], prefix='target')
    df_grouped = df_encoded.groupby("site-id").agg({'user':'count', 'target_female':'sum'}).reset_index()
    df_grouped['target_female'] = df_grouped['target_female'] / df_grouped['user']
    sex_dict = dict(zip(df_grouped["site-id"], df_grouped["target_female"]))

    return sex_dict

@timing
def get_logreg_train_data_meta(data_path):
    # read data
    data_meta = read_data(data_path, "meta", "train")
    data_meta = data_meta[["site-id", "user", "target"]]
    # calc sex score
    sex_dict_meta = get_sex_score_dict(data_meta)
    
    # save sex score dicts
    with open("models/sex_dict_meta.pkl", "wb") as f:
        pickle.dump(sex_dict_meta, f)

    data_meta["sex_score_meta"] = data_meta["site-id"].map(sex_dict_meta)
    data_meta = data_meta.groupby('user').agg(
        {"sex_score_meta": 'mean', 'target': 'last'}
    ).reset_index()

    return data_meta

    
def get_logreg_train_data_both(data_path):
    # read data
    data_meta = read_data(data_path, "meta", "train")
    data_meta = data_meta[["site-id", "user", "target"]]

    data_accepted = read_data(data_path, "accepted", "train")
    data_accepted = data_accepted.rename(columns={"accepted-site-id": "site-id"})
    data_accepted["site-id"] = data_accepted["site-id"].astype(int)

    # calc sex score
    sex_dict_meta = get_sex_score_dict(data_meta)
    sex_dict_accepted = get_sex_score_dict(data_accepted)

    # save sex score dicts
    with open("models/sex_dict_meta.pkl", "wb") as f:
        pickle.dump(sex_dict_meta, f)
    with open("models/sex_dict_accepted.pkl", "wb") as f:
        pickle.dump(sex_dict_accepted, f)

    # map score
    data_meta["sex_score_meta"] = data_meta["site-id"].map(sex_dict_meta)
    data_accepted["sex_score_accepted"] = data_accepted["site-id"].map(sex_dict_accepted)

    data_meta = data_meta.groupby('user').agg(
        {"sex_score_meta": 'mean', 'target': 'last'}
    ).reset_index()
    data_accepted = data_accepted.groupby('user').agg(
        {"sex_score_accepted": 'mean', 'target': 'last'}
    ).reset_index()

    data = data_meta.merge(data_accepted, how="left")
    data = data.dropna().reset_index(drop=False)

    return data

@timing
def get_logreg_val_data_meta(data_path):
    # read data
    data = read_data(data_path, "meta", "val")
    data = data[["site-id", "user", "target"]]

    # load sex score
    with open("models/sex_dict_meta.pkl", "rb") as f:
        sex_dict_meta = pickle.load(f)

    # map sex score
    data["sex_score_meta"] = data["site-id"].map(sex_dict_meta)
    # fill empty values
    data.loc[data.sex_score_meta.isna(), "sex_score_meta"] = data.sex_score_meta.mean()
    result = data.groupby("user")["sex_score_meta"].mean().reset_index().merge(
        data.groupby("user")["target"].last().reset_index(),
        how="left"
    )

    return result

@timing
def get_logreg_val_data_both(data_path):
    # read data
    data_meta = read_data(data_path, "meta", "val")
    data_meta = data_meta[["site-id", "user", "target"]]

    data_accepted = read_data(data_path, "accepted", "val")
    data_accepted = data_accepted.rename(columns={"accepted-site-id": "site-id"})
    data_accepted["site-id"] = data_accepted["site-id"].astype(int)

    # load sex score
    with open("models/sex_dict_meta.pkl", "rb") as f:
        sex_dict_meta = pickle.load(f)
    with open("models/sex_dict_accepted.pkl", "rb") as f:
        sex_dict_accepted = pickle.load(f)

    # map sex score
    data_meta["sex_score_meta"] = data_meta["site-id"].map(sex_dict_meta)
    data_accepted["sex_score_accepted"] = data_accepted["site-id"].map(sex_dict_accepted)

    # fill empty values
    data_meta.loc[data_meta.sex_score_meta.isna(), "sex_score_meta"] = data_meta.sex_score_meta.mean()
    result_meta = data_meta.groupby("user")["sex_score_meta"].mean().reset_index().merge(
        data_meta.groupby("user")["target"].last().reset_index(),
        how="left"
    )
    data_accepted.loc[data_accepted.sex_score_accepted.isna(), "sex_score_accepted"] = data_accepted.sex_score_accepted.mean()
    result_accepted = data_accepted.groupby("user")["sex_score_accepted"].mean().reset_index().merge(
        data_accepted.groupby("user")["target"].last().reset_index(),
        how="left"
    )
    
    # get merged resulting df
    result = result_meta.merge(
        result_accepted.drop("target", axis=1),
        how="left", left_on="user", right_on="user"
    )

    result["sex_score_accepted"] = result["sex_score_accepted"].fillna(result["sex_score_accepted"].mean())

    return result