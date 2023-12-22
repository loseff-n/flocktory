import json
import os
import pandas as pd
import pickle
from tqdm import tqdm

from utils import logger, timing

ORDERS_SCORES = True
if ORDERS_SCORES:
    logger.info(f"ORDERS_SCORES MODE ON")


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
    # clms = ["id", "count", "brand-id", "site-id","general-category-path", "target"]
    clms = ["id", "count", "brand-id", "site-id", "user", "target"]
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
    if False: # analyze general-category-path
        df = df.explode("general-category-path").reset_index(drop=True)
        df["general-category-path"] = df["general-category-path"].fillna(-1)
        df["general-category-path"] = df["general-category-path"].astype(str)

    if ORDERS_SCORES: # adding sex score instead of columns
        # get & save sex score dicts
        sex_dict_item = get_sex_score(df, "id") 
        sex_dict_site = get_sex_score(df, "site-id") 
        with open("models/sex_dict_item.pkl", "wb") as f:
            pickle.dump(sex_dict_item, f)
        with open("models/sex_dict_site.pkl", "wb") as f:
            pickle.dump(sex_dict_site, f)
        # map dicts to the new columns
        df["sex_score_item"] = df["id"].map(sex_dict_item)
        df["sex_score_site"] = df["site-id"].map(sex_dict_site)
        logger.info(f"empty sex score item: {df.sex_score_item.isna().sum()}")
        logger.info(f"empty sex score site: {df.sex_score_site.isna().sum()}")
        # drop unused columns
        if True: 
            del df["id"]
            del df["site-id"] 

        df = df.groupby('user').agg(
            {"sex_score_item": 'mean', "sex_score_site": "mean", 'target': 'last'}
        ).reset_index()

        logger.info(f"COLUMNS {df.columns}")

        del df["user"]

    # convert target to int
    df.target = (df.target == "male").astype(int)

    return df

def read_orders_data_scores_val(data_path):
    df = read_data(data_path, "orders", "val")
    clms = ["id", "count", "brand-id", "site-id", "user", "target"]
    df = df[clms]
    df[["brand-id", "site-id"]] = df[["brand-id", "site-id"]].fillna(-1)
    df[["brand-id", "site-id"]] = df[["brand-id", "site-id"]].astype(int).astype(str)

    # read sex score dicts
    with open("models/sex_dict_item.pkl", "rb") as f:
        sex_dict_item = pickle.load(f)
    with open("models/sex_dict_site.pkl", "rb") as f:
        sex_dict_site = pickle.load(f)

    # map dicts
    df["sex_score_item"] = df["id"].map(sex_dict_item)
    df["sex_score_site"] = df["site-id"].map(sex_dict_site)

    logger.info(f"empty sex score item: {df.sex_score_item.isna().sum()}")
    logger.info(f"empty sex score site: {df.sex_score_site.isna().sum()}")
    # drop unused columns
    if True: 
        del df["id"]
        del df["site-id"]

    df = df.groupby('user').agg(
        {"sex_score_item": 'mean', "sex_score_site": "mean", 'target': 'last'}
    ).reset_index()

    del df["user"]

    logger.info(f"COLUMNS {df.columns}")

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
    # read chunked dataframes from the valid source
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

def get_sex_score(df, column):
    # get share of female in column
    df = df.drop_duplicates(subset=["user", column])
    sex_dict = df.groupby(column)["target"].value_counts(normalize=True).unstack().reset_index()[[column, "female"]]
    sex_dict.female = sex_dict.female.fillna(0)
    sex_dict = sex_dict.rename(columns={"female": f"sex_score_{column}"})
    sex_dict[column] = sex_dict[column].astype(str)

    return dict(zip(sex_dict[column], sex_dict[f"sex_score_{column}"]))

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

def get_logreg_train_data_brands(data_path):
    # read data
    data_meta = read_data(data_path, "meta", "train")
    data_meta = data_meta[["site-id", "user", "target"]]

    logger.info(f"META READ")

    data_brands = read_data(data_path, "brands", "train")
    data_brands = data_brands.rename(columns={"visited-universal-brands": "site-id"})
    data_brands["site-id"] = data_brands["site-id"].astype(int)

    logger.info(f"BRANDS READ")

    # calc sex score
    sex_dict_meta = get_sex_score_dict(data_meta)
    logger.info(f"META DICT")
    sex_dict_brand = get_sex_score_dict(data_brands)
    logger.info(f"BRANDS DICT")

    # save sex score dicts
    with open("models/sex_dict_meta.pkl", "wb") as f:
        pickle.dump(sex_dict_meta, f)
    with open("models/sex_dict_brands.pkl", "wb") as f:
        pickle.dump(sex_dict_brand, f)

    # map score
    data_meta["sex_score_meta"] = data_meta["site-id"].map(sex_dict_meta)
    data_brands["sex_score_brands"] = data_brands["site-id"].map(sex_dict_brand)

    data_meta = data_meta.groupby('user').agg(
        {"sex_score_meta": 'mean', 'target': 'last'}
    ).reset_index()
    data_brands = data_brands.groupby('user').agg(
        {"sex_score_brands": 'mean', 'target': 'last'}
    ).reset_index()

    data = data_meta.merge(data_brands, how="left")
    data["sex_score_brands"] = data["sex_score_brands"].fillna(0.5)

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

@timing
def get_logreg_val_data_brands(data_path):
    # read data
    data_meta = read_data(data_path, "meta", "val")
    data_meta = data_meta[["site-id", "user", "target"]]

    data_brands = read_data(data_path, "brands", "val")
    data_brands = data_brands.rename(columns={"visited-universal-brands": "site-id"})
    data_brands["site-id"] = data_brands["site-id"].astype(int)

    # load sex score
    with open("models/sex_dict_meta.pkl", "rb") as f:
        sex_dict_meta = pickle.load(f)
    with open("models/sex_dict_brands.pkl", "rb") as f:
        sex_dict_brand = pickle.load(f)

    # map sex score
    data_meta["sex_score_meta"] = data_meta["site-id"].map(sex_dict_meta)
    data_brands["sex_score_brands"] = data_brands["site-id"].map(sex_dict_brand)

    # fill empty values
    data_meta.loc[data_meta.sex_score_meta.isna(), "sex_score_meta"] = data_meta.sex_score_meta.mean()
    result_meta = data_meta.groupby("user")["sex_score_meta"].mean().reset_index().merge(
        data_meta.groupby("user")["target"].last().reset_index(),
        how="left"
    )
    data_brands.loc[data_brands.sex_score_brands.isna(), "sex_score_brands"] = data_brands.sex_score_brands.mean()
    result_brands = data_brands.groupby("user")["sex_score_brands"].mean().reset_index().merge(
        data_brands.groupby("user")["target"].last().reset_index(),
        how="left"
    )
    
    # get merged resulting df
    result = result_meta.merge(
        result_brands.drop("target", axis=1),
        how="left", left_on="user", right_on="user"
    )

    result["sex_score_brands"] = result["sex_score_brands"].fillna(result["sex_score_brands"].mean())

    return result

@timing 
def logreg_predict_data(data_path):
    data_train = pd.concat([
        read_data(data_path, "meta", "train")[["site-id", "user", "target"]],
        read_data(data_path, "meta", "val")[["site-id", "user", "target"]]
    ]).reset_index(drop=True)

    # calc sex score & save as dict
    sex_dict_meta = get_sex_score_dict(data_train)
    with open("models/sex_dict_meta.pkl", "wb") as f:
        pickle.dump(sex_dict_meta, f)

    # update the data with the score
    data_train["sex_score_meta"] = data_train["site-id"].map(sex_dict_meta)
    data_train = data_train.groupby('user').agg(
        {"sex_score_meta": 'mean', 'target': 'last'}
    ).reset_index()

    data_test = read_data(data_path, "meta", "test")[["site-id", "user"]]
    data_test["sex_score_meta"] = data_test["site-id"].map(sex_dict_meta)
    data_test = data_test.groupby('user').agg(
        {"sex_score_meta": 'mean'}
    ).reset_index()

    return data_train, data_test