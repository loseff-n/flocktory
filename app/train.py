import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import data_process
from utils import logger, timing

@timing
def catboost_fit(data_path, cat_features, dataset_source):
    logger.info(f"catboost fit started...")
    if dataset_source == "visits":
        df = data_process.read_visits_data(data_path, dataset_type="train")
    elif dataset_source == "orders":
        df = data_process.read_orders_data(data_path, dataset_type="train")
    elif dataset_source == "meta":
        df = data_process.read_meta_data(data_path, dataset_type="train")

    # train test split
    xtr, xvl, ytr, yvl = train_test_split(
        df.drop("target", axis=1),
        df.target, 
        test_size=0.2, random_state=42
    )

    model = CatBoostClassifier(
        iterations=1000, 
        # depth=10, 
        learning_rate=0.05, 
        loss_function='Logloss', 
        verbose=1
    )

    model.fit(
        xtr, ytr, 
        eval_set=(xvl, yvl), 
        cat_features=cat_features, 
        plot=True
    )

    model.save_model('models/cb_model.cbm')
    logger.info(f"...catboost fit completed, model saved")

    return model

@timing 
def logreg_fit(data_path, two_feats=False):
    '''
    two_feats : ["both", "brands"]
    '''
    logger.info(f"logreg fit started...")
    # collect data
    if two_feats == "both":
        data = data_process.get_logreg_train_data_both(data_path)
        x = data[['sex_score_meta', 'sex_score_accepted']]
    elif two_feats == "brands":
        data = data_process.get_logreg_train_data_brands(data_path)
        x = data[['sex_score_meta', 'sex_score_brands']]
    else:
        data = data_process.get_logreg_train_data_meta(data_path)
        x = data[["sex_score_meta"]]
    y = data.target
    # split
    xtr, xts, ytr, yts = train_test_split(x,y, test_size=0.2, random_state=42)
    # fit
    model = LogisticRegression()
    model.fit(xtr, ytr)
    # evaluate and save
    metrics = calc_metrics(model, pd.concat([xts, yts], axis=1))

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

def logreg_predict(data_path):
    logger.info(f"preparing data for logreg predict...")
    data_train, data_test = data_process.logreg_predict_data(data_path)
    x_train = data_train[["sex_score_meta"]]
    y = data_train.target

    # 1 run check
    logger.info(f"logreg fit started...")
    xtr, xts, ytr, yts = train_test_split(x_train,y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(xtr, ytr)
    metrics = calc_metrics(model, pd.concat([xts, yts], axis=1))
    # 2 run predict
    logger.info(f"logreg predict started...")
    model = LogisticRegression()
    model.fit(x_train, y)
    y_pred = model.predict(data_test[["sex_score_meta"]])
    data_test["target"] = y_pred
    data_test = data_test[["user", "target"]]
    data_test.to_csv("data/test.csv", index=False)
    logger.info(f"...prediction completed and saved")

    return model, data_test
    

@timing
def calc_metrics(model, df_val):
    logger.info(f"model matrics:")
    ypred = model.predict(df_val.drop("target", axis=1))
    yprob = model.predict_proba(df_val.drop("target", axis=1))[:, 1]
    ytrue = df_val.target

    accuracy = accuracy_score(ytrue, ypred)
    conf_matrix = confusion_matrix(ytrue, ypred)
    class_report = classification_report(ytrue, ypred)

    logger.info(f"{accuracy}")
    logger.info(f"{conf_matrix}")
    logger.info(f"{class_report}")

    metrics = {
        "accuracy": accuracy,
        "conf_matrix": conf_matrix,
        "class_report": class_report
    }

    return metrics

