from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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
        verbose=0
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
def evaluate_model(model, df_val):
    logger.info(f"model matrics:")
    ypred = model.predict(df_val.drop("target", axis=1))
    yprob = model.predict_proba(df_val.drop("target", axis=1))[:, 1]
    ytrue = df_val.target

    accuracy = accuracy_score(ytrue, ypred)
    precision = precision_score(ytrue, ypred)
    recall = recall_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    roc_auc = roc_auc_score(ytrue, yprob)
    conf_matrix = confusion_matrix(ytrue, ypred)
                                
    metrics_dict = {
        'Accuracy': f"{accuracy :.4f}",
        'Precision': f"{precision :.4f}",
        'Recall': f"{recall :.4f}",
        'F1 Score': f"{f1 :.4f}",
        'ROC AUC': f"{roc_auc :.4f}",
        'Confusion Matrix': conf_matrix.tolist()
    }
    logger.info(f"{metrics_dict}")

    return metrics_dict


