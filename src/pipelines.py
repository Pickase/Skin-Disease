from xgboost import XGBClassifier

def build_xgb_pipeline():
    """
    Returns an XGBoost classifier with tuned hyperparameters.
    """
    model = XGBClassifier(
        colsample_bytree=0.5,
        gamma=1,
        learning_rate=0.01,
        max_depth=4,
        reg_lambda=0,
        subsample=0.4,
        n_estimators=200,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    return model
