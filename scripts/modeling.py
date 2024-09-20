import dask.dataframe as dd
import dask.array as da
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import mutual_info_regression
from dask import delayed
import numpy as np
import shap

def feature_selection_dask(X_train, y_train, num_partitions=10):
    X_train_dd = dd.from_pandas(X_train, npartitions=num_partitions)
    y_train_dd = dd.from_pandas(y_train, npartitions=num_partitions)

    def compute_scores(X_batch, y_batch):
        selector = mutual_info_regression(X_batch, y_batch.iloc[:, 0], n_neighbors=min(3, X_batch.shape[0] - 1))  
        return selector

    compute_scores_delayed = [delayed(compute_scores)(X_batch.compute(), y_batch.compute()) for X_batch, y_batch in zip(X_train_dd.to_delayed(), y_train_dd.to_delayed())]
    scores = da.stack([da.from_delayed(score, shape=(X_train.shape[1],), dtype=float) for score in compute_scores_delayed])
    scores = scores.mean(axis=0).compute(scheduler='processes')

    X_train_selected = X_train.loc[:, scores > np.mean(scores)]

    return X_train_selected

def handle_imbalanced_data_regression(X_train, y_train):
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=weights)
    return model

def train_and_evaluate_models(X_train_selected, X_test_selected, y_train_upsampled, y_test, target_columns):
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer
    from sklearn.linear_model import LinearRegression
    import xgboost as xgb
    from joblib import Parallel, delayed

    def mean_squared_logarithmic_error(y_true, y_pred):
        return np.mean((np.log(y_true + 1e-6) - np.log(y_pred + 1e-6)) ** 2)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    def train_model(model):
        model.fit(X_train_selected, y_train_upsampled[target_columns[0]])
        scores = cross_validate(model, X_train_selected, y_train_upsampled[target_columns[0]], cv=3, scoring=make_scorer(mean_squared_logarithmic_error, greater_is_better=False))
        return model, scores

    results = Parallel(n_jobs=-1)(delayed(train_model)(model) for model in models.values())

    for model, scores in results:
        print(f'{model.__class__.__name__} - MSLE: {-scores["test_score"].mean()}')

    # Get the best model
    best_model, best_scores = max(results, key=lambda x: x[1]["test_score"].mean())

    # Print the best model's performance
    print(f'Best Model: {best_model.__class__.__name__} - MSLE: {-best_scores["test_score"].mean()}')

    # Use the best model to make predictions on the test set
    y_pred = best_model.predict(X_test_selected)

    # Evaluate the best model's performance on the test set
    test_msle = mean_squared_logarithmic_error(y_test[target_columns[0]], y_pred)
    print(f'Test MSLE: {test_msle}')

def interpret_model(model, X_test_selected, max_display=57):
    explainer = shap.TreeExplainer(model, data=X_test_selected, feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_test_selected, check_additivity=False)
    shap.summary_plot(shap_values, X_test_selected, max_display=max_display)
