import pandas as pd
import numpy as np
import warnings
import csv
import os
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')
np.random.seed(123)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

output_dir = './trd_ml_output/'

def load_exp_data(pred_prop='Tensile_value', split_ratio=0.8, seed=666, add=False):
    df = pd.read_excel('./../data/data.xlsx')
    if not add:
        df.drop(columns=['热锻/℃'], inplace=True)

    if pred_prop != 'all':
        df = df.drop(columns=[col for col in ['Tensile_value', 'Yield_value', 'Elongation_value'] if col != pred_prop])

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(split_ratio * len(df))
    return df[:split_idx], df[split_idx:]

def eval_model(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred)
    }

def ml_pred_exp(reg, target='Tensile_value'):
    results = defaultdict(list)
    for seed in range(3):
        train_data, test_data = load_exp_data(pred_prop=target, seed=seed)
        X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
        X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

        regr = reg.fit(X_train, y_train)
        train_eval = eval_model(y_train, regr.predict(X_train))
        test_eval = eval_model(y_test, regr.predict(X_test))

        for key in train_eval:
            results[f'train_{key}'].append(train_eval[key])
            results[f'test_{key}'].append(test_eval[key])

    return {f'{key}_mean': np.mean(values) for key, values in results.items()}, \
           {f'{key}_std': np.std(values) for key, values in results.items()}

def save_results(path, headers, metrics, params):
    with open(path, 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerow([metrics[f'{key}_mean'] for key in headers[:4]] +
                           [metrics[f'{key}_std'] for key in headers[:4]] +
                           [params[key] for key in params])

def run_regressor(target, regressor_name, param_grid):
    regressor_funcs = {
        'svr': SVR,
        'rf': RandomForestRegressor,
        'xgb': xgb.XGBRegressor,
        'gbr': GradientBoostingRegressor,
        'krr': KernelRidge,
        'mlp': MLPRegressor
    }
    path = f"{output_dir}{target}_{regressor_name}.csv"
    headers = ['train_r2', 'test_r2', 'train_mae', 'test_mae', 'train_r2_std', 'test_r2_std', 'train_mae_std', 'test_mae_std'] + list(param_grid.keys())

    for params in [dict(zip(param_grid, v)) for v in product(*param_grid.values())]:
        reg = regressor_funcs[regressor_name](**params)
        mean_results, std_results = ml_pred_exp(reg, target=target)
        save_results(path, headers, {**mean_results, **std_results}, params)

def product(*args):
    if not args:
        return iter(((),))
    return (items + (item,)
            for items in product(*args[:-1]) for item in args[-1])

def run_all_regressors():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    regressors = {
        'svr': {'kernel': ['linear'], 'degree': range(10, 15), 'C': np.arange(2, 2.5, 0.1), 'epsilon': np.arange(0.2, 2.1, 0.1)},
        'rf': {'n_estimators': range(5, 25, 5), 'max_depth': range(4, 8), 'random_state': [666, 789, 123]},
        'xgb': {'base_score': np.arange(0.2, 0.4, 0.1), 'learning_rate': [0.05, 0.15], 'random_state': [666, 789, 123], 'verbosity': [0, 2]},
        'gbr': {'learning_rate': [0.01, 0.05, 0.1, 0.001, 0.15], 'n_estimators': range(100, 150, 5), 'random_state': [666, 789, 123]},
        'krr': {'alpha': np.arange(0.1, 0.2, 0.1), 'degree': range(10, 16)},
        'mlp': {'hidden_layer_sizes': [[2, 4, 4], [2, 4, 8]], 'activation': ['relu'], 'solver': ['lbfgs'], 'alpha': [1e-4, 1e-3]}
    }

    # targets = ['Tensile_value', 'Yield_value', 'Elongation_value']
    targets = ['Tensile_value']

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_regressor, target, regressor, params) 
                   for target in targets for regressor, params in regressors.items()]

        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    run_all_regressors()
