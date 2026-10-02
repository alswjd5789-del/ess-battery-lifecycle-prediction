import csv
import json
import os

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

from .features import FEATURES, TARGET_MAPE


def split_batches(df, random_state=42):
    """B1 → Train 80% / Valid 20%, B2 → Test, B3 → Additional.

    Returns
    -------
    train_df, valid_df, b2, b3 : pd.DataFrame
    """
    b1 = df[df['batch'] == 'B1']
    b2 = df[df['batch'] == 'B2']
    b3 = df[df['batch'] == 'B3']
    train_df, valid_df = train_test_split(b1, test_size=0.2, random_state=random_state)
    return train_df, valid_df, b2, b3


def build_model_configs():
    """DAY 1 설계서에서 선정한 3개 후보 모델 반환."""
    alphas = np.logspace(-2, 4, 50)
    return {
        'ElasticNet': GridSearchCV(
            ElasticNet(max_iter=20000),
            {'alpha': alphas, 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
            cv=5, scoring='neg_mean_absolute_percentage_error'
        ),
        'GBM': GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        'RF': RandomForestRegressor(n_estimators=300, random_state=42),
    }


def _metrics(y_true, y_pred):
    return {
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'r2':   r2_score(y_true, y_pred),
        'mae':  mean_absolute_error(y_true, y_pred),
    }


def train_and_evaluate(train_df, valid_df, b2, b3):
    """모든 모델을 학습하고 Train CV / Valid / Test B2 / Test B3 성능을 반환.

    Returns
    -------
    results : dict  {모델명: {cv, valid, test2, test3, best_model, scaler}}
    labels  : tuple (y_train, y_valid, y_test2, y_test3)
    """
    X_train = train_df[FEATURES].values;  y_train = train_df['cycle_life'].values
    X_valid = valid_df[FEATURES].values;  y_valid = valid_df['cycle_life'].values
    X_test2 = b2[FEATURES].values;        y_test2 = b2['cycle_life'].values
    X_test3 = b3[FEATURES].values;        y_test3 = b3['cycle_life'].values

    scaler  = StandardScaler().fit(X_train)
    X_tr_s  = scaler.transform(X_train)
    X_va_s  = scaler.transform(X_valid)
    X_te2_s = scaler.transform(X_test2)
    X_te3_s = scaler.transform(X_test3)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in build_model_configs().items():
        model.fit(X_tr_s, y_train)
        best = model if name in ('GBM', 'RF') else model.best_estimator_

        cv_pred = cross_val_predict(best, X_tr_s, y_train, cv=kf)
        cv      = _metrics(y_train, cv_pred);  cv.pop('mae')

        results[name] = {
            'best_model': best,
            'scaler':     scaler,
            'cv':    {**cv,  'pred': cv_pred},
            'valid': {**_metrics(y_valid,  best.predict(X_va_s)),  'pred': best.predict(X_va_s)},
            'test2': {**_metrics(y_test2,  best.predict(X_te2_s)), 'pred': best.predict(X_te2_s)},
            'test3': {**_metrics(y_test3,  best.predict(X_te3_s)), 'pred': best.predict(X_te3_s)},
        }

    return results, (y_train, y_valid, y_test2, y_test3)


def compute_gaps(results):
    """Notion 포맷 기준 Gap 5개를 results 딕셔너리에 추가."""
    for name, r in results.items():
        r['gaps'] = {
            'train_valid':    r['valid']['mape']  - r['cv']['mape'],
            'valid_test':     r['test2']['mape']  - r['valid']['mape'],
            'target_test':    r['test2']['mape']  - TARGET_MAPE,
            'b2_b3':          r['test3']['mape']  - r['test2']['mape'],
            'target_test_b3': r['test3']['mape']  - TARGET_MAPE,
        }
    return results


def save_summary(results, split_info, out_path, baseline=None):
    """성능 요약을 JSON으로 저장하고, 같은 폴더에 model_performance.csv도 생성.

    Parameters
    ----------
    baseline : dict, optional
        {cv, valid, test2, test3, gaps} 구조의 Baseline(Train 평균 상수 예측) 성능.
    """
    summary = {
        'best_model':  min(results, key=lambda k: results[k]['valid']['mape']),
        'split':       split_info,
        'target_mape': TARGET_MAPE,
        'results': {
            name: {
                'train_cv':  {'mape': round(r['cv']['mape'],    4), 'r2': round(r['cv']['r2'],    4)},
                'valid_ho':  {'mape': round(r['valid']['mape'], 4), 'r2': round(r['valid']['r2'], 4), 'mae': round(r['valid']['mae'], 2)},
                'test_b2':   {'mape': round(r['test2']['mape'], 4), 'r2': round(r['test2']['r2'], 4), 'mae': round(r['test2']['mae'], 2)},
                'test_b3':   {'mape': round(r['test3']['mape'], 4), 'r2': round(r['test3']['r2'], 4), 'mae': round(r['test3']['mae'], 2)},
                'gap_train_valid':    round(r['gaps']['train_valid'],    4),
                'gap_valid_test':     round(r['gaps']['valid_test'],     4),
                'gap_target_test':    round(r['gaps']['target_test'],    4),
                'gap_b2_b3':          round(r['gaps']['b2_b3'],          4),
                'gap_target_test_b3': round(r['gaps']['target_test_b3'], 4),
            }
            for name, r in results.items()
        },
        'baseline': {
            'train_cv':  {'mape': round(baseline['cv']['mape'],    4), 'r2': round(baseline['cv']['r2'],    4)},
            'valid_ho':  {'mape': round(baseline['valid']['mape'], 4), 'r2': round(baseline['valid']['r2'], 4)},
            'test_b2':   {'mape': round(baseline['test2']['mape'], 4), 'r2': round(baseline['test2']['r2'], 4)},
            'test_b3':   {'mape': round(baseline['test3']['mape'], 4), 'r2': round(baseline['test3']['r2'], 4)},
            'gap_train_valid':    round(baseline['gaps']['train_valid'],    4),
            'gap_valid_test':     round(baseline['gaps']['valid_test'],     4),
            'gap_target_test':    round(baseline['gaps']['target_test'],    4),
            'gap_b2_b3':          round(baseline['gaps']['b2_b3'],          4),
            'gap_target_test_b3': round(baseline['gaps']['target_test_b3'], 4),
        } if baseline is not None else None,
    }
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    # CSV 저장 (Baseline 포함)
    csv_path = os.path.join(os.path.dirname(out_path), 'model_performance.csv')
    gap_keys = ['train_valid', 'valid_test', 'target_test', 'b2_b3', 'target_test_b3']

    def _row(name, r):
        row = {
            'model':         name,
            'train_cv_mape': f"{r['cv']['mape']:.2f}",
            'valid_mape':    f"{r['valid']['mape']:.2f}",
            'test_b2_mape':  f"{r['test2']['mape']:.2f}",
            'test_b3_mape':  f"{r['test3']['mape']:.2f}",
        }
        for k in gap_keys:
            row[f'gap_{k}'] = f"{r['gaps'][k]:+.2f}"
        return row

    rows = []
    if baseline is not None:
        rows.append(_row('Baseline', baseline))
    for name, r in results.items():
        rows.append(_row(name, r))

    fieldnames = ['model', 'train_cv_mape', 'valid_mape', 'test_b2_mape', 'test_b3_mape',
                  'gap_train_valid', 'gap_valid_test', 'gap_target_test', 'gap_b2_b3', 'gap_target_test_b3']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return summary
