"""Baseline + proposed-model experiments on NSL-KDD."""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
from xgboost import XGBClassifier

from data_loader import load_raw, preprocess


def get_models(task='binary'):
    n_jobs = -1
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=n_jobs),
        'Naive Bayes':         GaussianNB(),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'KNN (k=5)':           KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
        'Random Forest':       RandomForestClassifier(n_estimators=100,
                                                      random_state=42, n_jobs=n_jobs),
        'XGBoost':             XGBClassifier(n_estimators=200, max_depth=8,
                                             learning_rate=0.1, random_state=42,
                                             n_jobs=n_jobs, eval_metric='logloss',
                                             tree_method='hist'),
    }


def evaluate(name, model, X_tr, y_tr, X_te, y_te, task='binary'):
    t0 = time.time(); model.fit(X_tr, y_tr); train_t = time.time() - t0
    t0 = time.time(); pred = model.predict(X_te); test_t = time.time() - t0

    avg = 'binary' if task == 'binary' else 'macro'
    row = {
        'Model': name,
        'Accuracy':  accuracy_score(y_te, pred),
        'Precision': precision_score(y_te, pred, average=avg, zero_division=0),
        'Recall':    recall_score(y_te, pred, average=avg, zero_division=0),
        'F1':        f1_score(y_te, pred, average=avg, zero_division=0),
        'Train(s)':  train_t,
        'Test(s)':   test_t,
    }
    if task == 'binary':
        cm = confusion_matrix(y_te, pred)
        tn, fp, fn, tp = cm.ravel()
        row['FPR'] = fp / (fp + tn)
        row['DetectionRate'] = tp / (tp + fn)
        try:
            proba = model.predict_proba(X_te)[:, 1]
            row['ROC-AUC'] = roc_auc_score(y_te, proba)
        except Exception:
            row['ROC-AUC'] = np.nan
    return row, pred


def main(task='binary'):
    print(f'\n{"="*72}\n  NSL-KDD INTRUSION DETECTION  --  {task.upper()} CLASSIFICATION\n{"="*72}')
    train, test = load_raw()
    X_tr, X_te, y_tr, y_te, feats, le = preprocess(train, test, task=task)
    print(f'Train {X_tr.shape} | Test {X_te.shape} | Features: {len(feats)}\n')

    rows, preds = [], {}
    for name, model in get_models(task).items():
        row, pred = evaluate(name, model, X_tr, y_tr, X_te, y_te, task)
        rows.append(row); preds[name] = pred
        print(f'  {name:22s} acc={row["Accuracy"]:.4f}  f1={row["F1"]:.4f}  '
              f'({row["Train(s)"]:.1f}s train)')

    df = pd.DataFrame(rows).sort_values('F1', ascending=False)
    out = f'results/results_{task}.csv'
    df.to_csv(out, index=False)
    print(f'\n--- RESULTS ({task}) ---')
    cols = ['Model','Accuracy','Precision','Recall','F1']
    if task == 'binary': cols += ['FPR','DetectionRate','ROC-AUC']
    print(df[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nSaved -> {out}')

    # Per-class report for the best model
    best = df.iloc[0]['Model']
    names = None if task == 'binary' else list(le.classes_)
    print(f'\n--- PER-CLASS REPORT: {best} ---')
    print(classification_report(y_te, preds[best], target_names=names, zero_division=0))

    np.save(f'results/preds_{task}.npy', preds[best])
    with open(f'results/best_{task}.json','w') as f:
        json.dump({'best_model': best, 'metrics': df.iloc[0].to_dict()}, f, indent=2, default=str)
    return df


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'binary'
    main(task)
