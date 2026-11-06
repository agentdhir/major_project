"""Proposed improvement: class-imbalance handling for rare attacks (R2L/U2R)."""
import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np, pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from data_loader import load_raw, preprocess

RARE = ['R2L', 'U2R']


def per_class_recall(y_te, pred, classes):
    r = recall_score(y_te, pred, average=None, labels=range(len(classes)), zero_division=0)
    return dict(zip(classes, r))


def run():
    train, test = load_raw()
    X_tr, X_te, y_tr, y_te, feats, le = preprocess(train, test, task='multiclass')
    classes = list(le.classes_)
    print('Class counts (train):', {classes[i]: int(c) for i, c in
          sorted(Counter(y_tr).items())})

    results, recalls = [], {}

    # --- 1. Baseline
    m = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                      random_state=42, n_jobs=-1, eval_metric='mlogloss',
                      tree_method='hist')
    m.fit(X_tr, y_tr); p = m.predict(X_te)
    results.append(('Baseline XGBoost', accuracy_score(y_te,p), f1_score(y_te,p,average='macro',zero_division=0)))
    recalls['Baseline XGBoost'] = per_class_recall(y_te, p, classes)

    # --- 2. Class weighting (cost-sensitive learning)
    cw = {c: len(y_tr)/(len(classes)*n) for c, n in Counter(y_tr).items()}
    sw = np.array([cw[y] for y in y_tr])
    m2 = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                       random_state=42, n_jobs=-1, eval_metric='mlogloss',
                       tree_method='hist')
    m2.fit(X_tr, y_tr, sample_weight=sw); p2 = m2.predict(X_te)
    results.append(('+ Cost-Sensitive Weights', accuracy_score(y_te,p2), f1_score(y_te,p2,average='macro',zero_division=0)))
    recalls['+ Cost-Sensitive Weights'] = per_class_recall(y_te, p2, classes)

    # --- 3. SMOTE oversampling of rare classes
    counts = Counter(y_tr)
    strat = {c: max(n, 5000) for c, n in counts.items()
             if classes[c] in RARE}
    sm = SMOTE(sampling_strategy=strat, random_state=42, k_neighbors=3)
    Xr, yr = sm.fit_resample(X_tr, y_tr)
    print('After SMOTE:', {classes[i]: int(c) for i, c in sorted(Counter(yr).items())})
    m3 = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                       random_state=42, n_jobs=-1, eval_metric='mlogloss',
                       tree_method='hist')
    m3.fit(Xr, yr); p3 = m3.predict(X_te)
    results.append(('+ SMOTE', accuracy_score(y_te,p3), f1_score(y_te,p3,average='macro',zero_division=0)))
    recalls['+ SMOTE'] = per_class_recall(y_te, p3, classes)

    # --- 4. SMOTE + cost-sensitive (proposed)
    cw4 = {c: len(yr)/(len(classes)*n) for c, n in Counter(yr).items()}
    sw4 = np.array([cw4[y] for y in yr])
    m4 = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.08,
                       subsample=0.9, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1, eval_metric='mlogloss',
                       tree_method='hist')
    m4.fit(Xr, yr, sample_weight=sw4); p4 = m4.predict(X_te)
    results.append(('PROPOSED (SMOTE+Cost+Tuned)', accuracy_score(y_te,p4), f1_score(y_te,p4,average='macro',zero_division=0)))
    recalls['PROPOSED (SMOTE+Cost+Tuned)'] = per_class_recall(y_te, p4, classes)

    df = pd.DataFrame(results, columns=['Method','Accuracy','Macro-F1'])
    print('\n' + '='*70); print('  ABLATION STUDY: IMBALANCE HANDLING'); print('='*70)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    rdf = pd.DataFrame(recalls).T[classes]
    print('\n--- PER-CLASS RECALL ---')
    print(rdf.to_string(float_format=lambda x: f'{x:.4f}'))

    df.to_csv('results/ablation.csv', index=False)
    rdf.to_csv('results/per_class_recall.csv')

    best = df.iloc[-1]['Method']
    print(f'\n--- FINAL REPORT: {best} ---')
    print(classification_report(y_te, p4, target_names=classes, zero_division=0))
    np.save('results/preds_proposed.npy', p4)
    np.save('results/y_test.npy', y_te)
    return df, rdf

if __name__ == '__main__':
    run()
