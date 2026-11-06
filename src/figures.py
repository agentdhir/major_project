"""Publication-quality figures for the paper/presentation."""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_loader import load_raw

plt.rcParams.update({'figure.dpi':150,'savefig.dpi':300,'font.size':10,
                     'axes.grid':True,'grid.alpha':0.3,'savefig.bbox':'tight'})
C = '#2E5A88'; C2 = '#C0392B'
FIG = 'results/figures'

# Fig 1: class distribution
tr, te = load_raw()
order = ['Normal','DoS','Probe','R2L','U2R']
fig, ax = plt.subplots(1,2, figsize=(11,4))
for a,(d,t) in zip(ax,[(tr,'Training set (KDDTrain+)'),(te,'Test set (KDDTest+)')]):
    v = d['category'].value_counts().reindex(order)
    b = a.bar(order, v.values, color=[C,C,C,C2,C2])
    a.set_title(t, fontweight='bold'); a.set_ylabel('Records'); a.set_yscale('log')
    for r,val in zip(b,v.values):
        a.text(r.get_x()+r.get_width()/2, val, f'{int(val):,}', ha='center', va='bottom', fontsize=8)
plt.suptitle('Fig. 1  Class imbalance in NSL-KDD (log scale) — R2L/U2R in red',
             fontweight='bold', y=1.06)
plt.tight_layout()
plt.savefig(f'{FIG}/fig1_class_distribution.png'); plt.close()

# Fig 2: model comparison
db = pd.read_csv('results/results_binary.csv').sort_values('F1')
fig, ax = plt.subplots(figsize=(9,4.5))
y = np.arange(len(db)); h=0.38
ax.barh(y-h/2, db['Accuracy'], h, label='Accuracy', color=C)
ax.barh(y+h/2, db['F1'], h, label='F1-score', color='#E8A33D')
ax.set_yticks(y); ax.set_yticklabels(db['Model']); ax.set_xlim(0,1)
for i,(a_,f_) in enumerate(zip(db['Accuracy'],db['F1'])):
    ax.text(a_+.01,i-h/2,f'{a_:.3f}',va='center',fontsize=8)
    ax.text(f_+.01,i+h/2,f'{f_:.3f}',va='center',fontsize=8)
ax.set_title('Fig. 2  Baseline model comparison (binary, KDDTest+)',fontweight='bold')
ax.legend(loc='lower right')
plt.savefig(f'{FIG}/fig2_model_comparison.png'); plt.close()

# Fig 3: ablation
ab = pd.read_csv('results/ablation.csv')
fig, ax = plt.subplots(figsize=(9,4.5))
x = np.arange(len(ab)); w=0.38
ax.bar(x-w/2, ab['Accuracy'], w, label='Accuracy', color=C)
ax.bar(x+w/2, ab['Macro-F1'], w, label='Macro-F1', color=C2)
for i,(a_,f_) in enumerate(zip(ab['Accuracy'],ab['Macro-F1'])):
    ax.text(i-w/2,a_+.008,f'{a_:.3f}',ha='center',fontsize=8)
    ax.text(i+w/2,f_+.008,f'{f_:.3f}',ha='center',fontsize=8,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([m.replace(' (','\n(') for m in ab['Method']], fontsize=8)
ax.set_ylim(0,1); ax.legend()
ax.set_title('Fig. 3  Ablation: effect of imbalance handling',fontweight='bold')
plt.savefig(f'{FIG}/fig3_ablation.png'); plt.close()

# Fig 4: per-class recall improvement
rc = pd.read_csv('results/per_class_recall.csv', index_col=0)
fig, ax = plt.subplots(figsize=(9,4.5))
x = np.arange(len(rc.columns)); w=0.2
for i,(name,row) in enumerate(rc.iterrows()):
    ax.bar(x+(i-1.5)*w, row.values, w, label=name)
ax.set_xticks(x); ax.set_xticklabels(rc.columns); ax.set_ylabel('Recall')
ax.set_title('Fig. 4  Per-class recall: rare classes (R2L, U2R) improve most',fontweight='bold')
ax.legend(fontsize=8); ax.set_ylim(0,1.05)
plt.savefig(f'{FIG}/fig4_per_class_recall.png'); plt.close()

# Fig 5: confusion matrix
yt = np.load('results/y_test.npy'); yp = np.load('results/preds_proposed.npy')
cm = confusion_matrix(yt, yp)
order_ = ['DoS','Normal','Probe','R2L','U2R']
cmn = cm.astype(float)/cm.sum(axis=1,keepdims=True)
fig, ax = plt.subplots(1,2,figsize=(12,4.6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=order_,
            yticklabels=order_, ax=ax[0], cbar=False)
ax[0].set_title('Counts',fontweight='bold'); ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('Actual')
sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', xticklabels=order_,
            yticklabels=order_, ax=ax[1], cbar=False, vmin=0, vmax=1)
ax[1].set_title('Row-normalised (recall)',fontweight='bold'); ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('Actual')
plt.suptitle('Fig. 5  Confusion matrix — proposed model',fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG}/fig5_confusion_matrix.png'); plt.close()

print('Figures written to', FIG)
