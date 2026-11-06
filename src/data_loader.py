"""NSL-KDD loading, attack-category mapping and preprocessing."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

CATEGORICAL = ['protocol_type', 'service', 'flag']

# Standard NSL-KDD 4-class attack taxonomy
ATTACK_MAP = {
    'normal': 'Normal',
    # DoS
    'neptune':'DoS','back':'DoS','land':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','mailbomb':'DoS','apache2':'DoS','processtable':'DoS',
    'udpstorm':'DoS','worm':'DoS',
    # Probe
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L',
    'xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
    # U2R
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'ps':'U2R','sqlattack':'U2R','xterm':'U2R',
}


def load_raw(train_path='data/KDDTrain+.txt', test_path='data/KDDTest+.txt'):
    train = pd.read_csv(train_path, names=COLUMNS)
    test = pd.read_csv(test_path, names=COLUMNS)
    for df in (train, test):
        df.drop(columns=['difficulty'], inplace=True)
        df['category'] = df['label'].map(ATTACK_MAP).fillna('Unknown')
        df['binary'] = (df['label'] != 'normal').astype(int)
    return train, test


def preprocess(train, test, task='binary'):
    """One-hot encode categoricals (aligned across splits), scale numerics."""
    target = 'binary' if task == 'binary' else 'category'

    X_tr = train.drop(columns=['label', 'category', 'binary'])
    X_te = test.drop(columns=['label', 'category', 'binary'])

    X_tr = pd.get_dummies(X_tr, columns=CATEGORICAL)
    X_te = pd.get_dummies(X_te, columns=CATEGORICAL)
    # Align test columns to training feature space (test has unseen services)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.astype(float))
    X_te_s = scaler.transform(X_te.astype(float))

    y_tr, y_te = train[target].values, test[target].values
    if task != 'binary':
        le = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
        y_tr, y_te = le.transform(y_tr), le.transform(y_te)
        return X_tr_s, X_te_s, y_tr, y_te, list(X_tr.columns), le

    return X_tr_s, X_te_s, y_tr, y_te, list(X_tr.columns), None
