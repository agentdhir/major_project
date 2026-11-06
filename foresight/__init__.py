"""
FORESIGHT — deep-learning early warning for network attacks.

Modules (see README.md and docs/METHODOLOGY.md):
    config      every hyper-parameter          simulator   kill-chain campaign streams
    data        NSL-KDD pools, feature encoder  windows     GPU sliding-window batches
    models      FORESIGHT-Net + baselines      train       training, site adaptation
    baselines   reactive IDS, rule, XGBoost    zero_day    autoencoder zero-day guard
    metrics     step- and event-level metrics   explain     attention + attribution
    experiments pipeline stages                figures     all figures
    inference   deployed-system API (dashboard, monitor)
"""
