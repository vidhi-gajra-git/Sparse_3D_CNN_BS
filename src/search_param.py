import itertools
import pandas as pd
#  needs lot of changes 
# from src.data import load_hsi
from src.model import HybridModel
from src.train import train_model
from src.utils import compute_metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
def validation_score(
    params,
    cube, X, Y,
    train_idx, val_idx,
    device,
    fixed_topk=20
):
    """
    Returns a single scalar validation score for hyperparameter search.
    """

    # -------- build model --------
    model = HybridModel(
        window_size=params["window_size"],
        H=cube.shape[0],
        W=cube.shape[1],
        latent_dim=params["latent_dim"],
        decoder_dropout=params["decoder_dropout"],
        decoder_hidden_mult=params["decoder_hidden_mult"],
        use_attention=False
    ).to(device)

    # -------- train model --------
    model, Z, C, A, history, recon = train_model(
        model=model,
        cube=cube,
        epochs=10,                  # SHORT training for search
        batch_size=16,
        lr=params["lr"],
        alpha=params["alpha"],
        beta=params["beta"],
        lambda_l1=params["lambda_l1"],
        lambda_tv=params["lambda_tv"],
        C_init_scale=params["C_init_scale"],
        C_lr_mult=params["C_lr_mult"],
        lambda_msssim=params["lambda_msssim"],
        msssim_levels=params["msssim_levels"],
        device=device,
        verbose=False
    )

    # -------- band selection --------
    band_scores = np.sum(np.abs(A), axis=1)
    topk_bands = np.argsort(band_scores)[-fixed_topk:][::-1]

    # -------- prepare val data --------
    X_val = X[val_idx][:, topk_bands]
    Y_val = Y[val_idx]

    X_tr = X[train_idx][:, topk_bands]
    Y_tr = Y[train_idx]

    # -------- classifier (fast) --------
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
     
    )
    clf.fit(X_tr, Y_tr)

    Y_pred = clf.predict(X_val)

    oa, aa, kappa = compute_metrics(Y_val, Y_pred)

    # -------- combined validation score --------
    score = (oa + aa + kappa) / 3.0

    return score

def run_hyperparam_search(cfg,cube, X,Y):
    search_space = {
        "latent_dim": [128, 256, 384],
        "decoder_dropout": [0.0, 0.1],
        "lr": [1e-4, 1e-3],
        "alpha": [5.0, 10.0, 20.0],
        "window_size": [3,5],
        "decoder_hidden_mult": [1,4,8],
        "beta": [1e-4,1e-5],
        "lambda_l1": [5e-4,1e-4,5e-5],
        "lambda_tv": [5e-4,1e-4,5e-5],
        "C_init_scale": [1,1e-1,1e-2],
        "C_lr_mult": [1.0,5.0,10.0,50.0],
        "lambda_msssim":  [1e-3,1e-4],
        "msssim_levels":  [2,3,5],

        
    }

    records = []
    best_score = -1
    best_params = {}

    for vals in itertools.product(*search_space.values()):
        params = dict(zip(search_space.keys(), vals))

        score = validation_score(params)  # replace with real val
        records.append({**params, "score": score})

        if score > best_score:
            best_score = score
            best_params = params

    return pd.DataFrame(records).sort_values("score", ascending=False), best_params
