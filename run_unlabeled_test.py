"""Run saved mushroom model on an unlabeled test dataset and do simple robustness checks.

Creates `predictions_unlabeled.csv` and prints a short summary. Also runs two lightweight
perturbation tests (randomize odor and mask odor) to measure prediction stability without labels.

Usage (from repo root):
    python run_unlabeled_test.py

The script prefers `mushroom_engineered.csv` (created by the notebook). If only
`mushroom_structured.csv` is present it will do minimal feature engineering to create
the columns the pipeline expects.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys


# Recreate TargetEncoderOOF here so joblib can unpickle pipelines that reference it.
from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoderOOF(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that computes out-of-fold (OOF) target-encoding for specified categorical columns during fit.
    Compatible with the implementation used when the model was trained and saved.
    """
    def __init__(self, cols=None, n_splits=5, random_state=0):
        self.cols = cols or []
        self.n_splits = n_splits
        self.random_state = random_state
        self.full_means_ = {}
        self.oof_ = None

    def fit(self, X, y):
        import pandas as _pd
        from sklearn.model_selection import KFold
        if y is None:
            raise ValueError('y is required for TargetEncoderOOF.fit')
        df_local = X.copy()
        df_local['_target_'] = _pd.Series(y, index=df_local.index)
        # compute full means
        for c in self.cols:
            self.full_means_[c] = df_local.groupby(c)['_target_'].mean().to_dict()
        # compute OOF per column (kept for inspection)
        oof_df = _pd.DataFrame(index=df_local.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for c in self.cols:
            oof_series = _pd.Series(index=df_local.index, dtype='float64')
            for train_idx, val_idx in kf.split(df_local):
                train = df_local.iloc[train_idx]
                means = train.groupby(c)['_target_'].mean()
                mapped = df_local.iloc[val_idx][c].map(means).astype('float64')
                oof_series.iloc[val_idx] = mapped
            # fillna with global mean
            global_mean = float(df_local['_target_'].mean())
            oof_series.fillna(global_mean, inplace=True)
            oof_df[f'{c}_te_transform'] = oof_series
        self.oof_ = oof_df
        return self

    def transform(self, X):
        """Return a plain 2D numpy array (float32) with columns in the same order as self.cols"""
        import numpy as _np
        import pandas as _pd
        rows = []
        for c in self.cols:
            means = self.full_means_.get(c, {})
            # fallback global mean
            global_mean = float(_pd.Series(list(means.values())).mean()) if len(means) > 0 else 0.0
            col = X[c].map(means).fillna(global_mean).astype('float32')
            rows.append(col.values)
        if len(rows) == 0:
            return _np.zeros((len(X), 0), dtype='float32')
        arr = _np.vstack(rows).T.astype('float32')
        return arr

    def get_feature_names_out(self, input_features=None):
        return [f'{c}_te_transform' for c in self.cols]


ROOT = Path('.').resolve()
ENG_CSV = ROOT / 'mushroom_engineered.csv'
STRUCT_CSV = ROOT / 'mushroom_structured.csv'
MODEL_CANDIDATES = [ROOT / 'best_mushroom_model_oof.joblib', ROOT / 'best_mushroom_model.joblib']
OUT_PRED = ROOT / 'predictions_unlabeled.csv'


def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            print(f'Loading model from: {p}')
            return joblib.load(p)
    raise FileNotFoundError(f'No model found at any of: {MODEL_CANDIDATES}')


def load_data():
    # Prefer engineered CSV (has features used by the pipeline). Otherwise create minimal engineered view.
    if ENG_CSV.exists():
        print(f'Loading engineered data from {ENG_CSV}')
        df = pd.read_csv(ENG_CSV)
        return df

    if STRUCT_CSV.exists():
        print(f'Loading structured data from {STRUCT_CSV} and doing minimal feature engineering')
        s = pd.read_csv(STRUCT_CSV)
        # minimal engineering to create the columns the pipeline expects
        s['label_num'] = s.get('label_num')  # may be missing
        s['odor_readable'] = s.get('odor_readable', s.get('odor'))
        s['cap_surface_readable'] = s.get('cap_surface_readable', s.get('cap_surface'))
        s['cap_shape_readable'] = s.get('cap_shape_readable', s.get('cap_shape'))
        s['cap_color_readable'] = s.get('cap_color_readable', s.get('cap_color'))
        # engineered composites used in notebook
        s['cap_shape_surface'] = s['cap_shape_readable'].astype(str).fillna('NA') + '_' + s['cap_surface_readable'].astype(str).fillna('NA')
        s['odor_capcolor'] = s['odor_readable'].astype(str).fillna('NA') + '|' + s['cap_color_readable'].astype(str).fillna('NA')
        # lightweight numeric features
        s['odor_freq'] = s['odor_readable'].astype(object).fillna('MISSING').map(lambda x: 0)
        try:
            odor_counts = s['odor_readable'].astype(object).fillna('MISSING').map(lambda x: s['odor_readable'].value_counts().get(x, 0))
            s['odor_freq'] = odor_counts
        except Exception:
            s['odor_freq'] = 0
        s['odor_poison_rate'] = 0.0
        s['has_stalk_root'] = (~s.get('stalk_root_readable', pd.Series([np.nan]*len(s))).isna()).astype('int8')
        return s

    raise FileNotFoundError('No suitable CSV found (looked for mushroom_engineered.csv or mushroom_structured.csv)')


def ensure_features(df, cat_for_te, num_features):
    missing = [c for c in (cat_for_te + num_features) if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing required columns for model inference: {missing}')


def predict_on_unlabeled(model, df):
    # model expects the columns: cat_for_te + num_features (see notebook). We'll infer them here.
    cat_for_te = ['cap_shape_surface', 'odor_capcolor']
    num_features = ['odor_freq', 'odor_poison_rate', 'has_stalk_root']
    ensure_features(df, cat_for_te, num_features)

    X = df[cat_for_te + num_features].copy()
    # Some sklearn pipelines expect exact dtypes; coerce to object/float as reasonable
    for c in cat_for_te:
        X[c] = X[c].astype(object)
    for c in num_features:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

    # Try predict_proba, otherwise fallback to predict
    preds = None
    probs = None
    try:
        probs = model.predict_proba(X)
        # assume binary and poisonous is class 1 if exists
        if probs.shape[1] == 2:
            prob_poison = probs[:, 1]
        else:
            # fallback: take last column
            prob_poison = probs[:, -1]
        preds = (prob_poison >= 0.5).astype(int)
    except Exception:
        print('predict_proba not available, using predict() then casting to int')
        preds = np.asarray(model.predict(X)).astype(int)
        prob_poison = preds.astype(float)

    out = df.copy().reset_index(drop=True)
    out['pred_label_num'] = preds
    out['pred_poison_prob'] = prob_poison
    out.to_csv(OUT_PRED, index=False)
    print(f'Saved predictions to: {OUT_PRED} (rows={len(out)})')

    # print summary
    print('\nPrediction summary:')
    print('Predicted poisonous (1) count:', int((out['pred_label_num'] == 1).sum()))
    print('Predicted edible (0) count:', int((out['pred_label_num'] == 0).sum()))
    print('Mean predicted poison probability:', float(out['pred_poison_prob'].mean()))
    return out, cat_for_te, num_features


def perturbation_tests(model, df, cat_for_te, num_features):
    """Run quick perturbations to estimate prediction stability without labels.

    Tests implemented:
      - randomize odor values (shuffle within column)
      - mask odor values (set to string 'MISSING')
    These measure how many predictions change vs baseline.
    """
    base, _, _ = predict_on_unlabeled(model, df)
    Xbase = df[cat_for_te + num_features].copy()

    results = {}

    # 1) shuffle odor component inside odor_readable / odor_capcolor: shuffle odor_capcolor
    df_shuffled = df.copy()
    df_shuffled['odor_capcolor'] = np.random.permutation(df_shuffled['odor_capcolor'].values)
    out_shuf, _, _ = predict_on_unlabeled(model, df_shuffled)
    changed = (out_shuf['pred_label_num'].values != base['pred_label_num'].values).sum()
    results['shuffle_odor_capcolor_changed'] = int(changed)
    results['shuffle_odor_capcolor_changed_pct'] = float(changed) / len(df) * 100.0

    # 2) mask odor -> set odor part to 'MISSING' for odor_capcolor
    df_mask = df.copy()
    # keep capcolor but replace odor part before '|'
    if '|' in str(df_mask['odor_capcolor'].iloc[0]):
        df_mask['odor_capcolor'] = df_mask['odor_capcolor'].astype(str).map(lambda s: 'MISSING|' + s.split('|')[-1])
    else:
        df_mask['odor_capcolor'] = 'MISSING'
    out_mask, _, _ = predict_on_unlabeled(model, df_mask)
    changed2 = (out_mask['pred_label_num'].values != base['pred_label_num'].values).sum()
    results['mask_odor_capcolor_changed'] = int(changed2)
    results['mask_odor_capcolor_changed_pct'] = float(changed2) / len(df) * 100.0

    print('\nPerturbation test results:')
    for k, v in results.items():
        print(f'{k}: {v}')

    # Save a compact CSV of comparisons
    comp = pd.DataFrame({
        'base_pred': base['pred_label_num'],
        'shuf_pred': out_shuf['pred_label_num'],
        'mask_pred': out_mask['pred_label_num']
    })
    comp.to_csv(ROOT / 'predictions_perturbation_compare.csv', index=False)
    print('Saved perturbation comparison to predictions_perturbation_compare.csv')


def try_shap_analysis(model, df, cat_for_te, num_features):
    try:
        import shap
        import numpy as _np
        X = df[cat_for_te + num_features].copy()
        # transform via pipeline preprocessor if present
        if hasattr(model, 'named_steps') and 'pre' in model.named_steps:
            Xt = model.named_steps['pre'].transform(X)
            try:
                Xt = _np.asarray(Xt, dtype=float)
            except Exception:
                Xt = _np.asarray(Xt)
        else:
            # if model pipeline not present, try passing X directly
            Xt = _np.asarray(X)

        # find classifier
        clf = None
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            clf = model.named_steps['clf']
        else:
            # maybe model itself is estimator
            clf = model

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(Xt)
        print('\nComputed SHAP values (shape info):', np.shape(shap_vals))
        # Save a small summary if possible
        try:
            shap.summary_plot(shap_vals, features=X if isinstance(X, pd.DataFrame) else None, show=False)
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig('shap_unlabeled_summary.png', dpi=150)
            plt.close()
            print('Saved SHAP summary to shap_unlabeled_summary.png')
        except Exception as e:
            print('Could not make SHAP plot:', e)
    except Exception as e:
        print('SHAP analysis skipped (shap not installed or failed):', str(e))


def main():
    model = load_model()
    df = load_data()

    # make unlabeled copy
    unlabeled = df.copy()
    for c in ['label', 'label_num', 'poisonous']:
        if c in unlabeled.columns:
            unlabeled = unlabeled.drop(columns=[c])

    # run baseline predictions
    out, cat_for_te, num_features = predict_on_unlabeled(model, unlabeled)

    # quick perturbation tests
    perturbation_tests(model, unlabeled, cat_for_te, num_features)

    # optional: try SHAP explainability on unlabeled set
    try_shap_analysis(model, unlabeled, cat_for_te, num_features)


if __name__ == '__main__':
    main()
