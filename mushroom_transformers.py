import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class TargetEncoderOOF(BaseEstimator, TransformerMixin):
    """Out-of-fold target encoder (sklearn-compatible).

    Computes per-category full means in fit(), stores OOF encodings for
    inspection, and maps categories to the stored full means in transform().
    """
    def __init__(self, cols=None, n_splits=5, random_state=0):
        self.cols = cols or []
        self.n_splits = n_splits
        self.random_state = random_state
        self.full_means_ = {}
        self.oof_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError('y is required for TargetEncoderOOF.fit')

        df_local = X.copy()
        df_local['_target_'] = pd.Series(y, index=df_local.index)

        # compute full means per category for each column
        for c in self.cols:
            self.full_means_[c] = df_local.groupby(c)['_target_'].mean().to_dict()

        # compute out-of-fold encodings for inspection
        oof_df = pd.DataFrame(index=df_local.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for c in self.cols:
            oof_series = pd.Series(index=df_local.index, dtype='float64')
            for train_idx, val_idx in kf.split(df_local):
                tr = df_local.iloc[train_idx]
                va = df_local.iloc[val_idx]
                means = tr.groupby(c)['_target_'].mean()
                oof_series.iloc[val_idx] = va[c].map(means)
            oof_series.fillna(df_local['_target_'].mean(), inplace=True)
            oof_df[c] = oof_series

        self.oof_ = oof_df
        # store global mean for unseen category fallback
        self.global_mean_ = float(df_local['_target_'].mean())
        return self

    def transform(self, X):
        Xdf = X.copy()
        n = len(Xdf)
        out = np.zeros((n, len(self.cols)), dtype='float32')
        for i, c in enumerate(self.cols):
            mapping = self.full_means_.get(c, {})
            default = getattr(self, 'global_mean_', 0.0)
            mapped = Xdf[c].map(mapping).fillna(default).astype('float32')
            out[:, i] = mapped.values

        return out
