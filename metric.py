from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


def regression_metric(y_true, y_pred):
    r, p = pearsonr(y_pred, y_true)
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'r': r,
        'p': p,
    }

