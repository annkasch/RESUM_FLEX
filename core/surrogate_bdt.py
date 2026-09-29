"""Boosted binary classifier over voxel coordinates and event features."""
from pathlib import Path
import numpy as np
from scipy.special import expit
from core.precision_recall import precision_recall


def event_features(batch):
    """Flatten [voxel,event] order without using labels or context statistics."""
    parts=[]
    if batch.theta is not None:
        parts.append(np.broadcast_to(batch.theta[:,None,:],
                     (batch.batch_size,batch.n_events,batch.theta.shape[-1])))
    if batch.phi is not None:
        parts.append(batch.phi)
    if not parts:
        raise ValueError('BDT requires at least one feature')
    return np.concatenate(parts,axis=-1).reshape(batch.batch_size*batch.n_events,-1)


def build_bdt(config):
    # Optional dependency: importing the other surrogates does not require sklearn.
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(loss='log_loss',
        learning_rate=config.learning_rate,max_iter=config.eval_every,
        max_leaf_nodes=config.max_leaf_nodes,max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,l2_regularization=config.l2_regularization,
        max_bins=config.max_bins,categorical_features=None,early_stopping=False,
        warm_start=True,random_state=config.seed,class_weight=None)


def score_bdt(model,batch):
    scores=np.asarray(model.decision_function(event_features(batch)),dtype=float)
    labels=batch.labels.ravel()
    if not np.isfinite(scores).all():raise ValueError('Nonfinite BDT scores')
    probabilities=expit(scores)
    observed=batch.labels.mean(1)
    predicted=probabilities.reshape(batch.batch_size,batch.n_events).mean(1)
    pr=precision_recall(labels,scores)
    residual=predicted-observed
    metrics=dict(voxels=batch.batch_size,events=len(labels),positives=int(labels.sum()),
        mean_observed=float(observed.mean()),mean_predicted=float(predicted.mean()),
        ratio_of_means=float(predicted.mean()/observed.mean()),
        pearson_r=float(np.corrcoef(observed,predicted)[0,1]) if predicted.std()>0 and observed.std()>0 else None,
        voxel_rate_mae=float(np.abs(residual).mean()),
        voxel_rate_rmse=float(np.sqrt(np.square(residual).mean())),
        bernoulli_log_loss=float(np.where(labels==1,np.logaddexp(0,-scores),np.logaddexp(0,scores)).mean()),
        brier_score=float(np.square(probabilities-labels).mean()),
        average_precision=pr['average_precision'],prevalence=pr['prevalence'])
    arrays=dict(logits=scores,labels=labels,observed=observed,predicted=predicted,
                precision=pr['precision'],recall=pr['recall'],logit_threshold=pr['logit_threshold'])
    return metrics,arrays


def save_bdt(path,model,metadata):
    import joblib
    import sklearn
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model_kind='hist_gradient_boosted_detection',model=model,
                     sklearn_version=sklearn.__version__,metadata=metadata),path)


def load_bdt(path):
    import joblib
    payload=joblib.load(path)
    if payload.get('model_kind')!='hist_gradient_boosted_detection':
        raise ValueError('Expected a boosted detection model')
    return payload['model'],payload
