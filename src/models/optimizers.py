from nevergrad.optimization import optimizerlib
from nevergrad import instrumentation as inst
from sklearn.metrics import f1_score

from nevergrad.optimization import optimizerlib
from nevergrad import instrumentation as inst
from sklearn.metrics import f1_score


def optimize_f1_threshold(predictions, labels, budget=200, num_workers=1):
    """Optimizes the cutoff threshold to maximise the utility score.

     """

    # Set optimizer and instrumentation (bounds)
    instrum = inst.Instrumentation(*[inst.var.Array(1).asscalar().bounded(-0.2, 0.2)])
    optimizer = optimizerlib.TwoPointsDE(instrumentation=instrum, budget=budget, num_workers=num_workers)

    # Optimize
    recommendation = optimizer.optimize(
        lambda thresh: -f1_score(labels,(predictions[:,1]>thresh).astype('int'),average='binary')
    )

    # Get the threshold and return the score
    threshold = recommendation.args[0]

    return threshold


def optimize_utility_threshold(predictions, definition='t_sepsis_min',scores=None, idxs=None, budget=200, num_workers=1):
    """Optimizes the cutoff threshold to maximise the utility score.

    Sepsis predictions must be binary labels. Dependent on where these are in a patients time-series, we achieve a
    different utility score for that prediction. Our current methodology involves regressing against the utility
    function such that our output predictions are number in $\mathbb{R}$ with the expectation that larger values
    correspond to a higher likelihood of sepsis. To convert onto binary predictions we must choose some cutoff value,
    `thresh` with which to predict 1 (sepsis) if `pred > thresh` else 0. Given a set of predictions, this function
    optimizes that `thresh` value such that we would achieve the maximum utility score. This `thresh` value can now be
    used as the final step in the full model.

    This function would take a huge amount of time to compute if we did not compute for every patient, the utility
    of a zero or of a one prediction at each time-point in their series. The downside of this is that we must always
    specify the indexes from the full dataset from which the predictions correspond to, as we load in this precomputed
    scores tensor and query at the indexes to get the score.

    Args:
        predictions (torch.Tensor): The predictions (or a subset of the predictions) on the data. NOTE: if idxs is
            specified, predictions must be specified as the subset of predictions corresponding to those indexes. That
            is predictions[idxs].
        idxs (torch.Tensor): The index locations of the predictions in the full dataset.
        budget (int): The number optimizer iterations.
        num_workers (int): The number of parallel workers in the optimization.

    Returns:
        float: The estimation of the optimal cutoff threshold for which to predict sepsis.
    """
    # Load the full version of the scores if not pre-specified
    if scores is None:
        scores = np.load('./scores'+definition[1:]+'.npy')

    if idxs is not None:
        scores = scores[idxs]

    # Set optimizer and instrumentation (bounds)
    instrum = inst.Instrumentation(*[inst.var.Array(1).asscalar().bounded(-0.2, 0.2)])
    optimizer = optimizerlib.TwoPointsDE(instrumentation=instrum, budget=budget, num_workers=num_workers)

    # Optimize
    recommendation = optimizer.optimize(
        lambda thresh: -compute_utility(scores, predictions, thresh)
    )

    # Get the threshold and return the score
    threshold = recommendation.args[0]

    return threshold
    
def compute_utility(scores, predictions, thresh):
    """Computes the utility score of the predictions given the scores for predicting 0 or 1 at each timepoint.
    Args:
        scores (torch.Tensor): The scores tensor that gives the score of a 0 or 1 at each timepoint.
        predictions (torch.Tensor): The predictions with indexed aligned with the score indexes.
        thresh (float): The threshold to cut the predictions off at.
    Returns:
        float: The normalised score.
    """
    len_scores, len_preds = len(scores), len(predictions)
    assert len_scores == len_preds, 'Num predictions: {}, Num scores: {}. These must be the same, ensure you have made ' \
                                    'the appropriate index reduction before calling this function'.format(len_scores, len_preds)

    # Precompute the inaction and perfect scores
    inaction_score = scores[:, 0].sum()
    perfect_score = scores[:, [0, 1]].max(axis=1).sum()

    # Apply the threshold
    predictions = (predictions > thresh).astype(int)

    # Get the actual score
    actual_score = scores[:, 1][predictions == 1].sum() + scores[:, 0][predictions == 0].sum()

    # Get the normalized score
    normalized_score = (actual_score - inaction_score) / (perfect_score - inaction_score)

    return normalized_score.item()
