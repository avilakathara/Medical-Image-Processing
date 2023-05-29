
import numpy as np

def evaluate_uncertainty(ground_truth, segmentation, uncertainty_field):
    gt = ground_truth
    s = segmentation
    uf = uncertainty_field

    vtp = np.vectorize(true_positive)
    vtn = np.vectorize(true_negative)
    vfp = np.vectorize(false_positive)
    vfn = np.vectorize(false_negative)

    return (
        produce_output("TP", vtp(gt, s, uf)),
        produce_output("TN", vtn(gt, s, uf)),
        produce_output("FP", vfp(gt, s, uf)),
        produce_output("FN", vfn(gt, s, uf))
    )

# DETECT FOR INCORRECT SEGMENT
def true_positive(gt, s, uf):
    return abs(gt - s) * uf

# DO NOT DETECT FOR CORRECT SEGMENT
def true_negative(gt, s, uf):
    return (1 - abs(gt - s)) * (1 - uf)

# DETECTED BUT CORRECT SEGMENT
def false_positive(gt, s, uf):
    return (1 - abs(gt - s)) * uf

# INCORRECT SEGMENT BUT NOT DETECTED
def false_negative(gt, s, uf):
    return abs(gt - s) * (1 - uf)

def produce_output(t, arr):
    arr = arr[arr.nonzero()]
    return t + ': ' + str(100 * round(np.mean(arr), 3)) + '%'
