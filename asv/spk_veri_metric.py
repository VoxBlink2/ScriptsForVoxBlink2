import numpy as np


def length_norm(mat):
    return mat / np.sqrt(np.sum(mat * mat, axis=1))[:, None]


def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided
    to equalize the counts over score partitions (if there is such
    partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones((labels.shape), dtype='f8')

    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')

    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    return fnr, fpr


def compute_eer(fnr, fpr):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det, c_det_ind = min(dcf), np.argmin(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det/c_def, c_det_ind


def compute_equalized_min_cost(labels, scores, ptar=[0.01, 0.001]):
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer = compute_eer(fnr, fpr)
    min_c = 0.
    for pt in ptar:
        tmp, idx = compute_c_norm(fnr, fpr, pt)
        min_c += tmp
    return eer*100, min_c / len(ptar)


class SVevaluation(object):
    def __init__(self, trial_file, utt, embd=None, ptar=[0.01, 0.001]):
        # trials file format: is_target(0 or 1) enrol_utt test_utt
        self.ptar = ptar
        self.update_embd(embd)
        self.utt_idx = {u:i for i, u in enumerate(utt)}
        self.update_trial(trial_file)

    def update_trial(self, trial_file):
        self.labels = [int(line.split()[0]) for line in open(trial_file)]
        self.trial_idx = [[self.utt_idx.get(line.split()[1]), self.utt_idx.get(line.split()[2])] for line in open(trial_file)]
        bad_idx = [i for i, ti in enumerate(self.trial_idx) if None in ti]
        for i in sorted(bad_idx, reverse=True):
            del self.trial_idx[i], self.labels[i]
        self.labels = np.array(self.labels)
        
        if len(bad_idx):
            print('Number of bad trials %d' % len(bad_idx))

    def update_cohort(self, cohort):
        cohort = length_norm(cohort)
        self.score_cohort = self.embd @ cohort.T
        self.idx_cohort = self.score_cohort.argsort()[:, ::-1]
        
    def update_embd(self, embd):
        self.embd = length_norm(embd) if embd is not None else None
        
    def eer_cost(self):      
        scores = [(self.embd[i] * self.embd[j]).sum() for i, j in self.trial_idx] 
        eer, cost = compute_equalized_min_cost(self.labels, np.array(scores), self.ptar)
        return eer, cost    
    
    
