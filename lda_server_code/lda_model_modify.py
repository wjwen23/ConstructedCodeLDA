from gensim.models.ldamodel import *
import numpy as np

DTYPE_TO_EPS = {
    'float16': 1e-5,
    'float32': 1e-25,      # <<<<=========== THE VALUE I CHANGE ===========
    'float64': 1e-100,
}


def inference(self, chunk, collect_sstats=False):
    try:
        len(chunk)
    except TypeError:
        # convert iterators/generators to plain list, so we have len() etc.
        chunk = list(chunk)
    if len(chunk) > 1:
        logger.debug("performing inference on a chunk of %i documents", len(chunk))

    # Initialize the variational distribution q(theta|gamma) for the chunk
    gamma = self.random_state.gamma(100., 1. / 100., (len(chunk), self.num_topics)).astype(self.dtype, copy=False)
    Elogtheta = dirichlet_expectation(gamma)
    expElogtheta = np.exp(Elogtheta)

    assert Elogtheta.dtype == self.dtype
    assert expElogtheta.dtype == self.dtype

    if collect_sstats:
        sstats = np.zeros_like(self.expElogbeta, dtype=self.dtype)
    else:
        sstats = None
    converged = 0

    for d, doc in enumerate(chunk):
        if len(doc) > 0 and not isinstance(doc[0][0], six.integer_types + (np.integer,)):
            # make sure the term IDs are ints, otherwise np will get upset
            ids = [int(idx) for idx, _ in doc]
        else:
            ids = [idx for idx, _ in doc]
        cts = np.array([cnt for _, cnt in doc], dtype=self.dtype)
        gammad = gamma[d, :]
        Elogthetad = Elogtheta[d, :]
        expElogthetad = expElogtheta[d, :]
        expElogbetad = self.expElogbeta[:, ids]

        # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
        # phinorm is the normalizer.
        # TODO treat zeros explicitly, instead of adding epsilon?
        eps = DTYPE_TO_EPS['float32']
        phinorm = np.dot(expElogthetad, expElogbetad) + eps

        # Iterate between gamma and phi until convergence
        for _ in range(self.iterations):
            lastgamma = gammad
            gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
            Elogthetad = dirichlet_expectation(gammad)
            expElogthetad = np.exp(Elogthetad)
            phinorm = np.dot(expElogthetad, expElogbetad) + eps
            # If gamma hasn't changed much, we're done.
            meanchange = mean_absolute_difference(gammad, lastgamma)
            if meanchange < self.gamma_threshold:
                converged += 1
                break
        gamma[d, :] = gammad
        assert gammad.dtype == self.dtype
        if collect_sstats:
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

    if len(chunk) > 1:
        logger.debug("%i/%i documents converged within %i iterations", converged, len(chunk), self.iterations)

    if collect_sstats:
        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats *= self.expElogbeta
        assert sstats.dtype == self.dtype

    assert gamma.dtype == self.dtype
    return gamma, sstats


def modify_lda_inference():
    LdaModel.inference = inference