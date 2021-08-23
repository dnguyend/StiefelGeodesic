import numpy as np
from ManNullRange.manifolds.RealFlag import RealFlag
from ManNullRange.manifolds.tools import asym, vecah, unvecah
from scipy.linalg import expm, null_space, expm_frechet, logm
from StiefelGeodesic.manifolds.tools import sbmat, linf
import StiefelGeodesic.manifolds.minimize_lfbgs as mlf

from StiefelGeodesic.manifolds.TrustNCG import _minimize_trust_ncg


class AR(object):
    def __init__(self, A, R):
        self.A = A
        self.R = R

    @property
    def norm(self):
        return np.sqrt(self.dot_prod(self))

    def zeros_like(self):
        return AR(np.zeros_like(self.A),
                  np.zeros_like(self.R))

    @classmethod
    def zerovec(cls, d, k):
        return AR(np.zeros((d, d)),
                  np.zeros((k, d)))

    @property
    def dim(self):
        d = self.A.shape[0]
        k = self.R.shape[0]
        return int(d*(d-1)/2) + k*d

    @classmethod    
    def clear_diag(cls, A, dvec):
        start = 0
        for j in range(1, dvec.shape[0]):
            A[start:start+dvec[j], start:start+dvec[j]] = 0
            start += dvec[j]
        return A

    @classmethod
    def rand(cls, dvec, k):
        d = dvec[1:].sum()
        A = np.random.randn(d, d)
        R = np.random.randn(k, d)
        A = cls.clear_diag(A - A.T, dvec)
        return cls(A, R)

    def dot_prod(self, other):
        return np.sum(self.A*other.A) +\
            np.sum(self.R*other.R)

    def __add__(self, other):
        return self.__class__(
            self.A + other.A,
            self.R + other.R)

    def __neg__(self):
        return self.__class__(
            -1*self.A,
            -1*self.R)        

    def __sub__(self, other):
        return self.__class__(
            self.A - other.A,
            self.R - other.R)

    def scalar_mul(self, other):
        return self.__class__(
            other*self.A,
            other*self.R)

    def __rmul__(self, other: float):
        return self.__class__(
            other*self.A,
            other*self.R)

    def mul_and_add(self, other, factor):
        return self.__class__(
            self.A + factor*other.R,
            self.R + factor*other.R)


def rlog_descent(flg, X, X1, tol=1e-10):
    """
    Use this only if 
    flg.alpha[:, 1] = flg.alpha[0, 1]
    flg.alpha[:, 0] = flg.alpha[0, 0]
    This is Stiefel geodesics
    Only use init_type 0 now = may change in the future
    ret_aligned: return the aligment
    """
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 0] - flg.alpha[:, 0]))))
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 1] - flg.alpha[:, 1]))))

    alf = flg.alpha[0, 1]/flg.alpha[0, 0]        
    d = flg.dvec[1:].sum()
    n = flg.dvec.sum()
    sqrt2 = np.sqrt(2)

    def getQ():
        """ algorithm: find a basis in linear span of Y Y1
        orthogonal to Y
        """
        u, s, v = np.linalg.svd(
            np.concatenate([X, X1], axis=1), full_matrices=False)
        k = (s > 1e-14).sum()
        good = u[:, :k]@v[:k, :k]
        qs = null_space(X.T@good)
        Q, _ = np.linalg.qr(good@qs)
        return Q

    # Q, s, _ = la.svd(Y1 - Y@Y.T@Y1, full_matrices=False)
    # Q = Q[:, :np.sum(np.abs(s) > 1e-14)]
    Q = getQ()
    k = Q.shape[1]
    p = flg.p
    lbd = flg.lbd

    def asym(mat):
        return 0.5*(mat - mat.T)

    XQ = np.array(np.bmat([X, Q]))
    # X2 = XQ.T@X1@X1.T@XQ
    X2 = XQ.T@(X1*lbd[None, :])@X1.T@XQ

    def sexp(A, R):
        ex1 = expm((1-2*alf)*A)
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        return X@ex2[:d, :d]@ex1 + Q@ex2[d:, :d]@ex1    
    
    def dist(v):
        #  = (dist0a(v) - d)*2
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        A, R = v.A, v.R
        x_mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)
        Mid = (ex*lbd[None, :])@ex.T
        return (- np.trace(X2@exh[:, :d]@Mid@exh[:, :d].T))

    def jac(A, R):
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        gdc = flg._g_idx
        # A, R = v.A, v.R
        x_mat = np.array(
            np.bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)

        blk = np.zeros_like(exh)
        blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
        blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@X2@exh[:, :d]

        fexh = 2*expm_frechet(x_mat, blk@X2)[1]
        fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]

        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]            
            fexh[br:br, br:br] = 0
            fex[br:br, br:br] = 0

        return (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),\
            fexh[d:, :d] - fexh[:d, d:].T

    def conv_to_tan(A, R):
        return X@A + Q@R

    from scipy.optimize import minimize
    adim = (flg.dvec[1:].sum()*flg.dvec[1:].sum() -
            (flg.dvec[1:]*flg.dvec[1:]).sum()) // 2
    tdim = d*k + adim

    vv = AR.zerovec(d, k)
    A, R = vv.A, vv.R
    max_cnt = 100
    done = False
    cnt = 0
    scl = np.sqrt(n*d)
    fl2 = flg.lbd*flg.lbd
    lbd2 = np.sum(fl2)
    gdc = flg._g_idx
    while not done and cnt < max_cnt:
        # omg = sexp(A, R) - Y1
        # omg_norm = np.linalg.norm(omg, np.inf)
        d1 = max(0, lbd2 + dist(AR(A, R)))        
        omg_norm = np.sqrt(d1)
        print(omg_norm)
        if omg_norm < tol*scl:
            done = True
            break
        else:
            # dA, dR = JacT(A, R, omg)
            dA, dR = jac(A, R)
            for r in range(1, p+1):
                if r not in gdc:
                    continue
                br, er = gdc[r]
                for s in range(1, r):
                    if s not in gdc:
                        continue
                    bs, es = gdc[s]
                    rt = (flg.lbd[br] - flg.lbd[bs])*(flg.lbd[br] - flg.lbd[bs])
                    dA[br:er, bs:es] = dA[br:er, bs:es]/rt
                    dA[bs:es, br:er] = dA[bs:es, br:er]/rt

            dR = (dR/fl2[None, :])/2
            
            # print(dA, dA1, dA - dA1)
            # print(dR, dR1, dR - dR1)
            dnorm = np.sqrt(np.sum(dA*dA) + np.sum(dR*dR))
            if dnorm == 0:
                break
            # A -= omg_norm/dnorm*dA
            # R -= omg_norm/dnorm*dR

            A -= dA
            R -= dR

            cnt  += 1
    return X@A + Q@R, cnt, done, A, R, Q


def log_descent(flg, Y, Y1, tol=1e-10, max_itr=100):
    """
    Use this only if 
    flg.alpha[:, 1] = flg.alpha[0, 1]
    flg.alpha[:, 0] = flg.alpha[0, 0]
    This is Stiefel geodesics
    Only use init_type 0 now = may change in the future
    ret_aligned: return the aligment
    """
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 0] - flg.alpha[:, 0]))))
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 1] - flg.alpha[:, 1]))))

    alf = flg.alpha[0, 1]/flg.alpha[0, 0]        
    d = flg.dvec[1:].sum()
    n = flg.dvec.sum()
    sqrt2 = np.sqrt(2)

    def getQ():
        """ algorithm: find a basis in linear span of Y Y1
        orthogonal to Y
        """
        u, s, v = np.linalg.svd(
            np.concatenate([Y, Y1], axis=1), full_matrices=False)
        k = (s > 1e-14).sum()
        good = u[:, :k]@v[:k, :k]
        qs = null_space(Y.T@good)
        Q, _ = np.linalg.qr(good@qs)
        return Q

    # Q, s, _ = la.svd(Y1 - Y@Y.T@Y1, full_matrices=False)
    # Q = Q[:, :np.sum(np.abs(s) > 1e-14)]
    Q = getQ()
    k = Q.shape[1]
    p = flg.p
    lbd = flg.lbd

    def asym(mat):
        return 0.5*(mat - mat.T)

    YQ = np.array(np.bmat([Y, Q]))
    # X2 = XQ.T@X1@X1.T@XQ
    Y2 = YQ.T@(Y1*lbd[None, :])@Y1.T@YQ

    def sexp(A, R):
        ex1 = expm((1-2*alf)*A)
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        return Y@ex2[:d, :d]@ex1 + Q@ex2[d:, :d]@ex1    

    def dist(v):
        #  = (dist0a(v) - d)*2
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        A, R = v.A, v.R
        x_mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)
        Mid = (ex*lbd[None, :])@ex.T
        return - np.sum((Y2@exh[:, :d]@Mid)*exh[:, :d])

    def jac(A, R):
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        gdc = flg._g_idx
        # A, R = v.A, v.R
        x_mat = np.array(
            np.bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)

        blk = np.zeros_like(exh)
        blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
        blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@Y2@exh[:, :d]

        fexh = 2*expm_frechet(x_mat, blk@Y2)[1]
        fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]

        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]            
            fexh[br:br, br:br] = 0
            fex[br:br, br:br] = 0

        return (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),\
            fexh[d:, :d] - fexh[:d, d:].T

    def fun(A, R):
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        gdc = flg._g_idx
        # A, R = v.A, v.R
        x_mat = np.array(
            np.bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)

        blk = np.zeros_like(exh)
        blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
        blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@Y2@exh[:, :d]

        fexh = 2*expm_frechet(x_mat, blk@Y2)[1]
        fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]
        
        Mid = (ex*lbd[None, :])@ex.T

        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]            
            fexh[br:br, br:br] = 0
            fex[br:br, br:br] = 0
        return - np.sum((Y2@exh[:, :d]@Mid)*(exh[:, :d])),\
            (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),\
            fexh[d:, :d] - fexh[:d, d:].T

    def conv_to_tan(A, R):
        return Y@A + Q@R

    adim = (flg.dvec[1:].sum()*flg.dvec[1:].sum() -
            (flg.dvec[1:]*flg.dvec[1:]).sum()) // 2
    tdim = d*k + adim

    vv = AR.zerovec(d, k)
    A, R = vv.A, vv.R
    done = False
    itr = 0
    fjacs = 0
    fvals = 0    
    scl = np.sqrt(n*d)
    fl2 = flg.lbd*flg.lbd
    lbd2 = np.sum(fl2)
    gdc = flg._g_idx

    while not done and itr < max_itr:
        # omg = sexp(A, R) - Y1
        # omg_norm = np.linalg.norm(omg, np.inf)
        fval, dA, dR = fun(A, R)
        fjacs += 1
        itr  += 1
        d1 = max(0, lbd2 + fval)
        omg_norm = np.sqrt(d1)
        # print(omg_norm)
        if omg_norm < tol*scl:
            done = True
            break
        else:
            for r in range(1, p+1):
                if r not in gdc:
                    continue
                br, er = gdc[r]
                for s in range(1, r):
                    if s not in gdc:
                        continue
                    bs, es = gdc[s]
                    rt = (flg.lbd[br] - flg.lbd[bs])*(flg.lbd[br] - flg.lbd[bs])
                    dA[br:er, bs:es] = dA[br:er, bs:es]/rt
                    dA[bs:es, br:er] = dA[bs:es, br:er]/rt

            dR = (dR/fl2[None, :])/2

            dnorm = np.sqrt(np.sum(dA*dA) + np.sum(dR*dR))
            if dnorm == 0:
                break
            A -= dA
            R -= dR

    return Y@A + Q@R, itr, fvals, fjacs, done
    

def log_steep_descent(flg, Y, Y1, tol=1e-10, max_itr=100):
    """
    Use this only if 
    flg.alpha[:, 1] = flg.alpha[0, 1]
    flg.alpha[:, 0] = flg.alpha[0, 0]
    This is Stiefel geodesics
    Only use init_type 0 now = may change in the future
    ret_aligned: return the aligment
    """
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 0] - flg.alpha[:, 0]))))
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 1] - flg.alpha[:, 1]))))

    alf = flg.alpha[0, 1]/flg.alpha[0, 0]        
    d = flg.dvec[1:].sum()
    n = flg.dvec.sum()
    # sqrt2 = np.sqrt(2)

    def getQ():
        """ algorithm: find a basis in linear span of Y Y1
        orthogonal to Y
        """
        u, s, v = np.linalg.svd(
            np.concatenate([Y, Y1], axis=1), full_matrices=False)
        k = (s > 1e-14).sum()
        good = u[:, :k]@v[:k, :k]
        qs = null_space(Y.T@good)
        Q, _ = np.linalg.qr(good@qs)
        return Q

    # Q, s, _ = la.svd(Y1 - Y@Y.T@Y1, full_matrices=False)
    # Q = Q[:, :np.sum(np.abs(s) > 1e-14)]
    Q = getQ()
    k = Q.shape[1]
    p = flg.p
    lbd = flg.lbd

    YQ = np.array(np.bmat([Y, Q]))
    # X2 = XQ.T@X1@X1.T@XQ
    Y2 = YQ.T@(Y1*lbd[None, :])@Y1.T@YQ

    def sexp(A, R):
        ex1 = expm((1-2*alf)*A)
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        return Y@ex2[:d, :d]@ex1 + Q@ex2[d:, :d]@ex1    

    def dist(A, R):
        #  = (dist0a(v) - d)*2
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        x_mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)
        Mid = (ex*lbd[None, :])@ex.T
        return - np.sum((Y2@exh[:, :d]@Mid)*exh[:, :d])

    def jac(A, R):
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        gdc = flg._g_idx
        # A, R = v.A, v.R
        x_mat = np.array(
            np.bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)

        blk = np.zeros_like(exh)
        blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
        blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@Y2@exh[:, :d]

        fexh = 2*expm_frechet(x_mat, blk@Y2)[1]
        fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]

        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]            
            fexh[br:br, br:br] = 0
            fex[br:br, br:br] = 0

        return (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),\
            fexh[d:, :d] - fexh[:d, d:].T

    def fun(A, R):
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        gdc = flg._g_idx
        # A, R = v.A, v.R
        x_mat = np.array(
            np.bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)

        blk = np.zeros_like(exh)
        blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
        blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@Y2@exh[:, :d]

        fexh = 2*expm_frechet(x_mat, blk@Y2)[1]
        fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]
        
        Mid = (ex*lbd[None, :])@ex.T

        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]            
            fexh[br:br, br:br] = 0
            fex[br:br, br:br] = 0
        return - np.sum((Y2@exh[:, :d]@Mid)*(exh[:, :d])),\
            (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),\
            fexh[d:, :d] - fexh[:d, d:].T

    def conv_to_tan(A, R):
        return Y@A + Q@R

    adim = (flg.dvec[1:].sum()*flg.dvec[1:].sum() -
            (flg.dvec[1:]*flg.dvec[1:]).sum()) // 2
    tdim = d*k + adim

    # vv = AR.zerovec(d, k)
    A = np.zeros((d, d))
    R = np.zeros((k, d))
    max_cj = 7
    done = False
    itr = 0
    scl = np.sqrt(n*d)
    fl2 = flg.lbd*flg.lbd
    lbd2 = np.sum(fl2)
    gdc = flg._g_idx

    ft = 1
    bt = .8
    fvals = 0
    fjacs = 0    
    while itr < max_itr:
        fval, dA, dR = fun(A, R)
        fjacs += 1
        itr += 1
        
        dst = max(0, lbd2 + fval)
        omg_norm = np.sqrt(dst)
        # print(omg_norm)
        if omg_norm < tol*scl:
            done = True
            break
        else:
            for r in range(1, p+1):
                if r not in gdc:
                    continue
                br, er = gdc[r]
                for s in range(1, r):
                    if s not in gdc:
                        continue
                    bs, es = gdc[s]
                    rt = (flg.lbd[br] - flg.lbd[bs])*(flg.lbd[br] - flg.lbd[bs])
                    dA[br:er, bs:es] = dA[br:er, bs:es]/rt
                    dA[bs:es, br:er] = dA[bs:es, br:er]/rt

            dR = (dR/fl2[None, :])/2
        cf = ft
        found = False
        cj = 0
        while not found and cj < max_cj:
            Anew = A - cf*dA
            Rnew = R - cf*dR
            dstnew = max(dist(Anew, Rnew) + lbd2, 0)
            fvals += 1
            if dstnew < dst:
                found = True
                A = Anew
                R = Rnew
                dst = dstnew
                # print("cf=%f dst=%f" % (cf, dst))
            else:
                cf *= bt
            cj += 1
        if not found:
            A = Anew
            R = Rnew
            dst = dstnew
        if np.sqrt(dst) < tol*np.sqrt(n*d):
            done = True
            break
    # print(dst)
    # print(cnt)
    return conv_to_tan(A, R), itr, fvals, fjacs, done


def log_lbfgs(flg, Y, Y1, tol=1e-10, ncor=5, max_itr=200):
    """
    Use this only if 
    flg.alpha[:, 1] = flg.alpha[0, 1]
    flg.alpha[:, 0] = flg.alpha[0, 0]
    This is Stiefel geodesics
    Only use init_type 0 now = may change in the future
    ret_aligned: return the aligment
    """
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 0] - flg.alpha[:, 0]))))
    assert(
        1e-14 > np.max(np.abs(np.abs(flg.alpha[:, 1] - flg.alpha[:, 1]))))

    alf = flg.alpha[0, 1]/flg.alpha[0, 0]        
    d = flg.dvec[1:].sum()
    n = flg.dvec.sum()
    sqrt2 = np.sqrt(2)

    def getQ():
        """ algorithm: find a basis in linear span of Y Y1
        orthogonal to Y
        """
        u, s, v = np.linalg.svd(
            np.concatenate([Y, Y1], axis=1), full_matrices=False)
        k = (s > 1e-14).sum()
        good = u[:, :k]@v[:k, :k]
        qs = null_space(Y.T@good)
        Q, _ = np.linalg.qr(good@qs)
        return Q

    # Q, s, _ = la.svd(Y1 - Y@Y.T@Y1, full_matrices=False)
    # Q = Q[:, :np.sum(np.abs(s) > 1e-14)]
    Q = getQ()
    k = Q.shape[1]
    p = flg.p
    lbd = flg.lbd

    def asym(mat):
        return 0.5*(mat - mat.T)

    YQ = np.array(np.bmat([Y, Q]))
    # X2 = XQ.T@X1@X1.T@XQ
    Y2 = YQ.T@(Y1*lbd[None, :])@Y1.T@YQ

    def vec(A, R):
        # for A, take all blocks [ij with i > j]
        lret = []
        for r in range(1, p+1):
            gdc = flg._g_idx
            if r not in gdc:
                continue
            br, er = gdc[r]
            for s in range(r+1, p+1):
                if s <= r:
                    continue
                bs, es = gdc[s]
                lret.append(A[br:er, bs:es].reshape(-1)*sqrt2)

        lret.append(R.reshape(-1))
        return np.concatenate(lret)

    def unvec(avec):
        A = np.zeros((d, d))
        R = np.zeros((k, d))
        gdc = flg._g_idx
        be = 0
        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]
            for s in range(r+1, p+1):
                if s <= r:
                    continue
                bs, es = gdc[s]
                dr = er - br
                ds = es - bs
                A[br:er, bs:es] = (avec[be: be+dr*ds]/sqrt2).reshape(dr, ds)
                A[bs:es, br:er] = - A[br:er, bs:es].T
                be += dr*ds
        R = avec[be:].reshape(k, d)
        return A, R
    
    def sexp(A, R):
        ex1 = expm((1-2*alf)*A)
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        return Y@ex2[:d, :d]@ex1 + Q@ex2[d:, :d]@ex1    
    
    def dist(v):
        #  = (dist0a(v) - d)*2
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        A, R = v.A, v.R
        x_mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)
        Mid = (ex*lbd[None, :])@ex.T
        return - np.sum((Y2@exh[:, :d]@Mid)*(exh[:, :d]))

    def jac(A, R):
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        gdc = flg._g_idx
        # A, R = v.A, v.R
        x_mat = np.array(
            np.bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)

        blk = np.zeros_like(exh)
        blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
        blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@Y2@exh[:, :d]

        fexh = 2*expm_frechet(x_mat, blk@Y2)[1]
        fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]

        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]            
            fexh[br:br, br:br] = 0
            fex[br:br, br:br] = 0

        return (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),\
            fexh[d:, :d] - fexh[:d, d:].T

    def fun(v):
        A, R = unvec(v)
        alf = flg.alpha[0, 1] / flg.alpha[0, 0]
        gdc = flg._g_idx
        # A, R = v.A, v.R
        x_mat = np.array(
            np.bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        exh = expm(x_mat)
        ex = expm((1-2*alf)*A)

        blk = np.zeros_like(exh)
        blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
        blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@Y2@exh[:, :d]

        fexh = 2*expm_frechet(x_mat, blk@Y2)[1]
        fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]
        
        Mid = (ex*lbd[None, :])@ex.T

        for r in range(1, p+1):
            if r not in gdc:
                continue
            br, er = gdc[r]            
            fexh[br:br, br:br] = 0
            fex[br:br, br:br] = 0
        return - np.sum((Y2@exh[:, :d]@Mid)*(exh[:, :d])),\
            vec((1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),
                fexh[d:, :d] - fexh[:d, d:].T)
    
    def conv_to_tan(A, R):
        return Y@A + Q@R

    def make_vec(xi):
        return vec(Y.T@xi, Q.T@xi)

    adim = (flg.dvec[1:].sum()*flg.dvec[1:].sum() -
            (flg.dvec[1:]*flg.dvec[1:]).sum()) // 2
    tdim = d*k + adim

    x0 = np.zeros(tdim)

    fl2 = flg.lbd*flg.lbd
    lbd2 = np.sum(fl2)
    
    def pcv():
        lret = []
        for r in range(1, p+1):
            gdc = flg._g_idx
            if r not in gdc:
                continue
            br, er = gdc[r]
            for s in range(r+1, p+1):
                if s <= r:
                    continue
                bs, es = gdc[s]
                rt = (flg.lbd[br] - flg.lbd[bs])*(flg.lbd[br] - flg.lbd[bs])

                lret.append(
                    (np.ones((er-br, es-bs))/rt).reshape(-1))

        lret.append(
            ((np.ones((k, d))/fl2[None, :])/2).reshape(-1))
        return np.concatenate(lret)

    pcV = pcv()

    def pcond(x):
        return pcV

    maxFunEvals = max_itr
    maxIter = max_itr
    if tol is not None:
        optTol = tol
        progTol = tol*1e-3
    else:
        optTol = 1e-12
        progTol = 1e-15
    corrections = ncor  # historical values stored
    c1 = 1e-4
    c2 = 0.9
    max_ls = 25
    Pcond = pcond

    options = (maxFunEvals, maxIter, optTol, progTol,
               corrections, c1, c2, max_ls, Pcond)
    x, f, exitflag, output = mlf.minimize(fun, x0, options)
        
    A1, R1 = unvec(x)
    if flg.log_stats:
        return conv_to_tan(A1, R1), output
    else:
        return conv_to_tan(A1, R1)
