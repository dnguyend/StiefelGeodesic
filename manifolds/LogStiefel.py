import numpy as np
from ManNullRange.manifolds.RealStiefel import RealStiefel
from ManNullRange.manifolds.tools import asym, vecah, unvecah
from scipy.linalg import expm, null_space, expm_frechet, logm
from StiefelGeodesic.manifolds.tools import sbmat, linf
import StiefelGeodesic.manifolds.minimize_lfbgs as mlf

from StiefelGeodesic.manifolds.TrustNCG import _minimize_trust_ncg


class AR(object):
    """ Pair of A and R matrix. Used for customized NCG
    """
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
    def rand(cls, d, k):
        A = np.random.randn(d, d)
        R = np.random.randn(k, d)
        A = A - A.T
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
    

def log_custom_trust_ncg(stf, Y, Y1, show_steps=False, init_type=1):
    """Inverse of exp

    Parameters
    ----------
    Y    : a manifold point
    Y1  : tangent vector

    Returns
    ----------
    eta such that stf.exp(X, eta) = Y1

    Algorithm: use the scipy.optimize trust region method
    to minimize in eta ||stf.exp(Y, eta) - Y1||_F^2
    _F is the Frobenius norm in R^{n\times d}
    The jacobian could be computed by the expm_frechet function
    """
    alf = stf.alpha[1]/stf.alpha[0]
    d = stf.d
    adim = (d*(d-1))//2

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
    if k == 0:
        # Y1 and Y has the same linear span
        A = logm(Y.T @ Y1)

        if stf.log_stats:
            return Y@A, [('success', True), ('message', 'aligment')]
        return Y@A

    def dist(v):
        A, R = v.A, v.R
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        M = ex2[:d, :d]
        N = ex2[d:, :d]

        return -np.trace(Y1.T@(Y@M+Q@N)@expm((1-2*alf)*A))

    def jac(v):
        A, R = v.A, v.R
        ex1 = expm((1-2*alf)*A)

        mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        E = sbmat(
            [[ex1@Y1.T@Y, ex1@Y1.T@Q],
             [np.zeros_like(R), np.zeros((k, k))]])

        ex2, fe2 = expm_frechet(mat, E)
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        YMQN = (Y@M+Q@N)

        partA = asym(
            (1-2*alf)*expm_frechet((1-2*alf)*A, Y1.T@YMQN)[1])

        partA += 2*alf*asym(fe2[:d, :d])
        partR = -(fe2[:d, d:].T - fe2[d:, :d])

        return AR(partA, partR)

    def hessp(v, xi):
        dlt = 1e-9
        return (jac(v+dlt*xi) - jac(v)).scalar_mul(1/dlt)

    def conv_to_tan(A, R):
        return Y@A + Q@R

    eta0 = stf.proj(Y, Y1-Y)
    A0 = asym(Y.T@eta0)
    R0 = Q.T@eta0 - (Q.T@Y)@(Y.T@eta0)

    if init_type != 0:
        x0 = AR(A0, R0)
    else:
        x0 = AR.zerovec(d, k)

    def printxk(xk):
        print(jac(xk).norm, dist(xk))

    if show_steps:
        callback = printxk
    else:
        callback = None

    res = {'fun': np.nan, 'x': np.zeros_like(x0),
           'success': False,
           'message': 'minimizer exception'}
    if stf.log_gtol is None:
        res = _minimize_trust_ncg(
            dist, x0, jac=jac, hessp=hessp, callback=callback)
        # res = _minimize_trust_ncg(
        #    dist, x0, jac=jac, callback=callback)            
    else:
        if stf.log_method.startswith('trust'):            
            res = _minimize_trust_ncg(
                dist, x0, jac=jac, hessp=hessp,
                callback=callback,
                gtol=stf.log_gtol)

    stat = [(a, res[a]) for a in res.keys() if a not in ['x', 'jac']]
    A1, R1 = res['x'].A, res['x'].R
    if stf.log_stats:
        return conv_to_tan(A1, R1), stat
    else:
        return conv_to_tan(A1, R1)            


def log_descent(stf, Y, Y1, tol=1e-10, max_itr=100):
    alf = stf.alpha[1]/stf.alpha[0]
    n, d = stf.n, stf.d

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

    Q = getQ()
    k = Q.shape[1]
    
    ZTY = Y1.T@Y    
    ZTQ = Y1.T@Q
    
    def sexp(A, R):
        ex1 = expm((1-2*alf)*A)
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        return Y@ex2[:d, :d]@ex1 + Q@ex2[d:, :d]@ex1    
    
    def jac(A, R):
        ex1 = expm((1-2*alf)*A)

        mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        E = sbmat(
            [[ex1@ZTY, ex1@ZTQ],
             [np.zeros_like(R), np.zeros((k, k))]])

        ex2, fe2 = expm_frechet(mat, E)
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        ZYMQN = ZTY@M+ZTQ@N

        partA = asym(
            (1-2*alf)*expm_frechet((1-2*alf)*A, ZYMQN)[1])

        partA += 2*alf*asym(fe2[:d, :d])
        partR = -(fe2[:d, d:].T - fe2[d:, :d])
        return partA, partR            

    def fun(A, R):
        ex1 = expm((1-2*alf)*A)

        mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        E = sbmat(
            [[ex1@ZTY, ex1@ZTQ],
             [np.zeros_like(R), np.zeros((k, k))]])

        ex2, fe2 = expm_frechet(mat, E)
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        ZYMQN = ZTY@M+ZTQ@N

        partA = asym(
            (1-2*alf)*expm_frechet((1-2*alf)*A, ZYMQN)[1])

        partA += 2*alf*asym(fe2[:d, :d])
        partR = -(fe2[:d, d:].T - fe2[d:, :d])

        # return (Y@M+Q@N)@expm((1-2*alf)*A), partA, partR
        return -np.sum(ZYMQN.T * expm((1-2*alf)*A)), partA, partR        
    
    eta0 = stf.proj(Y, Y1-Y)
    A = asym(Y.T@eta0)
    R = Q.T@eta0 - (Q.T@Y)@(Y.T@eta0)

    # max_itr = 1000
    done = False
    itr = 0
    fjacs = 0
    fvals = 0
    scl = np.sqrt(n*d)

    while not done and itr < max_itr:
        f, dA, dR = fun(A, R)
        fjacs += 1
        itr  += 1            
        # omg = Yt - Y1
        # omg_norm = np.linalg.norm(omg, np.inf)
        if np.sqrt(max(f + d, 0)) < tol/scl:
            done = True
            break
        else:
            dnorm = np.sqrt(np.sum(dA*dA) + np.sum(dR*dR))
            if dnorm == 0:
                break
            A -= dA
            R -= dR
    return Y@A + Q@R, itr, fvals, fjacs, done


def log_shooting_AR(stf, Y, Y1, tol=1e-10, TT=20):
    alf = stf.alpha[1]/stf.alpha[0]
    n, d = stf.n, stf.d
    eta = stf.proj(Y, Y1-Y)

    max_itr = 100
    done = False
    itr = 0

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

    Q = getQ()
    k = Q.shape[1]

    def sexp(A, R):
        ex1 = expm((1-2*alf)*A)
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        return Y@ex2[:d, :d]@ex1 + Q@ex2[d:, :d]@ex1    
    
    while not done and itr < max_itr:
        A = asym(Y.T@eta)
        R = Q.T@eta - (Q.T@Y)@(Y.T@eta)        
        omg = sexp(A, R) - Y1
        
        omg_norm = np.linalg.norm(omg)
        # print(omg_norm)
        if omg_norm < tol:
            done = True
            break
        else:
            # dnorm = np.sqrt(np.sum(dA*dA) + np.sum(dR*dR))
            # if dnorm == 0:
            #    break
            for tt in range(TT, -1, -1):
                Yt = sexp(tt*A, tt*R)
                omg = stf.proj(Yt, omg)
                # print(tt)
            eta -= omg
            itr  += 1
    return eta


def log_shooting_raw(stf, Y, Y1, tol=1e-10, TT=20):
    alf = stf.alpha[1]/stf.alpha[0]
    n, d = stf.n, stf.d
    eta = stf.proj(Y, Y1-Y)

    max_cnt = 100
    done = False
    cnt = 0

    while not done and cnt < max_cnt:
        omg = stf.exp_alt(Y, eta) - Y1        
        omg_norm = np.linalg.norm(omg)
        # print(omg_norm)
        if omg_norm < tol:
            done = True
            break
        else:
            # dnorm = np.sqrt(np.sum(dA*dA) + np.sum(dR*dR))
            # if dnorm == 0:
            #    break
            for tt in range(TT, -1, -1):
                Yt = stf.exp_alt(Y, tt*eta)
                omg = stf.proj(Yt, omg)
                # print(tt)
            eta -= omg
            cnt  += 1
    return eta
    

def log_steep_descent(stf, Y, Y1, tol=1e-10, max_itr=100):
    """Inverse of exp

    Parameters
    ----------
    Y    : a manifold point
    Y1  : tangent vector

    Returns
    ----------
    eta such that stf.exp(X, eta) = Y1

    Algorithm: use the scipy.optimize trust region method
    to minimize in eta ||stf.exp(Y, eta) - Y1||_F^2
    _F is the Frobenius norm in R^{n\times d}
    The jacobian could be computed by the expm_frechet function
    """
    alf = stf.alpha[1]/stf.alpha[0]
    d = stf.d
    n = stf.n
    # adim = (d*(d-1))//2

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
    ZTY = Y1.T@Y    
    ZTQ = Y1.T@Q
    
    if k == 0:
        # Y1 and Y has the same linear span
        A = logm(ZTY.T)

        if stf.log_stats:
            return Y@A, [('success', True), ('message', 'aligment')]
        return Y@A

    def dist(A, R):
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        M = ex2[:d, :d]
        N = ex2[d:, :d]

        # return -np.trace(Y1.T@(Y@M+Q@N)@expm((1-2*alf)*A))
        ZYMQN = ZTY@M+ZTQ@N
        # return -np.sum(Y1*((Y@M+Q@N)@expm((1-2*alf)*A)))
        return -np.sum(ZYMQN.T*expm((1-2*alf)*A))        

    def jac(A, R):
        ex1 = expm((1-2*alf)*A)

        mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        E = sbmat(
            [[ex1@ZTY, ex1@ZTQ],
             [np.zeros_like(R), np.zeros((k, k))]])

        ex2, fe2 = expm_frechet(mat, E)
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        ZYMQN = ZTY@M+ZTQ@N

        partA = asym(
            (1-2*alf)*expm_frechet((1-2*alf)*A, ZYMQN)[1])

        partA += 2*alf*asym(fe2[:d, :d])
        partR = -(fe2[:d, d:].T - fe2[d:, :d])

        return AR(partA, partR)

    def fun(A, R):
        ex1 = expm((1-2*alf)*A)

        mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        E = sbmat(
            [[ex1@ZTY, ex1@ZTQ],
             [np.zeros_like(R), np.zeros((k, k))]])

        ex2, fe2 = expm_frechet(mat, E)
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        ZYMQN = ZTY@M+ZTQ@N

        partA = asym(
            (1-2*alf)*expm_frechet((1-2*alf)*A, ZYMQN)[1])

        partA += 2*alf*asym(fe2[:d, :d])
        partR = -(fe2[:d, d:].T - fe2[d:, :d])

        # return d - np.sum(Y1*((Y@M+Q@N)@expm((1-2*alf)*A))), partA, partR
        return d - np.sum(ZYMQN.T * expm((1-2*alf)*A)), partA, partR        

    def conv_to_tan(A, R):
        return Y@A + Q@R

    eta0 = stf.proj(Y, Y1-Y)
    A = asym(Y.T@eta0)
    R = Q.T@eta0 - (Q.T@Y)@(Y.T@eta0)

    # max_itr = 1000
    max_cj = 7
    done = False
    itr = 0
    scl = np.sqrt(n*d)
    ft = 1
    bt = .8
    fvals = 0
    fjacs = 0
    while itr < max_itr:
        fval, dA, dR = fun(A, R)
        fjacs += 1
        itr += 1
        
        cf = ft
        found = False
        dst = max(0, fval)
        fnorm = np.sqrt(dst)
        if fnorm < tol*scl:
            done = True
            break
        
        cj = 0
        while not found and cj < max_cj:
            Anew = A - cf*dA
            Rnew = R - cf*dR
            dstnew = max(0, dist(Anew, Rnew) + d)
            fvals += 1
            
            if dstnew < dst:
                found = True
                A = Anew
                R = Rnew
                dst = dstnew
            else:
                cf *= bt
            cj += 1
        if not found:
            A = Anew
            R = Rnew
            dst = dstnew
        # if np.sqrt(dst) < tol*scl:
        if np.sqrt(dst) < tol:
            done = True
            break
    # print(dst)
    # print(cnt)
    return conv_to_tan(A, R), itr, fvals, fjacs, done


def log_lbfgs(stf, Y, Y1, show_steps=False, init_type=1,
              tol=None, ncor=5):
    """Inverse of exp

    Parameters
    ----------
    Y    : a manifold point
    Y1  : tangent vector

    Returns
    ----------
    eta such that stf.exp(X, eta) = Y1

    Algorithm: use the scipy.optimize trust region method
    to minimize in eta ||stf.exp(Y, eta) - Y1||_F^2
    _F is the Frobenius norm in R^{n\times d}
    The jacobian could be computed by the expm_frechet function
    """
    alf = stf.alpha[1]/stf.alpha[0]
    d = stf.d
    adim = (d*(d-1))//2

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

    ZTY = Y1.T@Y    
    ZTQ = Y1.T@Q
    
    if k == 0:
        # Y1 and Y has the same linear span
        A = logm(ZTY.T)

        if stf.log_stats:
            return Y@A, [('success', True), ('message', 'aligment')]
        return Y@A    

    def vec(A, R):
        return np.concatenate(
            [vecah(A), R.reshape(-1)])

    def unvec(avec):
        return unvecah(avec[:adim]), avec[adim:].reshape(k, d)

    def dist(v):
        A, R = unvec(v)
        ex2 = expm(
            sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        ZYMQN = ZTY@M+ZTQ@N
        return -np.sum(ZYMQN.T*expm((1-2*alf)*A))

    def jac(v):
        A, R = unvec(v)
        ex1 = expm((1-2*alf)*A)

        mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        E = sbmat(
            [[ex1@ZTY, ex1@ZTQ],
             [np.zeros_like(R), np.zeros((k, k))]])

        ex2, fe2 = expm_frechet(mat, E)
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        ZYMQN = ZTY@M+ZTQ@N
        
        # YMQN = (Y@M+Q@N)

        partA = asym(
            (1-2*alf)*expm_frechet((1-2*alf)*A, ZYMQN)[1])

        partA += 2*alf*asym(fe2[:d, :d])
        partR = -(fe2[:d, d:].T - fe2[d:, :d])

        return vec(partA, partR)

    def fun_jac(v):
        A, R = unvec(v)
        ex1 = expm((1-2*alf)*A)

        mat = sbmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])
        E = sbmat(
            [[ex1@Y1.T@Y, ex1@ZTQ],
             [np.zeros_like(R), np.zeros((k, k))]])

        ex2, fe2 = expm_frechet(mat, E)
        M = ex2[:d, :d]
        N = ex2[d:, :d]
        
        ZYMQN = ZTY@M+ZTQ@N
        # YMQN = (Y@M+Q@N)

        partA = asym(
            (1-2*alf)*expm_frechet((1-2*alf)*A, ZYMQN)[1])

        partA += 2*alf*asym(fe2[:d, :d])
        partR = -(fe2[:d, d:].T - fe2[d:, :d])

        return -np.sum(ZYMQN.T * expm((1-2*alf)*A)),\
            vec(partA, partR)
    
    def conv_to_tan(A, R):
        return Y@A + Q@R

    eta0 = stf.proj(Y, Y1-Y)
    A0 = asym(Y.T@eta0)
    R0 = Q.T@eta0 - (Q.T@Y)@(Y.T@eta0)

    if init_type != 0:
        x0 = vec(A0, R0)
    else:
        x0 = np.zeros(adim + stf.d*k)

    res = {'fun': np.nan, 'x': np.zeros_like(x0),
           'success': False,
           'message': 'minimizer exception'}

    if True:
        maxFunEvals = 100
        maxIter = 100
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
        
        def pcond(x):
            return np.ones(adim + stf.d*k)

        options = (maxFunEvals, maxIter, optTol, progTol,
                   corrections, c1, c2, max_ls, pcond)
        x, f, exitflag, output = mlf.minimize(fun_jac, x0, options)
            
    A1, R1 = unvec(x)    
    if stf.log_stats:
        return conv_to_tan(A1, R1), output
    else:
        return conv_to_tan(A1, R1)


