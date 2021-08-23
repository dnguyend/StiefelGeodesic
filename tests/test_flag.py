import numpy as np
from scipy.linalg import null_space, expm, logm, expm_frechet

from ManNullRange.manifolds.tools import sym, asym, vecah, unvecah
from ManNullRange.manifolds.RealFlag import RealFlag
from StiefelGeodesic.manifolds.tools import linf, sbmat
from StiefelGeodesic.manifolds.LogFlag import (
    log_lbfgs, log_descent, log_steep_descent, AR)
    

def Edist(flg, X, Y):
    """ Euclidean distance. Useful to compare two
    elememt
    """
    YTX = Y.T@X
    d2 = 2*(np.sum(flg.lbd * flg.lbd) - np.trace(
        (YTX*flg.lbd[None, :])@(YTX.T*flg.lbd[None, :])))
    return np.sqrt(max(d2, 0))    
    

def compare_flag():
    from time import perf_counter

    def doit(alf, dveci, npi, Nsamples, tol, max_itr=100):
        np.random.seed(0)
        dvec = np.array(dveci)
        p = dvec.shape[0] - 1
        alpha = np.full((p, p+1), alf)
        alpha[:, 0] = 1.

        flg = RealFlag(dvec, alpha, log_stats=True, log_method='trust-krylov')
        # flbf = RealFlag(dvec, alpha, log_stats=True, log_method='l-bfgs-b')
        
        ret = {}
        ITRS = {'DESCENT': np.empty(Nsamples),
                'STEEP': np.empty(Nsamples),                
                'CUSTOM_LBFGS': np.empty(Nsamples)}
        
        FVALS = {'DESCENT': np.empty(Nsamples),
                 'STEEP': np.empty(Nsamples),
                 'CUSTOM_LBFGS': np.empty(Nsamples)}
        DONES = {'DESCENT': np.empty(Nsamples, dtype=int),
                 'STEEP': np.empty(Nsamples, dtype=int),
                 'CUSTOM_LBFGS': np.empty(Nsamples, dtype=int)}
        
        for i in range(Nsamples):
            print('Doing %d' % i)
            Y = flg.rand()
            xi = flg.randvec(Y)*np.pi*npi
            Y1 = flg.exp(Y, xi)        
            t0 = perf_counter()
            xi1, st1 = flg.log(Y, Y1)
            t1 = perf_counter()
            
            ret[i] = {'SCIPY_NCG': {
                'time': t1 - t0,
                'err': Edist(flg, flg.exp(Y, xi1), Y1)}}

            """
            t0 = perf_counter()
            xi3, st3 = flbf.log(Y, Y1)
            t1 = perf_counter()
            ret[i]['SCIPY_LBFGS'] = {
                'time': t1 - t0,
                'err': linf(flbf.exp(Y, xi3) - Y1)}
            """

            t0 = perf_counter()
            xi4, st4 = log_lbfgs(flg, Y, Y1, tol=1e-15)
            t1 = perf_counter()
            ret[i]['CUSTOM_LBFGS'] = {'time': t1 - t0,
                                      'err': Edist(
                                          flg, flg.exp(Y, xi4), Y1)}
            ITRS['CUSTOM_LBFGS'][i] = st4['iterations']            
            FVALS['CUSTOM_LBFGS'][i] = st4['funcCount']*3
            DONES['CUSTOM_LBFGS'][i] = st4['firstorderopt'] < 1e-3            

            t0 = perf_counter()
            xi5, itr5, fvals5, fjacs5, done = log_descent(
                flg, Y, Y1, tol, max_itr=max_itr)
            t1 = perf_counter()
            ret[i]['DESCENT'] = {'time': t1 - t0,
                                 'err': Edist(flg, flg.exp(Y, xi5), Y1)}
            ITRS['DESCENT'][i] = itr5            
            FVALS['DESCENT'][i] = fvals5 + 3*fjacs5
            DONES['DESCENT'][i] = done or (ret[i]['DESCENT']['err'] < 1e-3)

            t0 = perf_counter()
            xi6, itr6, fvals6, fjacs6, done = log_steep_descent(
                flg, Y, Y1, tol, max_itr=max_itr)
            t1 = perf_counter()
            ret[i]['STEEP'] = {'time': t1 - t0,
                               'err': Edist(flg, flg.exp(Y, xi5), Y1)}
            ITRS['STEEP'][i] = itr5
            FVALS['STEEP'][i] = fvals6 + 3*fjacs6            
            DONES['STEEP'][i] = done or (ret[i]['DESCENT']['err'] < 1e-3)
            
        mat = np.zeros((len(ret), 8))
        for i in ret:
            mat[i, :] = [
                ret[i]['SCIPY_NCG']['time'],
                ret[i]['DESCENT']['time'],
                ret[i]['STEEP']['time'],                
                ret[i]['CUSTOM_LBFGS']['time'],

                ret[i]['SCIPY_NCG']['err'],
                ret[i]['DESCENT']['err'],
                ret[i]['STEEP']['err'],                     
                ret[i]['CUSTOM_LBFGS']['err']                     
            ]
        np.savez_compressed(
            'compare_flag_%s_al_%0.1f_npi_%0.2f' % ('_'.join(
                [str(aa) for aa in dvec]), alf, npi),
            mat=mat)
        print(np.mean(mat, axis=0))
        return ret, ITRS, FVALS, DONES

    ret = doit(0.5, [2, 3, 4], .4, 10, 1e-8, 300)
    ret = doit(0.5, [2, 3, 4], .99, 10, 1e-9, 300)
    ret = doit(1., [2, 3, 4], .4, 10, 1e-9, 300)
    ret = doit(1., [2, 3, 4], .99, 10, 1e-9, 300)

    ret = doit(0.1, [5, 3, 1], .5, 50, 1e-8)
    ret = doit(0.1, [5, 3, 1], .99, 50, 1e-9)
    ret = doit(0.1, [5, 3, 1], 1.3, 50, 1e-9)
    
    ret = doit(0.5, [5, 3, 1], .5, 50, 1e-8)
    ret = doit(0.5, [5, 3, 1], .99, 50, 1e-9)
    ret = doit(0.5, [5, 3, 1], 1.3, 50, 1e-9)
    
    ret = doit(1., [5, 3, 1], .5, 50, 1e-9)
    ret = doit(1., [5, 3, 1], .99, 50, 1e-9)
    ret = doit(1., [5, 3, 1], 1.3, 50, 1e-9)
    
    ret = doit(1.2, [5, 3, 1], .5, 50, 1e-9)
    ret = doit(1.2, [5, 3, 1], .99, 50, 1e-9)
    ret = doit(1.2, [5, 3, 1], 1.3, 50, 1e-9)

    ret = doit(0.5, [960, 20, 15, 5], .5, 10, 1e-9)
    ret = doit(0.5, [960, 20, 15, 5], .99, 10, 1e-9)
    ret = doit(1., [960, 20, 15, 5], .5, 10, 1e-9)
    ret = doit(1., [960, 20, 15, 5], .99, 10, 1e-9)
    
    ret = doit(0.5, [1500, 50, 20, 30], .5, 10, 1e-10)
    ret = doit(0.5, [1500, 50, 20, 30], .99, 10, 1e-10)
    ret = doit(1., [1500, 50, 20, 30], .5, 10, 1e-11)
    ret = doit(1., [1500, 50, 20, 30], .99, 10, 1e-11)

    ret = doit(0.5, [1500, 250, 100, 150], .5, 10, 1e-12)
    ret = doit(0.5, [1500, 250, 100, 150], .99, 10, 1e-12)
    ret = doit(1., [1500, 250, 100, 150], .5, 10, 1e-12)
    ret = doit(1., [1500, 250, 100, 150], .99, 10, 1e-12)
    
    ret = doit(0.5, [1500, 300, 120, 180], .5, 10, 1e-10)
    ret = doit(0.5, [1500, 300, 120, 180], .99, 10, 1e-10)
    ret = doit(1., [1500, 300, 120, 180], .5, 10, 1e-10)
    ret = doit(1., [1500, 300, 120, 180], .99, 10, 1e-10)
    
    ret = doit(0.5, [1500, 500, 200, 300], .5, 10, 1e-10)
    ret = doit(0.5, [1500, 500, 200, 300], .99, 10, 1e-10)
    ret = doit(1.0, [1500, 500, 200, 300], .5, 10, 1e-10)
    ret = doit(1., [1500, 500, 200, 300], .99, 10, 1e-10)
    

def time_fun_val_jac(flg, npi, NAR, N):
    """Inverse of exp

    Parameters
    ----------
    Y    : a manifold point
    Y1  : tangent vector

    Returns
    ----------
    eta such that flg.exp(X, eta) = Y1

    Algorithm: use the scipy.optimize trust region method
    to minimize in eta ||flg.exp(Y, eta) - Y1||_F^2
    _F is the Frobenius norm in R^{n\times d}
    The jacobian could be computed by the expm_frechet function
    """
    alf = flg.alpha[1]/flg.alpha[0]
    d = flg.dvec[1:].sum()
    adim = (flg.dvec[1:].sum()*flg.dvec[1:].sum() -
            (flg.dvec[1:]*flg.dvec[1:]).sum()) // 2
    
    vals = np.zeros((NAR, N))
    jacs = np.zeros((NAR, N))
    funs = np.zeros((NAR, N))
    n = flg.dvec.sum()
    sqrt2 = np.sqrt(2)    

    for jj in range(N):
        Y = flg.rand()
        xi = flg.randvec(Y)
        Y1 = flg.exp(Y, xi)

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
        
        from time import perf_counter
        
        for jjj in range(NAR):
            v = AR.rand(flg.dvec, k)
            t0 = perf_counter()
            dist(v.A, v.R)
            t1 = perf_counter()
            vals[jjj, jj] = t1 - t0

            t0 = perf_counter()
            jac(v.A, v.R)
            t1 = perf_counter()
            jacs[jjj, jj] = t1 - t0
            
            t0 = perf_counter()
            fun(v.A, v.R)
            t1 = perf_counter()
            funs[jjj, jj] = t1 - t0
    return vals, jacs, funs


def test_time():
    np.random.seed(0)
    alf = .5

    dvec = np.array([1500, 900, 60, 40])

    # dvec = np.array([500, 60, 30, 10])
    
    # alpha = np.array([1, alf])
    p = dvec.shape[0] - 1
    
    alpha = np.full((p, p+1), alf)
    alpha[:, 0] = 1.
    
    flg = RealFlag(dvec, alpha, log_stats=True, log_method='trust-ncg')
    NAR = 10
    N = 2
    npi = 1
    vals, jacs, funs = time_fun_val_jac(flg, npi, NAR, N)
    print(vals.mean(axis=1), jacs.mean(axis=1), funs.mean(axis=1))    
    print(vals.mean(), jacs.mean(), funs.mean())


def ft():
    np.random.seed(0)
    from scipy.linalg import expm, expm_frechet

    N = 2000
    B = np.random.randn(N, N)
    # B = B + B.T
    # A, _ = np.linalg.qr(np.random.randn(1000, 1000))

    # v = np.abs(np.random.randn(1000)) + 1
    # A = (A.T*v[None, :])@A
    A = np.random.randn(N, N)
    A = A - A.T
    # timeit -n 10 expm(A)
    # timeit -n 5 expm_frechet(A, B)
