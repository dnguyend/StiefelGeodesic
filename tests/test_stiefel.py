import numpy as np
from scipy.linalg import null_space, expm, logm, expm_frechet

from ManNullRange.manifolds.tools import sym, asym, vecah, unvecah
from ManNullRange.manifolds.RealStiefel import RealStiefel
from StiefelGeodesic.manifolds.tools import linf, sbmat
from StiefelGeodesic.manifolds.LogStiefel import (
    log_custom_trust_ncg, log_lbfgs, log_descent,
    log_steep_descent, log_shooting_AR, log_shooting_raw)


def custom_vs_scipy():
    """ custom ncg and lbfgs versus scipy
    """
    from time import perf_counter
    np.random.seed(0)
    
    def doit(alf, n, p, Nruns=10):
        alpha = np.array([1, alf])
        stf = RealStiefel(
            n, p, alpha, log_stats=True, log_method='trust-ncg')

        slbf = RealStiefel(
            n, p, alpha, log_stats=True, log_method='L-BFGS-B')
        slbf.log_gtol = 1e-6
        
        ret = {}
        
        for i in range(Nruns):
            # print('Doing %d' % i)
            Y = stf.rand()
            xi = stf.randvec(Y)*np.pi*.5
            Y1 = stf.exp(Y, xi)

            t0 = perf_counter()
            xi1, st1 = stf.log(Y, Y1)
            t1 = perf_counter()
            # print('D %f' % (t1-t0))
            # print(linf(stf.exp(Y, xi1) - Y1))
            ret[i] = {'SCIPY_NCG' : {
                'time': t1 - t0,
                'err': linf(stf.exp(Y, xi1) - Y1)}}
                
            t0 = perf_counter()
            xi2, st2 = log_custom_trust_ncg(stf, Y, Y1)
            t1 = perf_counter()
            ret[i]['CUSTOM_NCG'] = {'time': t1 - t0,
                                    'err': linf(stf.exp(Y, xi2) - Y1)}

            t0 = perf_counter()
            xi3, st3 = slbf.log(Y, Y1)
            t1 = perf_counter()
            ret[i]['SCIPY_LBFGS'] = {
                'time': t1 - t0,
                'err': linf(stf.exp(Y, xi3) - Y1)}

            t0 = perf_counter()
            xi4, st4 = log_lbfgs(stf, Y, Y1, tol=1e-6)
            t1 = perf_counter()
            ret[i]['CUSTOM_LBFGS'] = {'time': t1 - t0,
                                      'err': linf(stf.exp(Y, xi4) - Y1)}
            
        mat = np.zeros((len(ret), 8))
        # print(ret)
        for i in ret:
            mat[i, :] = [ret[i]['SCIPY_NCG']['time'],
                         ret[i]['CUSTOM_NCG']['time'],
                         ret[i]['SCIPY_LBFGS']['time'],
                         ret[i]['CUSTOM_LBFGS']['time'],
                         ret[i]['SCIPY_NCG']['err'],
                         ret[i]['CUSTOM_NCG']['err'],
                         ret[i]['SCIPY_LBFGS']['err'],
                         ret[i]['CUSTOM_LBFGS']['err']
                         ]

        print(alf, n, p, np.mean(mat, axis=0))

    doit(.5, 5, 3)
    doit(.5, 1500, 30)
    doit(.5, 1500, 100)

    doit(1, 5, 3)
    doit(1, 1500, 30)
    doit(1, 1500, 100)


def compare_all():
    from geomstats.geometry.stiefel import StiefelCanonicalMetric

    from time import perf_counter
    
    def doit(alf, n, p, npi, Nsamples, tol):
        np.random.seed(0)
        alpha = np.array([1, alf])

        stf = RealStiefel(n, p, alpha, log_stats=True, log_method='trust-ncg')
        stf.log_gtol = None

        gstf = StiefelCanonicalMetric(n, p)
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
            Y = stf.rand()
            xi = stf.randvec(Y)*np.pi*npi
            Y1 = stf.exp(Y, xi)        
            t0 = perf_counter()
            xi2 = gstf.log(Y1.copy(), Y.copy(), tol=1e-6)
            t1 = perf_counter()

            if alf == .5:
                ret[i] = {'Z' : {
                    'time': t1 - t0,
                    'err': linf(gstf.exp(xi2, Y) - Y1)}}
            else:
                ret[i] = {}                

            t0 = perf_counter()
            xi4, st4 = log_lbfgs(stf, Y, Y1, tol=1e-15)
            t1 = perf_counter()
            ret[i]['CUSTOM_LBFGS'] = {'time': t1 - t0,
                                      'err': linf(stf.exp(Y, xi4) - Y1)}
            
            ITRS['CUSTOM_LBFGS'][i] = st4['iterations']
            FVALS['CUSTOM_LBFGS'][i] = st4['funcCount']*3
            DONES['CUSTOM_LBFGS'][i] = st4['firstorderopt'] < 1e-3

            t0 = perf_counter()
            xi5, itr5, fvals5, fjacs5, done = log_descent(stf, Y, Y1, tol)
            t1 = perf_counter()
            ret[i]['DESCENT'] = {'time': t1 - t0,
                                 'err': linf(stf.exp(Y, xi5) - Y1)}
            ITRS['DESCENT'][i] = itr5
            FVALS['DESCENT'][i] = fvals5 + 3*fjacs5
            DONES['DESCENT'][i] = done or (ret[i]['DESCENT']['err'] < 1e-3)

            t0 = perf_counter()
            xi6, itr6, fvals6, fjacs6, done =\
                log_steep_descent(stf, Y, Y1, tol, max_itr=100)
            t1 = perf_counter()
            ret[i]['STEEP'] = {'time': t1 - t0,
                               'err': linf(stf.exp(Y, xi6) - Y1)}
            ITRS['STEEP'][i] = itr6
            FVALS['STEEP'][i] = fvals6 + 3*fjacs6
            DONES['STEEP'][i] = done or (ret[i]['STEEP']['err'] < 1e-3)

        if (alf == .5):
            header = "Ztime DESCENTtime STEEPtime LBFGStime Zerr DESCENTerr STEEPerr LBFGSerr"
            mat = np.zeros((len(ret), 8))
            for i in ret:
                mat[i, :] = [ret[i]['Z']['time'],
                             ret[i]['DESCENT']['time'],
                             ret[i]['STEEP']['time'],
                             ret[i]['CUSTOM_LBFGS']['time'],

                             ret[i]['Z']['err'],
                             ret[i]['DESCENT']['err'],
                             ret[i]['STEEP']['err'],                         
                             ret[i]['CUSTOM_LBFGS']['err']                     
                             ]
        else:
            mat = np.zeros((len(ret), 6))
            header = "DESCENTtime STEEPtime LBFGStime DESCENTerr STEEPerr LBFGSerr"
            for i in ret:                
                mat[i, :] = [
                    ret[i]['DESCENT']['time'],
                    ret[i]['STEEP']['time'],
                    ret[i]['CUSTOM_LBFGS']['time'],

                    ret[i]['DESCENT']['err'],
                    ret[i]['STEEP']['err'],                         
                    ret[i]['CUSTOM_LBFGS']['err']                     
                ]
                
        np.savez_compressed(
            'compare_stf_%d_%d_al_%0.1f_npi_%0.2f' % (n, p, alf, npi),
            mat=mat)
        print("n=%d p=%d alpha=%f npi=%f" % (n, p, alf, npi))
        print(header)
        print(np.mean(mat, axis=0))
        print("Mean Iters")
        print("DESCENT %f STEEP %f LBFGS %f" % 
              (ITRS['DESCENT'].mean(),
               ITRS['STEEP'].mean(),
               ITRS['CUSTOM_LBFGS'].mean()))
        print("Mean FVALS")
        print("DESCENT %f STEEP %f LBFGS %f" % 
              (FVALS['DESCENT'].mean(),
               FVALS['STEEP'].mean(),
               FVALS['CUSTOM_LBFGS'].mean()))

        return ret, ITRS, FVALS, DONES

    ret = doit(0.1, 4, 2, .5, 10, 1e-8)
    ret = doit(0.1, 4, 2, .99, 10, 1e-8)
    ret = doit(0.1, 4, 2, 1.3, 10, 1e-8)

    ret = doit(0.5, 4, 2, .5, 10, 1e-8)
    ret = doit(0.5, 4, 2, .99, 10, 1e-8)
    ret = doit(0.5, 4, 2, 1.3, 10, 1e-8)
    
    rr = {}
    for i in range(10):
        Y = stf.rand()
        xi = stf.randvec(Y)*np.pi*1.3
        Y1 = stf.exp(Y, xi)        
        t0 = perf_counter()
        xi2 = gstf.log(Y1.copy(), Y.copy())
        t1 = perf_counter()

        rr[i] = {'Z' : {
            'time': t1 - t0,
            'err': linf(gstf.exp(xi2, Y) - Y1)}}
    
    tbl = {}
    fnames = [a for a in os.listdir('.') if a.startswith('compare_stf_1500_200_al_0.5_')]
    for a in fnames:
        tbl[a] = np.load(a, allow_pickle=True)['mat'].tolist()

    # first One
    ret = doit(0.5, 4, 2, .5, 10, 1e-8)
    ret = doit(0.5, 4, 2, .99, 10, 1e-8)
    ret = doit(0.5, 4, 2, 1.3, 10, 1e-8)

    ret = doit(1., 4, 2, .5, 10, 1e-8)
    ret = doit(1., 4, 2, .99, 10, 1e-8)
    ret = doit(1., 4, 2, 1.3, 10, 1e-8)
    
    ret = doit(1.3, 4, 2, .5, 10, 1e-8)
    ret = doit(1.3, 4, 2, .99, 10, 1e-8)
    ret = doit(1.3, 4, 2, 1.3, 10, 1e-8)
    
    ret = doit(0.5, 7, 3, .5, 10, 1e-8)
    ret = doit(0.5, 7, 3, .8, 10, 1e-9)
    ret = doit(1., 7, 3, .5, 10, 1e-9)
    ret = doit(1., 7, 3, .8, 10, 1e-9)

    # second one
    ret = doit(0.5, 100, 40, .5, 10, 1e-9)
    ret = doit(0.5, 100, 40, .99, 10, 1e-9)
    ret = doit(0.5, 100, 40, 1.3, 10, 1e-9)
    
    ret = doit(1., 100, 40, .5, 10, 1e-9)
    ret = doit(1., 100, 40, .99, 10, 1e-9)

    ret = doit(0.5, 1500, 100, .5, 10, 1e-10)
    ret = doit(0.5, 1500, 100, .8, 10, 1e-12)
    
    ret = doit(0.5, 1500, 200, .5, 10, 1e-10)
    ret = doit(0.5, 1500, 200, .8, 10, 1e-12)
    ret = doit(1., 1500, 200, .5, 10, 1e-11)
    ret = doit(1., 1500, 200, .8, 10, 1e-11)

    ret = doit(0.5, 1500, 500, .5, 10, 5e-12)
    ret = doit(0.5, 1500, 500, .8, 10, 5e-12)
    ret = doit(1., 1500, 500, .5, 10, 5e-12)
    ret = doit(1., 1500, 500, .8, 10, 5e-12)
    
    ret = doit(0.5, 1500, 700, .5, 10, 5e-12)
    ret = doit(0.5, 1500, 700, .99, 10, 1e-12)
    ret = doit(1., 1500, 700, .5, 10, 1e-12)
    ret = doit(1., 1500, 700, .99, 10, 1e-12)

    # third one
    ret = doit(0.5, 1500, 1000, .5, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())
    
    ret = doit(0.5, 1500, 1000, .99, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())

    ret = doit(0.5, 1500, 1000, 1.3, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())
    
    ret = doit(1.0, 1500, 1000, .5, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())
    
    ret = doit(1., 1500, 1000, .99, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())    
    

def compare_all_raw():
    from geomstats.geometry.stiefel import StiefelCanonicalMetric

    from time import perf_counter
    
    def doit(alf, n, p, npi, Nsamples, tol, with_ncg=True):
        np.random.seed(0)
        alpha = np.array([1, alf])

        stf = RealStiefel(n, p, alpha, log_stats=True, log_method='trust-ncg')
        stf.log_gtol = None

        gstf = StiefelCanonicalMetric(n, p)
        # slbf = RealStiefel(n, p, alpha, log_stats=True, log_method='l-bfgs-b')
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
            Y = stf.rand()
            xi = stf.randvec(Y)*np.pi*npi
            Y1 = stf.exp(Y, xi)        
            t0 = perf_counter()
            xi2 = gstf.log(Y1.copy(), Y.copy(), tol=1e-6)
            t1 = perf_counter()

            if alf == .5:
                ret[i] = {'Z' : {
                    'time': t1 - t0,
                    'err': linf(gstf.exp(xi2, Y) - Y1)}}
            else:
                ret[i] = {}                

            # print('Done Z')
            if with_ncg:
                t0 = perf_counter()
                xi1, st1 = stf.log(Y, Y1)
                t1 = perf_counter()
                # print('D %f' % (t1-t0))
                # print(linf(stf.exp(Y, xi1) - Y1))
                ret[i]['SCIPY_NCG'] = {
                    'time': t1 - t0,
                    'err': linf(stf.exp(Y, xi1) - Y1)}
                # print('Done SCI_NCG')        

            """
                t0 = perf_counter()
                xi2, st2 = log_custom_trust_ncg(stf, Y, Y1)
                t1 = perf_counter()
                ret[i]['CUSTOM_NCG'] = {'time': t1 - t0,
                                        'err': linf(stf.exp(Y, xi2) - Y1)}
            # print('Done CUSTOM_NCG')        
            """
            """
            t0 = perf_counter()
            xi3, st3 = slbf.log(Y, Y1)
            t1 = perf_counter()
            ret[i]['SCIPY_LBFGS'] = {
                'time': t1 - t0,
                'err': linf(stf.exp(Y, xi3) - Y1)}
            """

            t0 = perf_counter()
            xi4, st4 = log_lbfgs(stf, Y, Y1, tol=1e-15)
            t1 = perf_counter()
            ret[i]['CUSTOM_LBFGS'] = {'time': t1 - t0,
                                      'err': linf(stf.exp(Y, xi4) - Y1)}
            
            ITRS['CUSTOM_LBFGS'][i] = st4['iterations']
            FVALS['CUSTOM_LBFGS'][i] = st4['funcCount']*3
            DONES['CUSTOM_LBFGS'][i] = st4['firstorderopt'] < 1e-3

            t0 = perf_counter()
            xi5, itr5, fvals5, fjacs5, done = log_descent(stf, Y, Y1, tol)
            t1 = perf_counter()
            ret[i]['DESCENT'] = {'time': t1 - t0,
                                 'err': linf(stf.exp(Y, xi5) - Y1)}
            ITRS['DESCENT'][i] = itr5
            FVALS['DESCENT'][i] = fvals5 + 3*fjacs5
            DONES['DESCENT'][i] = done or (ret[i]['DESCENT']['err'] < 1e-3)

            t0 = perf_counter()
            xi6, itr6, fvals6, fjacs6, done =\
                log_steep_descent(stf, Y, Y1, tol, max_itr=100)
            t1 = perf_counter()
            ret[i]['STEEP'] = {'time': t1 - t0,
                               'err': linf(stf.exp(Y, xi6) - Y1)}
            ITRS['STEEP'][i] = itr6
            FVALS['STEEP'][i] = fvals6 + 3*fjacs6
            DONES['STEEP'][i] = done or (ret[i]['STEEP']['err'] < 1e-3)

        if with_ncg and (alf == .5):
            mat = np.zeros((len(ret), 10))
            for i in ret:
                mat[i, :] = [ret[i]['Z']['time'],
                             ret[i]['SCIPY_NCG']['time'],
                             ret[i]['DESCENT']['time'],
                             ret[i]['STEEP']['time'],
                             ret[i]['CUSTOM_LBFGS']['time'],

                             ret[i]['Z']['err'],
                             ret[i]['SCIPY_NCG']['err'],
                             ret[i]['DESCENT']['err'],
                             ret[i]['STEEP']['err'],                         
                             ret[i]['CUSTOM_LBFGS']['err']                     
                             ]
        elif (not with_ncg) and (alf == .5):
            mat = np.zeros((len(ret), 8))
            for i in ret:
                mat[i, :] = [ret[i]['Z']['time'],
                             ret[i]['DESCENT']['time'],
                             ret[i]['STEEP']['time'],
                             ret[i]['CUSTOM_LBFGS']['time'],

                             ret[i]['Z']['err'],
                             ret[i]['DESCENT']['err'],
                             ret[i]['STEEP']['err'],                         
                             ret[i]['CUSTOM_LBFGS']['err']                     
                             ]
        elif with_ncg:
            mat = np.zeros((len(ret), 8))
            for i in ret:
                mat[i, :] = [ret[i]['SCIPY_NCG']['time'],
                             ret[i]['DESCENT']['time'],
                             ret[i]['STEEP']['time'],
                             ret[i]['CUSTOM_LBFGS']['time'],

                             ret[i]['SCIPY_NCG']['err'],
                             ret[i]['DESCENT']['err'],
                             ret[i]['STEEP']['err'],                         
                             ret[i]['CUSTOM_LBFGS']['err']                     
                             ]
            else:
                mat = np.zeros((len(ret), 6))
                for i in ret:                
                    mat[i, :] = [
                        ret[i]['DESCENT']['time'],
                        ret[i]['STEEP']['time'],
                        ret[i]['CUSTOM_LBFGS']['time'],

                        ret[i]['DESCENT']['err'],
                        ret[i]['STEEP']['err'],                         
                        ret[i]['CUSTOM_LBFGS']['err']                     
                    ]
            
        np.savez_compressed(
            'compare_stf_%d_%d_al_%0.1f_npi_%0.2f' % (n, p, alf, npi),
            mat=mat)
        print(np.mean(mat, axis=0))
        return ret, ITRS, FVALS, DONES

    ret = doit(0.1, 4, 2, .5, 10, 1e-8, with_ncg=True)
    ret = doit(0.1, 4, 2, .99, 10, 1e-8)
    ret = doit(0.1, 4, 2, 1.3, 10, 1e-8)

    # first One
    ret = doit(0.5, 4, 2, .5, 10, 1e-8)
    ret = doit(0.5, 4, 2, .99, 10, 1e-8)
    ret = doit(0.5, 4, 2, 1.3, 10, 1e-8)

    ret = doit(1., 4, 2, .5, 10, 1e-8)
    ret = doit(1., 4, 2, .99, 10, 1e-8)
    ret = doit(1., 4, 2, 1.3, 10, 1e-8)
    
    ret = doit(1.3, 4, 2, .5, 10, 1e-8)
    ret = doit(1.3, 4, 2, .99, 10, 1e-8)
    ret = doit(1.3, 4, 2, 1.3, 10, 1e-8)
    
    ret = doit(0.5, 7, 3, .5, 10, 1e-8)
    ret = doit(0.5, 7, 3, .8, 10, 1e-9)
    ret = doit(1., 7, 3, .5, 10, 1e-9)
    ret = doit(1., 7, 3, .8, 10, 1e-9)

    # second one
    ret = doit(0.5, 100, 40, .5, 10, 1e-9)
    ret = doit(0.5, 100, 40, .99, 10, 1e-9)
    ret = doit(0.5, 100, 40, 1.3, 10, 1e-9)
    
    ret = doit(1., 100, 40, .5, 10, 1e-9)
    ret = doit(1., 100, 40, .99, 10, 1e-9)
    
    ret = doit(0.5, 1500, 200, .5, 10, 1e-10)
    ret = doit(0.5, 1500, 200, .8, 10, 1e-12)
    ret = doit(1., 1500, 200, .5, 10, 1e-11)
    ret = doit(1., 1500, 200, .8, 10, 1e-11)

    ret = doit(0.5, 1500, 500, .5, 10, 5e-12)
    ret = doit(0.5, 1500, 500, .8, 10, 5e-12)
    ret = doit(1., 1500, 500, .5, 10, 5e-12)
    ret = doit(1., 1500, 500, .8, 10, 5e-12)
    
    ret = doit(0.5, 1500, 700, .5, 10, 5e-12)
    ret = doit(0.5, 1500, 700, .99, 10, 1e-12)
    ret = doit(1., 1500, 700, .5, 10, 1e-12)
    ret = doit(1., 1500, 700, .99, 10, 1e-12)

    # third one
    ret = doit(0.5, 1500, 1000, .5, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())
    
    ret = doit(0.5, 1500, 1000, .99, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())

    ret = doit(0.5, 1500, 1000, 1.3, 10, 1e-12)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())
    
    ret = doit(1.0, 1500, 1000, .5, 10, 1e-12, with_ncg=False)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())
    
    ret = doit(1., 1500, 1000, .99, 10, 1e-12, with_ncg=False)
    print(ret[2]['CUSTOM_LBFGS'].mean(), ret[2]['DESCENT'].mean())


def compare_shooting():
    from geomstats.geometry.stiefel import StiefelCanonicalMetric
    from time import perf_counter
    
    def shootit(alf, n, p, npi, Nsamples, tol):
        np.random.seed(0)
        alpha = np.array([1, alf])

        stf = RealStiefel(n, p, alpha, log_stats=True, log_method='trust-ncg')
        stf.log_gtol = None

        gstf = StiefelCanonicalMetric(n, p)
        # slbf = RealStiefel(n, p, alpha, log_stats=True, log_method='l-bfgs-b')
        ret = {}
        CNTS = {'DESCENT': np.empty(Nsamples),
                'STEEP': np.empty(Nsamples),                
                'CUSTOM_LBFGS': np.empty(Nsamples)}
        DONES = {'DESCENT': np.empty(Nsamples, dtype=int),
                 'STEEP': np.empty(Nsamples, dtype=int),
                 'CUSTOM_LBFGS': np.empty(Nsamples, dtype=int)}
        
        for i in range(Nsamples):
            print('Doing %d' % i)
            Y = stf.rand()
            xi = stf.randvec(Y)*np.pi*npi
            Y1 = stf.exp(Y, xi)        
            t0 = perf_counter()
            xi2 = gstf.log(Y1.copy(), Y.copy(), tol=1e-6)
            t1 = perf_counter()

            ret[i] = {'Z' : {
                'time': t1 - t0,
                'err': linf(gstf.exp(xi2, Y) - Y1)}}

            t0 = perf_counter()
            xi1 = log_shooting_AR(stf, Y, Y1, tol=1e-8)
            t1 = perf_counter()
            ret[i]['SHOOT_AR'] = {
                'time': t1 - t0,
                'err': linf(stf.exp(Y, xi1) - Y1)}

            t0 = perf_counter()
            xi2 = log_shooting_raw(stf, Y, Y1, tol=1e-10)
            # xi2 = rlog_shooting_org(stf, Y, Y1, tol=1e-10)
            print(linf(stf.exp(Y, xi2) - Y1))
            
            t1 = perf_counter()
            ret[i]['SHOOT_RAW'] = {'time': t1 - t0,
                                   'err': linf(stf.exp(Y, xi2) - Y1)}

            t0 = perf_counter()
            xi4, st4 = log_lbfgs(stf, Y, Y1, tol=1e-15)
            t1 = perf_counter()
            ret[i]['CUSTOM_LBFGS'] = {'time': t1 - t0,
                                      'err': linf(stf.exp(Y, xi4) - Y1)}
            
            CNTS['CUSTOM_LBFGS'][i] = st4['funcCount']
            DONES['CUSTOM_LBFGS'][i] = st4['firstorderopt'] < 1e-3

            t0 = perf_counter()
            xi5, cnt, done = log_descent(stf, Y, Y1, tol)
            t1 = perf_counter()
            ret[i]['DESCENT'] = {'time': t1 - t0,
                                 'err': linf(stf.exp(Y, xi5) - Y1)}
            CNTS['DESCENT'][i] = cnt
            DONES['DESCENT'][i] = done or (ret[i]['DESCENT']['err'] < 1e-3)
            
        mat = np.zeros((len(ret), 10))
        for i in ret:
            mat[i, :] = [ret[i]['Z']['time'],
                         ret[i]['SHOOT_AR']['time'],
                         ret[i]['SHOOT_RAW']['time'],
                         ret[i]['DESCENT']['time'],
                         ret[i]['CUSTOM_LBFGS']['time'],

                         ret[i]['Z']['err'],
                         ret[i]['SHOOT_AR']['err'],
                         ret[i]['SHOOT_RAW']['err'],
                         ret[i]['DESCENT']['err'],
                         ret[i]['CUSTOM_LBFGS']['err']]
        print(np.mean(mat, axis=0))
        return ret, CNTS, DONES

    ret = shootit(0.1, 4, 2, .5, 10, 1e-8)
    ret = shootit(0.1, 4, 2, .99, 10, 1e-8)
    ret = shootit(0.1, 4, 2, 1.3, 10, 1e-8)

    # first One
    ret = shootit(0.5, 4, 2, .5, 10, 1e-8)
    ret = shootit(0.5, 4, 2, .99, 10, 1e-8)
    ret = shootit(0.5, 4, 2, 1.3, 10, 1e-8)

    ret = shootit(1., 4, 2, .5, 10, 1e-8)

    ret = shootit(1., 4, 2, .3, 10, 1e-8)
    ret = shootit(.5, 10, 2, .44, 10, 1e-8)
    ret = shootit(.5, 10, 2, .5, 10, 1e-8)
    ret = shootit(.5, 30, 2, .5, 10, 1e-8)

    ret = shootit(1., 4, 2, .99, 10, 1e-8)
    ret = shootit(1., 4, 2, 1.3, 10, 1e-8)
    
    ret = shootit(1.3, 4, 2, .5, 10, 1e-8)
    ret = shootit(1.3, 4, 2, .99, 10, 1e-8)
    ret = shootit(1.3, 4, 2, 1.3, 10, 1e-8)
    
    ret = shootit(0.5, 7, 3, .44, 10, 1e-8)
    ret = shootit(0.5, 7, 3, .99, 10, 1e-9)
    ret = shootit(0.5, 7, 3, 1.3, 10, 1e-9)
    
    ret = shootit(1., 7, 3, .5, 10, 1e-9)
    ret = shootit(1., 7, 3, .8, 10, 1e-9)

    # second one
    ret = shootit(0.5, 100, 40, .5, 10, 1e-9)
    ret = shootit(0.5, 100, 40, .99, 10, 1e-9)
    ret = shootit(0.5, 100, 40, 1.3, 10, 1e-9)
    
    ret = shootit(1., 100, 40, .5, 10, 1e-9)
    ret = shootit(1., 100, 40, .99, 10, 1e-9)
    
    ret = shootit(0.5, 1500, 200, .5, 10, 1e-10)
    ret = shootit(0.5, 1500, 200, .8, 10, 1e-12)
    ret = shootit(1., 1500, 200, .5, 10, 1e-11)
    ret = shootit(1., 1500, 200, .8, 10, 1e-11)

    ret = shootit(0.5, 1500, 500, .5, 10, 5e-12)
    ret = shootit(0.5, 1500, 500, .8, 10, 5e-12)
    ret = shootit(1., 1500, 500, .5, 10, 5e-12)
    ret = shootit(1., 1500, 500, .8, 10, 5e-12)
    
    ret = shootit(0.5, 1500, 700, .5, 10, 5e-12)
    ret = shootit(0.5, 1500, 700, .99, 10, 1e-12)
    ret = shootit(1., 1500, 700, .5, 10, 1e-12)
    ret = shootit(1., 1500, 700, .99, 10, 1e-12)

    # third one
    ret = shootit(0.5, 1500, 1000, .5, 10, 1e-12)
    ret = shootit(0.5, 1500, 1000, .99, 10, 1e-12)
    ret = shootit(0.5, 1500, 1000, 1.3, 10, 1e-12)
    
    ret = shootit(1.0, 1500, 1000, .5, 10, 1e-12)
    ret = shootit(1., 1500, 1000, .99, 10, 1e-12)    
    

def time_fun_val_jac(stf, npi, NAR, N):
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
    vals = np.zeros((NAR, N))
    jacs = np.zeros((NAR, N))
    funs = np.zeros((NAR, N))

    for jj in range(N):
        Y = stf.rand()
        xi = stf.randvec(Y)
        Y1 = stf.exp(Y, xi)
        
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
        if k == 0:
            # Y1 and Y has the same linear span
            A = logm(Y.T @ Y1)

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

            return -np.sum(Y1 * ((Y@M+Q@N)@expm((1-2*alf)*A)))

        def jac(v):
            A, R = unvec(v)
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

            return vec(partA, partR)

        def fun(v):
            A, R = unvec(v)
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

            return -np.sum(Y1 * ((Y@M+Q@N)@expm((1-2*alf)*A))),\
                vec(partA, partR)

        from time import perf_counter
        
        for jjj in range(NAR):
            v = np.random.rand(adim + d*k)
            t0 = perf_counter()
            dist(v)
            t1 = perf_counter()
            vals[jjj, jj] = t1 - t0

            t0 = perf_counter()
            jac(v)
            t1 = perf_counter()
            jacs[jjj, jj] = t1 - t0
            
            t0 = perf_counter()
            fun(v)
            t1 = perf_counter()
            funs[jjj, jj] = t1 - t0
    return vals, jacs, funs


def test_time():
    np.random.seed(0)

    alf = .5
    n = 1500
    d = 1000

    alpha = np.array([1, alf])    
    stf = RealStiefel(n, d, alpha, log_stats=True, log_method='trust-ncg')
    NAR = 10
    N = 2
    npi = 1
    vals, jacs, funs = time_fun_val_jac(stf, npi, NAR, N)
    print(vals.mean(axis=1), jacs.mean(axis=1), funs.mean(axis=1))    
    print(vals.mean(), jacs.mean(), funs.mean())

