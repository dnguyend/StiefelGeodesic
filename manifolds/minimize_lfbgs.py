import numpy as np
from numpy import zeros,  concatenate, empty_like
from numba import njit


def isLegal(v):
    return (np.sum(np.any(~np.isreal(v))) == 0) and\
        (np.sum(np.isnan(v)) == 0) and (np.sum(np.isinf(v))==0)


@njit
def _lbfgs_calc(g, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag):
    """BFGS Search Direction
    
    This function returns the (L-BFGS) approximate inverse Hessian,
    multiplied by the negative gradient
    Use saved data Y S YS
    To avoid repeated allocation of memory use fix Y, S, YS but
    cycling the indices
    data is saved between start and end, inclusively both ends
    """

    # Set up indexing
    nVars, maxCorrections = S.shape
    if lbfgs_start == 0:
        ind = np.arange(lbfgs_end+1)
        nCor = lbfgs_end - lbfgs_start + 1
    else:
        ind = np.concatenate(
            (np.arange(
                lbfgs_start, maxCorrections),
             np.arange(lbfgs_end+1)))
        nCor = maxCorrections
    al = np.zeros(nCor)
    # we use q and z as different valriables in the algorithm
    # description but here we use one name to save storage    
    z = -g
    nid = ind.shape[0]
    for j in range(ind.shape[0]):
        i = ind[nid-j-1]
        al[i] = np.sum(S[:, i]*z)*rho[i]
        z -= al[i]*Y[:, i]
        
    # Multiply by Initial Hessian.
    z = Hdiag*z
    
    for i in ind:
        be_i = np.sum(Y[:, i]*z)*rho[i]
        z += S[:, i]*(al[i]-be_i)
    return z


@njit
def _save(y, s, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag, pcondx):
    """ append data to the memory. To avoid repeated allocation
    of memory use fixed storage but keep tract of indices, and recycle if the
    storage is full.
    Data to save:
    S: change in x (steps) (also called q_i)
    Y: Change in gradient
    YS: contraction of Y and S
    """
    ys = np.sum(y*s)
    skipped = 0
    corrections = S.shape[1]
    if ys > 1e-10:
        if lbfgs_end < corrections-1:
            lbfgs_end = lbfgs_end+1
            if lbfgs_start != 0:
                if lbfgs_start == corrections-1:
                    lbfgs_start = 0
                else:
                    lbfgs_start = lbfgs_start+1
        else:
            lbfgs_start = min(1, corrections-1)
            lbfgs_end = 0

        S[:, lbfgs_end] = s
        Y[:, lbfgs_end] = y
        rho[lbfgs_end] = 1/ys

        # Update scale of initial Hessian approximation
        Hdiag = ys/np.sum(y*pcondx*y)*pcondx
    else:
        skipped = 1
    return lbfgs_start, lbfgs_end, Hdiag, skipped


@njit
def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = np.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    d_norm = np.abs(d).max()
    g = g.copy()
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.copy()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if np.abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.copy()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.copy()
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if np.abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if np.abs(t - max(bracket)) < np.abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.copy()
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if np.abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.copy()
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


def minimize(funObj, x0, options):
    maxFunEvals, maxIter, optTol, progTol,\
        corrections, c1, c2, max_ls, Pcond = options
    
    # Initialize
    p = x0.shape[0]
    d = zeros(p)
    x = x0.copy()
    t = 1
    pcondx = Pcond(x)
    
    funEvalMultiplier = 1
    
    # Evaluate Initial Point
    f, g = funObj(x)
    funEvals = 1

    # Compute optimality of initial point
    optCond = np.max(np.abs(g))

    # Exit if initial point is optimal
    if optCond <= optTol:
        exitflag = 1
        msg = 'Optimality Condition below optTol'
        output = {'iterations': 0,
                  'funcCount': 1,
                  'firstorderopt': max(np.abs(g)),
                  'message': msg}
        return x, f, exitflag, output

    d = -g # Initially use steepest descent direction
    S = zeros((p, corrections))
    Y = zeros((p, corrections))
    rho = zeros(corrections)
    lbfgs_start = -1
    lbfgs_end = -1
    Hdiag = np.ones(p)
    
    # Perform up to a maximum of 'maxIter' descent steps:
    for i in range(maxIter):
        # COMPUTE DESCENT DIRECTION 
        # Update the direction and step sizes
        if i > 0:
            lbfgs_start, lbfgs_end, Hdiag, skipped =\
                _save(
                    g-g_old, t*d, S, Y, rho,
                    lbfgs_start, lbfgs_end, Hdiag, pcondx)
            d = _lbfgs_calc(g, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag)
        g_old = g

        if not isLegal(d):
            print('Step direction is illegal!\n')
            output = {'iterations': i,
                      'funcCount': funEvals*funEvalMultiplier,
                      'firstorderopt': np.nan,
                      'message': 'Step direction is illegal!\n'}

            return x, f, -1, output

        # COMPUTE STEP LENGTH
        # Directional Derivative
        gtd = np.sum(g*d)

        # Check that progress can be made along direction
        if gtd > -progTol:
            exitflag = 2
            msg = 'Directional Derivative below progTol'
            break

        # Select Initial Guess
        if i == 0:
            t = min(1, 1/np.sum(np.abs(g)))
        else:
            t = 1
        f_old = f
        gtd_old = gtd
        
        def obj_func(x, t, d):
            return funObj(x + t*d)
        f, g, t, LSfunEvals = _strong_wolfe(
            obj_func, x, t, d, f, g, gtd, c1, c2, progTol, max_ls)
        funEvals = funEvals + LSfunEvals
        x = x + t*d

        # Check Optimality Condition
        if optCond <= optTol:
            exitflag = 1
            msg = 'Optimality Condition below optTol'
            break

        # Check for lack of progress

        if max(np.abs(t*d)) <= progTol:
            exitflag = 2
            msg = 'Step Size below progTol'
            break

        if abs(f-f_old) < progTol:
            exitflag = 2
            msg = 'Function Value changing by less than progTol'
            break

        # Check for going over iteration/evaluation limit
        if funEvals*funEvalMultiplier >= maxFunEvals:
            exitflag = 0
            msg = 'Reached Maximum Number of Function Evaluations'
            break

        if i == maxIter:
            exitflag = 0
            msg = 'Reached Maximum Number of Iterations'
            break

    output = {'iterations': i,
              'funcCount': funEvals*funEvalMultiplier,
              'firstorderopt': max(abs(g)),
              'message': msg}
    return x, f, exitflag, output


def test():
    def pcond(x):
        return 1/np.arange(1, x.shape[0]+1)

    def pcondI(x):
        return np.ones_like(x)    
    
    maxFunEvals = 100
    maxIter = 100
    optTol = 1e-15
    progTol = 1e-15
    corrections = 10  # historical values stored
    c1 = 1e-4
    c2 = 0.9
    max_ls = 25
    Pcond = pcond
    
    options = (maxFunEvals, maxIter, optTol, progTol,
               corrections, c1, c2, max_ls, Pcond)

    optionsI = (maxFunEvals, maxIter, optTol, progTol,
                corrections, c1, c2, max_ls, pcondI)

    p = 50
    v = np.arange(p)*np.arange(p)
    
    def funObj(x):
        return np.sum((x-v)*np.arange(1, x.shape[0]+1)*(x-v)),\
            2*np.arange(1, x.shape[0]+1)*(x-v)

    x0 = np.ones(p)

    x, f, exitflag, output = minimize(funObj, x0, options)
    x, f, exitflag, output = minimize(funObj, x0, optionsI)
    
    def rosenbrock(x):
        """ rosenbrock This function returns the function value, partial derivatives
        and Hessian of the (general dimension) rosenbrock function, given by:
        
               f(x) = sum_{i=1:D-1} 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2 
        
         where D is the dimension of x. The true minimum is 0 at x = (1 1 ... 1).
        
        """
        D = x.shape[0]
        f = np.sum(100*(x[1:D]-x[:D-1]**2)**2 + (1-x[:D-1])**2)

        df = zeros(D)
        # df[:D-1] = - 400*x[:D-1]*(x[1:D]-x[:D-1]**2) - 2*(1-x[1:D-1])
        # df[1:D] += 200*(x[1:D]-x[:D-1]**2)

        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        df[1:-1] = (200 * (xm - xm_m1**2) -
                    400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
        df[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        df[-1] = 200 * (x[-1] - x[-2]**2)        
        return f, df

    x = np.random.randn(p)
    Dx = np.random.randn(p)
    dlt = 1e-8
    f0, df0 = rosenbrock(x)
    f1, df1 = rosenbrock(x+dlt*Dx)
    print((f1 - f0)/dlt)
    print(np.sum(df0*Dx))

    from time import perf_counter
    import scipy.optimize as sco
    
    p = 5
    x0 = np.random.randn(p)

    def pcondR(y):
        return np.ones_like(y)
        x = np.ones_like(y)
        ret = np.empty_like(x)
        ret[0] = (1200*x[0]**2 - 400*x[1]+2)
        ret[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
        ret[-1] = 200
        # H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1) + np.diag(ret)
        # ev, _ = np.linalg.eigh(H)
        return 1/ret

    optionsR = (maxFunEvals, maxIter, optTol, progTol,
                corrections, c1, c2, max_ls, pcondR)
    
    t0 = perf_counter()    
    res = sco.minimize(
        rosenbrock, x0, method='L-BFGS-B',
        jac=True, callback=None, tol=1e-15)
    t1 = perf_counter()
    print(t1 - t0)
    print(res)

    t0 = perf_counter()
    x, f, exitflag, output = minimize(rosenbrock, x0, optionsR)
    t1 = perf_counter()
    print(t1 - t0)
    print(x, f)
    
