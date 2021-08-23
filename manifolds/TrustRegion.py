"""Trust-region optimization."""
import math

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from scipy.optimize.optimize import (
    _check_unknown_options, wrap_function, _status_message,
    OptimizeResult)

__all__ = []

FD_METHODS = ('2-point', '3-point', 'cs')


class ScalarFunction(object):
    """Scalar function and its derivatives.

    This class defines a scalar function F: R^n->R and methods for
    computing or approximating its first and second derivatives.

    Parameters
    ----------
    fun : callable
        evaluates the scalar function. Must be of the form ``fun(x, *args)``,
        where ``x`` is the argument in the form of a 1-D array and ``args`` is
        a tuple of any additional fixed parameters needed to completely specify
        the function. Should return a scalar.
    x0 : array-like
        Provides an initial set of variables for evaluating fun. Array of real
        elements of size (n,), where 'n' is the number of independent
        variables.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the scalar
        function.
    grad : {callable, '2-point', '3-point', 'cs'}
        Method for computing the gradient vector.
        If it is a callable, it should be a function that returns the gradient
        vector:

            ``grad(x, *args) -> array_like, shape (n,)``

        where ``x`` is an array with shape (n,) and ``args`` is a tuple with
        the fixed parameters.
        Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
        to select a finite difference scheme for numerical estimation of the
        gradient with a relative step size. These finite difference schemes
        obey any specified `bounds`.
    hess : {callable, '2-point', '3-point', 'cs'}
        Method for computing the Hessian matrix. If it is callable, it should
        return the  Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        where x is a (n,) ndarray and `args` is a tuple with the fixed
        parameters. Alternatively, the keywords {'2-point', '3-point', 'cs'}
        select a finite difference scheme for numerical estimation. Or, objects
        implementing.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.
    finite_diff_rel_step : None or array_like
        Relative step size to use. The absolute step size is computed as
        ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``, possibly
        adjusted to fit into the bounds. For ``method='3-point'`` the sign
        of `h` is ignored. If None then finite_diff_rel_step is selected
        automatically,
    finite_diff_bounds : tuple of array_like
        Lower and upper bounds on independent variables. Defaults to no bounds,
        (-np.inf, np.inf). Each bound must match the size of `x0` or be a
        scalar, in the latter case the bound will be the same for all
        variables. Use it to limit the range of function evaluation.
    epsilon : None or array_like, optional
        Absolute step size to use, possibly adjusted to fit into the bounds.
        For ``method='3-point'`` the sign of `epsilon` is ignored. By default
        relative steps are used, only if ``epsilon is not None`` are absolute
        steps used.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `grad`, hess` and corresponding attributes `f`, `g` and `H`. The following
    things should be considered:

        1. Use only public methods `fun`, `grad` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """
    def __init__(self, fun, x0, args, grad, hess, finite_diff_rel_step,
                 finite_diff_bounds, epsilon=None):
        if not callable(grad) and grad not in FD_METHODS:
            raise ValueError(
                f"`grad` must be either callable or one of {FD_METHODS}."
            )

        if not (callable(hess) or hess in FD_METHODS):
            raise ValueError(
                f"`hess` must be either callable"
                f" or one of {FD_METHODS}."
            )

        if grad in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the gradient is estimated via "
                             "finite-differences, we require the Hessian "
                             "to be estimated using one of the "
                             "quasi-Newton strategies.")

        self.x = x0
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False

        finite_diff_options = {}
        if grad in FD_METHODS:
            finite_diff_options["method"] = grad
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["bounds"] = finite_diff_bounds
        if hess in FD_METHODS:
            finite_diff_options["method"] = hess
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["abs_step"] = epsilon
            finite_diff_options["as_linear_operator"] = True

        # Function evaluation
        def fun_wrapped(x):
            self.nfev += 1
            return fun(x, *args)

        def update_fun():
            self.f = fun_wrapped(self.x)

        self._update_fun_impl = update_fun
        self._update_fun()

        # Gradient evaluation
        if callable(grad):
            def grad_wrapped(x):
                self.ngev += 1
                return grad(x, *args)

            def update_grad():
                self.g = grad_wrapped(self.x)

        elif grad in FD_METHODS:
            def update_grad():
                self._update_fun()
                self.ngev += 1
                self.g = approx_derivative(fun_wrapped, self.x, f0=self.f,
                                           **finite_diff_options)

        self._update_grad_impl = update_grad
        self._update_grad()

        # Hessian Evaluation
        if callable(hess):
            self.H = hess(x0, *args)
            self.H_updated = True
            self.nhev += 1

            if isinstance(self.H, LinearOperator):
                def hess_wrapped(x):
                    self.nhev += 1
                    return hess(x, *args)

            else:
                def hess_wrapped(x):
                    self.nhev += 1
                    return hess(x, *args)
                # self.H = np.atleast_2d(np.asarray(self.H))

            def update_hess():
                self.H = hess_wrapped(self.x)

        elif hess in FD_METHODS:
            def update_hess():
                self._update_grad()
                self.H = approx_derivative(grad_wrapped, self.x, f0=self.g,
                                           **finite_diff_options)
                return self.H

            update_hess()
            self.H_updated = True

        self._update_hess_impl = update_hess

        def update_x(x):
            self.x = x
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False
        self._update_x_impl = update_x

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_grad(self):
        if not self.g_updated:
            self._update_grad_impl()
            self.g_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    def fun(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_fun()
        return self.f

    def grad(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_grad()
        return self.g

    def hess(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_hess()
        return self.H

    def fun_and_grad(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        self._update_fun()
        self._update_grad()
        return self.f, self.g


def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """
    Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    hess : {callable,  '2-point', '3-point', 'cs', None}
        Computes the Hessian matrix. If it is callable, it should return the
        Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
        finite difference scheme for numerical estimation.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.

    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # ScalarFunction requires something for hess, so we give a dummy
        # implementation here if nothing is provided, return a value of None
        # so that downstream minimisers halt. The results of `fun.hess`
        # should not be used.
        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    sf = ScalarFunction(fun, x0, args, grad, hess,
                        finite_diff_rel_step, bounds, epsilon=epsilon)

    return sf


class BaseQuadraticSubproblem(object):
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method.

    Values of the objective function, Jacobian and Hessian (if provided) at
    the current iterate ``x`` are evaluated on demand and then stored as
    attributes ``fun``, ``jac``, ``hess``.
    """

    def __init__(self, x, fun, jac, hess=None, hessp=None):
        self._x = x
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp

    def __call__(self, p):
        return self.fun + self.jac.dot_prod(p) + 0.5 * p.dot_prod(self.hessp(p))

    @property
    def fun(self):
        """Value of objective function at current iteration."""
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        """Value of Jacobian of objective function at current iteration."""
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        """Value of Hessian of objective function at current iteration."""
        if self._h is None:
            self._h = self._hess(self._x)
        return self._h

    def hessp(self, p):
        if self._hessp is not None:
            return self._hessp(self._x, p)
        else:
            return self.hess.dot_prod(p)

    @property
    def jac_mag(self):
        """Magnitude of jacobian of objective function at current iteration."""
        if self._g_mag is None:
            self._g_mag = self.jac.norm
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        """
        Solve the scalar quadratic equation ||z + t d|| == trust_radius.
        This is like a line-sphere intersection.
        Return the two values of t, sorted from low to high.
        """
        a = d.dot_prod(d)
        b = 2 * z.dot_prod(d)
        c = z.dot_prod(z) - trust_radius**2
        sqrt_discriminant = math.sqrt(b*b - 4*a*c)

        # The following calculation is mathematically
        # equivalent to:
        # ta = (-b - sqrt_discriminant) / (2*a)
        # tb = (-b + sqrt_discriminant) / (2*a)
        # but produce smaller round off errors.
        # Look at Matrix Computation p.97
        # for a better justification.
        aux = b + math.copysign(sqrt_discriminant, b)
        ta = -aux / (2*a)
        tb = -2*c / aux
        return sorted([ta, tb])

    def solve(self, trust_radius):
        raise NotImplementedError('The solve method should be implemented by '
                                  'the child class')


def _minimize_trust_region(fun, x0, args=(), jac=None, hess=None, hessp=None,
                           subproblem=None, initial_trust_radius=1.0,
                           max_trust_radius=1000.0, eta=0.15, gtol=1e-4,
                           maxiter=None, disp=False, return_all=False,
                           callback=None, inexact=True, **unknown_options):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.
        inexact : bool
            Accuracy to solve subproblems. If True requires less nonlinear
            iterations, but more vector products. Only effective for method
            trust-krylov.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """
    _check_unknown_options(unknown_options)

    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region '
                         'methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is currently required for trust-region methods')
    if subproblem is None:
        raise ValueError('A subproblem solving strategy is required for '
                         'trust-region methods')
    if not (0 <= eta < 0.25):
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the '
                         'max trust radius')

    # force the initial guess into a nice format
    # x0 = np.asarray(x0).flatten()

    # A ScalarFunction representing the problem. This caches calls to fun, jac,
    # hess.
    sf = _prepare_scalar_function(fun, x0, jac=jac, hess=hess, args=args)
    fun = sf.fun
    jac = sf.grad
    if hess is not None:
        hess = sf.hess
    # ScalarFunction doesn't represent hessp
    nhessp, hessp = wrap_function(hessp, args)

    # limit the number of iterations
    if maxiter is None:
        maxiter = x0.dim*200

    # init the search status
    warnflag = 0

    # initialize the search
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0

    # search for the function min
    # do not even start if the gradient is small enough
    while m.jac_mag >= gtol:

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            p, hits_boundary = m.solve(trust_radius)
        except np.linalg.linalg.LinAlgError:
            warnflag = 3
            break

        # calculate the predicted value at the proposed point
        predicted_value = m(p)

        # define the local approximation at the proposed point
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)

        # evaluate the ratio defined in equation (4.4)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2*trust_radius, max_trust_radius)

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed

        # append the best guess, call back, increment the iteration count
        if return_all:
            allvecs.append(np.copy(x))
        if callback is not None:
            callback(np.copy(x))
        k += 1

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            warnflag = 0
            break

        # check if we have looked at enough iterations
        if k >= maxiter:
            warnflag = 1
            break

    # print some stuff if requested
    status_messages = (
            _status_message['success'],
            _status_message['maxiter'],
            'A bad approximation caused failure to predict improvement.',
            'A linalg error occurred, such as a non-psd Hessian.',
            )
    if disp:
        if warnflag == 0:
            print(status_messages[warnflag])
        else:
            print('Warning: ' + status_messages[warnflag])
        print("         Current function value: %f" % m.fun)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)
        print("         Hessian evaluations: %d" % (sf.nhev + nhessp[0]))

    result = OptimizeResult(x=x, success=(warnflag == 0), status=warnflag,
                            fun=m.fun, jac=m.jac, nfev=sf.nfev, njev=sf.ngev,
                            nhev=sf.nhev + nhessp[0], nit=k,
                            message=status_messages[warnflag])

    if hess is not None:
        result['hess'] = m.hess

    if return_all:
        result['allvecs'] = allvecs

    return result
