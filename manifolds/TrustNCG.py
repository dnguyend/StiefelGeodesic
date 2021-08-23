"""Newton-CG trust-region optimization."""
import math

import numpy as np
import scipy.linalg
from .TrustRegion import (_minimize_trust_region, BaseQuadraticSubproblem)

__all__ = []


def _minimize_trust_ncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                        **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    the Newton conjugate gradient trust-region algorithm.

    Options
    -------
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.

    """
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG trust-region '
                         'minimization')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Newton-CG trust-region minimization')
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  hessp=hessp, subproblem=CGSteihaugSubproblem,
                                  **trust_region_options)


class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""
    def solve(self, trust_radius):
        """
        Solve the subproblem using a conjugate gradient method.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : ndarray
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        Notes
        -----
        This is algorithm (7.2) of Nocedal and Wright 2nd edition.
        Only the function that computes the Hessian-vector product is required.
        The Hessian itself is not required, and the Hessian does
        not need to be positive semidefinite.
        """

        # get the norm of jacobian and define the origin
        p_origin = self.jac.zeros_like()

        # define a default tolerance
        tolerance = min(0.5, math.sqrt(self.jac_mag)) * self.jac_mag

        # Stop the method if the search direction
        # is a direction of nonpositive curvature.
        if self.jac_mag < tolerance:
            hits_boundary = False
            return p_origin, hits_boundary

        # init the state for the first iteration
        z = p_origin
        r = self.jac
        d = -r

        # Search for the min of the approximation of the objective function.
        while True:

            # do an iteration
            Bd = self.hessp(d)
            dBd = d.dot_prod(Bd)
            if dBd <= 0:
                # Look at the two boundary points.
                # Find both values of t to get the boundary points such that
                # ||z + t d|| == trust_radius
                # and then choose the one with the predicted min value.
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                pa = z + ta * d
                pb = z + tb * d
                if self(pa) < self(pb):
                    p_boundary = pa
                else:
                    p_boundary = pb
                hits_boundary = True
                return p_boundary, hits_boundary
            r_squared = r.dot_prod(r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if z_next.norm >= trust_radius:
                # Find t >= 0 to get the boundary point such that
                # ||z + t d|| == trust_radius
                ta, tb = self.get_boundaries_intersections(z, d, trust_radius)
                p_boundary = z + tb * d
                hits_boundary = True
                return p_boundary, hits_boundary
            r_next = r + alpha * Bd
            r_next_squared = r_next.dot_prod(r_next)
            if math.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return z_next, hits_boundary
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            # update the state for the next iteration
            z = z_next
            r = r_next
            d = d_next


if False:
    class AR(object):
        def __init__(self, A, R):
            self.A = A
            self.R = R

        @property
        def norm(self):
            return np.sqrt(self.dot_prod(self))

        def zerovec(self):
            return AR(np.zeros_like(self.A),
               np.zeros_like(self.R))
            
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
        
    def fun(v):
        return np.sum(v*v)

    def jac(v):
        return 2*v

    def hessp(v, xi):
        return 2*xi

    def funAR(v):
        return np.sum((v.A-S)*(v.A-S)) +\
            np.sum((v.R-R)*(v.R-R))

    def jacAR(v):
        return 2*(v - AR(S, R))

    def hesspAR(v, xi):
        return 2*xi
    
    x0 = np.random.randn(10)
    
    # ret = _minimize_trust_ncg(fun, x0, jac=jac, hessp=hessp)

    d = 3
    k = 2
    S = np.arange(d*d).reshape(d, d)
    S = S - S.T
    R = np.arange(d*k).reshape(k, d)
    
        
    x0 = AR.rand(d, k)
    ret = _minimize_trust_ncg(funAR, x0, jac=jacAR, hessp=hesspAR)
    print(ret)
    
