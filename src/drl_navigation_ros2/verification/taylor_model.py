#!/usr/bin/env python3
"""
Taylor Model arithmetic and POLAR reachability verification core.

Implements the framework from:
  "Reachability Verification Based Reliability Assessment for Deep Reinforcement Learning"

Key components:
  - TaylorModel: polynomial part p(z) + interval remainder I, z in [-1, 1]^n
  - TaylorArithmetic: propagation through linear layers (eq. 4)
  - BernsteinPolynomial: activation function approximation (eq. 1, 2)
  - apply_relu_optimized: three-case ReLU propagation (eq. 8)
"""

import numpy as np
import sympy as sym
from functools import reduce
import operator as op
import math


def ncr(n, r):
    """Binomial coefficient C(n, r)."""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


class TaylorModel:
    """
    Taylor Model: TM = p(z) + I, z in [-1, 1]^n.

    p(z): polynomial (sympy.Poly), analogous to the centre of a norm ball.
    I = [I^-, I^+]: interval remainder (radius).
    """
    def __init__(self, poly, remainder):
        self.poly = poly            # sympy.Poly
        self.remainder = remainder  # [lower, upper]

    def __repr__(self):
        return f"TaylorModel(poly={self.poly}, remainder={self.remainder})"


class TaylorArithmetic:
    """Taylor model arithmetic operations."""

    def weighted_sumforall(self, taylor_models, weights, bias):
        """
        Linear-layer propagation : sum_i(w_i * TM_i) + b.

        Args:
            taylor_models : list of TaylorModel
            weights       : list of scalars
            bias          : scalar

        Returns:
            TaylorModel
        """
        temp_poly = sum(weights[i] * tm.poly for i, tm in enumerate(taylor_models)) + bias

        # Conservative remainder: sum of |w_i| * max(|I^-_i|, |I^+_i|)
        temp_remainder = sum(
            abs(weights[i]) * max(abs(tm.remainder[0]), abs(tm.remainder[1]))
            for i, tm in enumerate(taylor_models)
        )
        remainder = [-temp_remainder, temp_remainder]

        if not isinstance(temp_poly, sym.Poly):
            gens = taylor_models[0].poly.gens
            temp_poly = sym.Poly(temp_poly, *gens) if gens else sym.Poly(temp_poly)

        return TaylorModel(temp_poly, remainder)

    def add(self, tm1, tm2):
        """TM1 + TM2."""
        new_poly = tm1.poly + tm2.poly
        new_remainder = [
            tm1.remainder[0] + tm2.remainder[0],
            tm1.remainder[1] + tm2.remainder[1],
        ]
        if not isinstance(new_poly, sym.Poly):
            gens = tm1.poly.gens
            new_poly = sym.Poly(new_poly, *gens) if gens else sym.Poly(new_poly)
        return TaylorModel(new_poly, new_remainder)

    def product(self, tm1, tm2):
        """
        TM1 * TM2 (simplified — high-order truncation not applied here).

        Error bound: |p1|*|I2| + |p2|*|I1| + |I1|*|I2|
        """
        if isinstance(tm2, (int, float)):
            return self.constant_product(tm1, tm2)
        if isinstance(tm1, (int, float)):
            return self.constant_product(tm2, tm1)

        new_poly = tm1.poly * tm2.poly

        poly1_bound = compute_poly_bounds(tm1.poly)
        poly2_bound = compute_poly_bounds(tm2.poly)
        r1 = max(abs(tm1.remainder[0]), abs(tm1.remainder[1]))
        r2 = max(abs(tm2.remainder[0]), abs(tm2.remainder[1]))
        p1 = max(abs(poly1_bound[0]), abs(poly1_bound[1]))
        p2 = max(abs(poly2_bound[0]), abs(poly2_bound[1]))
        error = p1 * r2 + p2 * r1 + r1 * r2

        if not isinstance(new_poly, sym.Poly):
            gens = tm1.poly.gens
            new_poly = sym.Poly(new_poly, *gens) if gens else sym.Poly(new_poly)

        return TaylorModel(new_poly, [-error, error])

    def constant_product(self, taylor_model, constant):
        """c * TM."""
        new_poly = constant * taylor_model.poly
        new_remainder = [
            constant * taylor_model.remainder[0],
            constant * taylor_model.remainder[1],
        ]
        if not isinstance(new_poly, sym.Poly):
            gens = taylor_model.poly.gens
            new_poly = sym.Poly(new_poly, *gens) if gens else sym.Poly(new_poly)
        return TaylorModel(new_poly, new_remainder)


class BernsteinPolynomial:
    """
    Bernstein polynomial approximation of activation functions
    """

    def __init__(self, error_steps=4000):
        self.error_steps = error_steps
        self.bern_poly = None
        self._poly_func = None  # cached lambdify function

    def approximate(self, a, b, order, activation_name):
        """
        Fit Bernstein polynomial p_sigma over [a, b]:
            p_sigma = sum_{v=0}^{d} sigma(a + (b-a)/d * v) * C(d,v) * B_v(y)

        Args:
            a, b            : input interval
            order           : polynomial order
            activation_name : 'relu' | 'tanh' | 'cos' | 'sin' | 'elu'

        Returns:
            sympy.Poly
        """
        d_max = 8
        interval_width = b - a

        if interval_width > 1e-8:
            d_p = np.abs(np.floor(d_max / math.log10(1 / interval_width)))
        else:
            d_p = order

        d = max(2, min(order, d_p))

        # Mapping coefficients: [a, b] -> [0, 1]
        coe1_1 = -a / (b - a)
        coe1_2 =  1 / (b - a)
        coe2_1 =  b / (b - a)
        coe2_2 = -1 / (b - a)

        x = sym.Symbol('x')
        bern_poly = 0 * x

        for v in range(int(d) + 1):
            c = ncr(int(d), v)
            point = a + (b - a) / d * v

            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            elif activation_name == 'cos':
                f_value = np.cos(float(point))
            elif activation_name == 'sin':
                f_value = np.sin(float(point))
            elif activation_name == 'elu':
                f_value = float(point) if point > 0 else np.exp(float(point)) - 1
            else:
                raise ValueError(f"Unsupported activation: {activation_name}")

            basis = (
                ((coe1_2 * x + coe1_1) ** v) *
                ((coe2_1 + coe2_2 * x) ** (d - v))
            )
            bern_poly += c * f_value * basis

        if bern_poly == 0:
            bern_poly = 1e-16 * x

        self.bern_poly = bern_poly
        self._poly_func = sym.lambdify(x, bern_poly, 'numpy')
        return sym.Poly(bern_poly, x)

    def compute_error(self, a, b, activation_name):
        """
        Estimate Bernstein approximation error bound:
            epsilon = max_i |p_sigma(s_i) - sigma(s_i)| + (b - a) / m
        """
        if self._poly_func is None:
            x = sym.Symbol('x')
            self._poly_func = sym.lambdify(x, self.bern_poly, 'numpy')

        epsilon = 0
        m = self.error_steps

        for v in range(m + 1):
            point = a + (b - a) / m * (v + 0.5)

            if activation_name == 'relu':
                f_value = max(0, point)
            elif activation_name == 'tanh':
                f_value = np.tanh(float(point))
            elif activation_name == 'cos':
                f_value = np.cos(float(point))
            elif activation_name == 'sin':
                f_value = np.sin(float(point))
            elif activation_name == 'elu':
                f_value = float(point) if point > 0 else np.exp(float(point)) - 1

            b_value = self._poly_func(point)
            epsilon = max(epsilon, abs(f_value - b_value))

        return epsilon + (b - a) / m


def compute_tm_bounds(tm):
    """
    Compute Taylor model bounds via Minkowski sum :
        [min p(z) + I^-, max p(z) + I^+],  z in [-1, 1]^n.
    """
    poly = tm.poly
    temp_upper = 0
    temp_lower = 0

    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        if sum(monom) == 0:
            temp_upper += coeff
            temp_lower += coeff
        else:
            temp_upper += abs(coeff)
            temp_lower -= abs(coeff)

    return float(temp_lower + tm.remainder[0]), float(temp_upper + tm.remainder[1])


def compute_poly_bounds(poly):
    """Interval bound of a polynomial over z in [-1, 1]^n (no remainder term)."""
    temp_upper = 0
    temp_lower = 0
    for i in range(len(poly.monoms())):
        coeff = poly.coeffs()[i]
        monom = poly.monoms()[i]
        if sum(monom) == 0:
            temp_upper += coeff
            temp_lower += coeff
        else:
            temp_upper += abs(coeff)
            temp_lower -= abs(coeff)
    return float(temp_lower), float(temp_upper)


def apply_activation(tm, bern_poly, bern_error, max_order):
    """
    Propagate a Taylor model through an activation function via polynomial composition.

    Steps:
      1. Compose: p_out = p_bern(p_in)
      2. Truncate to max_order
      3. Accumulate truncation + Bernstein approximation errors
    """
    composed = sym.polys.polytools.compose(bern_poly, tm.poly)

    # Truncate
    poly_truncated = 0
    for i in range(len(composed.monoms())):
        monom = composed.monoms()[i]
        if sum(monom) <= max_order:
            temp = 1
            for j, exp in enumerate(monom):
                temp *= composed.gens[j] ** exp
            poly_truncated += composed.coeffs()[i] * temp

    poly_truncated = (
        sym.Poly(poly_truncated, *composed.gens) if composed.gens
        else sym.Poly(poly_truncated)
    )

    _, truncation_error = compute_poly_bounds(composed - poly_truncated)

    # Propagate input remainder through Bernstein polynomial
    total_remainder = 0
    for i in range(len(bern_poly.monoms())):
        degree = sum(bern_poly.monoms()[i])
        if degree < 1:
            continue
        elif degree == 1:
            total_remainder += abs(bern_poly.coeffs()[i] * tm.remainder[1])
        else:
            r_mag = max(abs(tm.remainder[0]), abs(tm.remainder[1]))
            total_remainder += abs(bern_poly.coeffs()[i] * (r_mag ** degree))

    remainder = [
        -total_remainder - truncation_error - bern_error,
         total_remainder + truncation_error + bern_error,
    ]
    return TaylorModel(poly_truncated, remainder)


def apply_relu_optimized(tm, bern_order=1, error_steps=4000):
    """
    Three-case ReLU propagation :
      - b <= 0 : output is identically 0
      - a >= 0 : output equals input (identity)
      - a < 0 < b : Bernstein polynomial approximation

    Args:
        tm         : input TaylorModel
        bern_order : Bernstein polynomial order (used only for the crossing case)
        error_steps: sampling points for error estimation

    Returns:
        TaylorModel
    """
    a, b = compute_tm_bounds(tm)

    if a >= 0:
        return tm

    if b <= 0:
        zero_poly = sym.Poly(0, *tm.poly.gens) if tm.poly.gens else sym.Poly(0)
        return TaylorModel(zero_poly, [0.0, 0.0])

    BP = BernsteinPolynomial(error_steps=error_steps)
    bern_poly = BP.approximate(a, b, bern_order, 'relu')
    bern_error = BP.compute_error(a, b, 'relu')
    return apply_activation(tm, bern_poly, bern_error, bern_order)


def create_initial_tm(state, observation_error):
    """
    Create initial Taylor models from a state vector:
        TM_i = observation_error * z_i + state[i],  z_i in [-1, 1]
    """
    state_dim = len(state)
    z_symbols = [sym.Symbol(f'z{i}') for i in range(state_dim)]
    TM_state = []
    for i in range(state_dim):
        poly = sym.Poly(observation_error * z_symbols[i] + state[i], *z_symbols)
        TM_state.append(TaylorModel(poly, [0.0, 0.0]))
    return TM_state


def tm_to_numpy(tm_list):
    """Extract the constant term (centre) of each TaylorModel as a numpy array."""
    centers = []
    for tm in tm_list:
        if len(tm.poly.monoms()) > 0:
            for i, monom in enumerate(tm.poly.monoms()):
                if sum(monom) == 0:
                    centers.append(float(tm.poly.coeffs()[i]))
                    break
            else:
                centers.append(0.0)
        else:
            centers.append(0.0)
    return np.array(centers)


def tm_to_interval(tm):
    """Return [lower, upper] bound of a TaylorModel."""
    a, b = compute_tm_bounds(tm)
    return [a, b]
