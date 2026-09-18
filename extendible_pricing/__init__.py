"""Extendible (holder-extendible) call option 가격결정 & 그릭스 계산 패키지.

자세한 상품 설명은 docs/extendible_option.md, 방법론(Forward vs Backward
Monte Carlo) 설명은 extendible_option.py의 모듈 docstring을 참고.
"""
from .black_scholes import bs_greeks, bs_price
from .extendible_option import (
    ExtendibleParams,
    extend_abandon_boundary,
    extendible_payoff_at_T1,
    price_forward_analytic_mc,
    price_lognormal_quad,
    price_nested_backward_mc,
)
from .greeks import mc_greeks

__all__ = [
    "bs_price",
    "bs_greeks",
    "ExtendibleParams",
    "extendible_payoff_at_T1",
    "extend_abandon_boundary",
    "price_forward_analytic_mc",
    "price_nested_backward_mc",
    "price_lognormal_quad",
    "mc_greeks",
]
