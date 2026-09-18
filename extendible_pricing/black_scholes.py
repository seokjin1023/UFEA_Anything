"""블랙-숄즈(Black-Scholes) 유러피언 옵션 공식 모음.

이 모듈은 extendible option 가격결정의 두 군데에서 "빌딩 블록"으로 쓰인다.

1. `T1` 시점에서 옵션을 연장할지 말지를 결정하려면, 연장 시 얻게 되는
   "잔존만기 (T2-T1)짜리 새 콜옵션"의 그 시점 가치(=continuation value)가
   필요하다. 기초자산이 기하브라운운동(GBM)을 따르는 한 이 값은
   블랙-숄즈 폐형식으로 정확히 알려져 있다. 그래서 `extendible_option.py`의
   forward Monte Carlo는 `T1`까지만 시뮬레이션하고, 그 노드에서 이 모듈의
   `bs_price`를 그대로 대입해 사용한다 (= "backward로 미리 풀어놓은 해"를
   forward 시뮬레이션 경로 위에 얹는 방식). 이렇게 하면 T1->T2 구간을 다시
   시뮬레이션(nested MC)하거나 회귀분석(Longstaff-Schwartz)을 할 필요가 없다.
2. 몬테카를로 가격/그릭스 계산 결과를 검증하기 위한 벤치마크(바닐라 콜/풋)로
   사용한다.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S: np.ndarray, K: float, r: float, q: float, sigma: float, tau: np.ndarray):
    """블랙-숄즈 d1, d2를 계산한다.

    S, tau는 배열이어도 되도록 numpy 브로드캐스팅을 그대로 활용한다
    (몬테카를로에서 수십만 개의 시뮬레이션된 S1에 한 번에 벡터 연산으로
    적용하기 위함 - 파이썬 for-loop보다 수백 배 빠르다).
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    # tau<=0 이면 옵션이 이미 만기 -> log/sqrt에서 0-division 방지를 위해
    # 아주 작은 값으로 클리핑하고, 최종 가격은 payoff 함수에서 별도 처리.
    safe_tau = np.where(tau > 1e-12, tau, 1e-12)
    sqrt_tau = np.sqrt(safe_tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * safe_tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2


def bs_price(S, K: float, r: float, q: float, sigma: float, tau, is_call: bool = True):
    """유러피언 옵션의 블랙-숄즈 현재가치 (배당수익률 q 포함, Merton 1973 확장형).

    tau<=0인 원소는 만기가 이미 도래한 것으로 보고 내재가치(intrinsic value)를
    그대로 반환한다 (continuation value를 T1=T2 극한, 즉 tau2=0 근처에서
    호출해도 값이 깨지지 않도록 하기 위한 경계 처리).
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, d2 = _d1_d2(S, K, r, q, sigma, tau)
    if is_call:
        price = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)
    return np.where(tau > 1e-12, price, intrinsic)


def bs_greeks(S, K: float, r: float, q: float, sigma: float, tau, is_call: bool = True):
    """바닐라 옵션의 해석적 그릭스 (검증용 벤치마크로만 사용).

    반환: delta, gamma, vega, theta(연 단위), rho, vanna(=d Delta/d sigma),
    vomma(=d Vega/d sigma, 'volga'라고도 부름).
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, d2 = _d1_d2(S, K, r, q, sigma, tau)
    sqrt_tau = np.sqrt(np.where(tau > 1e-12, tau, 1e-12))
    pdf_d1 = norm.pdf(d1)

    if is_call:
        delta = np.exp(-q * tau) * norm.cdf(d1)
        rho = K * tau * np.exp(-r * tau) * norm.cdf(d2)
        theta = (
            -S * np.exp(-q * tau) * pdf_d1 * sigma / (2 * sqrt_tau)
            - r * K * np.exp(-r * tau) * norm.cdf(d2)
            + q * S * np.exp(-q * tau) * norm.cdf(d1)
        )
    else:
        delta = -np.exp(-q * tau) * norm.cdf(-d1)
        rho = -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
        theta = (
            -S * np.exp(-q * tau) * pdf_d1 * sigma / (2 * sqrt_tau)
            + r * K * np.exp(-r * tau) * norm.cdf(-d2)
            - q * S * np.exp(-q * tau) * norm.cdf(-d1)
        )

    gamma = np.exp(-q * tau) * pdf_d1 / (S * sigma * sqrt_tau)
    vega = S * np.exp(-q * tau) * pdf_d1 * sqrt_tau
    vanna = -np.exp(-q * tau) * pdf_d1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vanna": vanna,
        "vomma": vomma,
    }
