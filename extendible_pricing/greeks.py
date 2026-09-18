"""Extendible option의 몬테카를로 그릭스(Greeks) 계산.

## 왜 유한차분(finite difference) + 공통난수법(CRN)인가

이 상품은 T1에서 "즉시행사 vs 연장 vs 포기"의 분기(kink)가 K1에 존재하므로
그릭스에 대한 깔끔한 폐형식이 없다 (분기 함수의 미분이 K1에서 불연속).
따라서 그릭스는 파라미터를 살짝 흔들어 다시 가격을 매기는 유한차분법으로
구한다.

문제는 감마/바나/보마처럼 2차 차분으로 얻는 그릭스는 몬테카를로 표준오차가
차분 스텝 크기의 제곱에 반비례하여 커지기 때문에(2차 차분 -> 노이즈 증폭),
매번 새로운 난수로 가격을 다시 매기면 추정치가 완전히 노이즈에 묻힌다.
이를 해결하기 위해 **공통난수법(Common Random Numbers, CRN)**을 쓴다:
파라미터를 bump-up/bump-down 할 때 *똑같은* 표준정규 난수 Z를 재사용해서
경로별로 짝을 지어 차분한다. 이러면 몬테카를로 노이즈의 상당 부분이
차분 과정에서 서로 상쇄되어, 실제 곡률(=상품 고유의 non-linearity)만
남는다. 이 그릭스들이 정말로 K1 근방에서 커지는 것을 보여주는 것이
Taleb의 "고차 그릭스가 지배적" 논지를 재현하는 핵심이다.

## 차분 스텝 크기(dS, dvol)를 너무 작게 잡으면 안 되는 이유

이 상품의 payoff는 S1=K1에서 꺾이는 kink를 갖는다. 파라미터를 살짝
움직이면 이 kink의 위치도 같이 움직이므로, 일부 시뮬레이션 경로는
"bump-up에서는 즉시행사 분기, bump-down에서는 연장 분기"처럼 분기 자체가
바뀌어버린다. 이런 경로는 CRN을 쓰더라도 완전히 상쇄되지 않고 2차 차분에
그대로 노이즈로 남는데, 차분 스텝(dS, dvol)이 작을수록 분모(dvol^2 등)가
작아 이 노이즈가 크게 증폭된다. 그래서 이 모듈은 (표준적인 "아주 작은"
bump 대신) 다소 넉넉한 기본 스텝(dS_rel=2%, dvol=2%p)과 충분히 큰 경로 수
(기본 2,000,000)를 사용해, "잘림 오차(bias)"와 "몬테카를로 분산 증폭"
사이의 균형을 맞춘다. 실제로 시도해보면 dvol=1e-3처럼 아주 작은 스텝을
쓸 경우 vomma 추정치가 부호까지 흔들릴 정도로 불안정해지는 것을 확인할 수
있다.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .extendible_option import ExtendibleParams, extendible_payoff_at_T1, _simulate_S1


def _reprice_with_Z(p: ExtendibleParams, Z: np.ndarray) -> float:
    """주어진 (고정된) 난수 Z로 forward-analytic MC 가격을 계산.

    Z를 고정하고 p(파라미터)만 바꿔가며 이 함수를 호출하는 것이 바로 CRN이다.
    """
    S1 = _simulate_S1(p, Z)
    payoff = extendible_payoff_at_T1(S1, p)
    return float(np.mean(np.exp(-p.r * p.T1) * payoff))


def mc_greeks(
    p: ExtendibleParams,
    n_paths: int = 2_000_000,
    seed: int = 0,
    dS_rel: float = 2e-2,
    dvol: float = 2e-2,
    dT: float = 1.0 / 365.0,
) -> dict:
    """중심 유한차분(central finite difference) + CRN으로 그릭스 일체를 계산.

    반환하는 그릭스:
      price  : 기준 가격 (CRN에 쓰인 Z에서의 forward-analytic MC 추정치)
      delta  : dPrice/dS               (1차, 방향성 리스크)
      gamma  : d^2Price/dS^2           (2차, 스팟에 대한 곡률)
      vega   : dPrice/dsigma           (1차, 변동성 리스크)
      vomma  : d^2Price/dsigma^2       (2차, 변동성의 변동성 리스크 = volga)
      vanna  : d^2Price/(dS dsigma)    (2차 교차항, 스팟-변동성 상호작용)
      theta  : -dPrice/dt (시간이 흐를 때, 즉 T1이 줄어들 때의 가치변화)
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    z_half = rng.standard_normal(half)
    Z = np.concatenate([z_half, -z_half])  # antithetic + 이후 모든 bump에서 재사용(CRN)

    dS = p.S0 * dS_rel

    P0 = _reprice_with_Z(p, Z)

    # --- Delta, Gamma: 스팟(S0)을 +-dS 흔든다 ---
    p_up = replace(p, S0=p.S0 + dS)
    p_dn = replace(p, S0=p.S0 - dS)
    P_Su = _reprice_with_Z(p_up, Z)
    P_Sd = _reprice_with_Z(p_dn, Z)
    delta = (P_Su - P_Sd) / (2 * dS)
    gamma = (P_Su - 2 * P0 + P_Sd) / (dS ** 2)

    # --- Vega, Vomma: 변동성(sigma)을 +-dvol 흔든다 ---
    p_vu = replace(p, sigma=p.sigma + dvol)
    p_vd = replace(p, sigma=p.sigma - dvol)
    P_vu = _reprice_with_Z(p_vu, Z)
    P_vd = _reprice_with_Z(p_vd, Z)
    vega = (P_vu - P_vd) / (2 * dvol)
    vomma = (P_vu - 2 * P0 + P_vd) / (dvol ** 2)

    # --- Vanna: 스팟과 변동성을 동시에 흔드는 교차 2차 차분 ---
    p_uu = replace(p, S0=p.S0 + dS, sigma=p.sigma + dvol)
    p_ud = replace(p, S0=p.S0 + dS, sigma=p.sigma - dvol)
    p_du = replace(p, S0=p.S0 - dS, sigma=p.sigma + dvol)
    p_dd = replace(p, S0=p.S0 - dS, sigma=p.sigma - dvol)
    P_uu = _reprice_with_Z(p_uu, Z)
    P_ud = _reprice_with_Z(p_ud, Z)
    P_du = _reprice_with_Z(p_du, Z)
    P_dd = _reprice_with_Z(p_dd, Z)
    vanna = (P_uu - P_ud - P_du + P_dd) / (4 * dS * dvol)

    # --- Theta: T1과 T2를 동시에 dT만큼 줄인다 (두 만기 사이 간격 T2-T1은 고정 유지) ---
    # 캘린더 상에서 "오늘"이 dT만큼 앞으로 흐르면 두 만기까지 남은 시간이
    # 똑같이 dT씩 줄어들 뿐, 두 만기 사이의 간격(T2-T1)은 변하지 않는다.
    p_theta = replace(p, T1=max(p.T1 - dT, 1e-6), T2=max(p.T2 - dT, 1e-6 + (p.T2 - p.T1)))
    P_theta = _reprice_with_Z(p_theta, Z)
    theta = (P_theta - P0) / (-dT)  # 관례: 시간이 흐를수록(dt>0) 가치가 줄면 theta<0

    return {
        "price": P0,
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "vomma": float(vomma),
        "vanna": float(vanna),
        "theta": float(theta),
        "Z": Z,  # 재사용/재현용 (예: Taylor 근사 검증 시 같은 Z로 실제 reprice와 비교)
    }
