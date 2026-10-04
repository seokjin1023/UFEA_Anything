"""SABR 확률변동성 모델: Hagan 점근 내재변동성 공식과 슬라이스(만기 하나) 캘리브레이션.

## 모델

    dF_t     = alpha_t * F_t^beta * dW1_t        (F_t: 선도가격)
    d alpha_t = nu * alpha_t * dW2_t              (alpha_0 = alpha)
    d<W1, W2>_t = rho dt

    alpha : 초기 변동성 수준 (ATM 변동성의 크기를 결정)
    beta  : 백본(backbone) 지수. 0이면 정규(Normal), 1이면 로그정규(Lognormal) 모형.
            관측 불가능한 값이라 캘리브레이션하지 않고 사전에 고정하는 것이 관행이다
            (Hagan et al. 2002; West 2005). 주식에서는 beta = 1이 흔한 선택이다.
    rho   : 선도가격과 변동성의 상관계수 (주식은 보통 음수 -> 하방 스큐)
    nu    : 변동성의 변동성 (vol-of-vol, 스마일의 곡률을 결정)

만기 T, 행사가 K, 선도가격 F에 대한 블랙(로그정규) 내재변동성의 근사해가 Hagan 공식이며,
이 모듈은 그 공식(`sabr_implied_vol`)과 시장 스마일에 맞추는 최소제곱 보정(`calibrate_sabr_slice`)을
구현한다.

## 참고문헌 (이 모듈이 구현의 근거로 삼은 논문 / 교과서)

[1] Hagan, P. S., Kumar, D., Lesniewski, A. S., Woodward, D. E. (2002),
    "Managing Smile Risk", Wilmott Magazine, September 2002, pp. 84-108.
    -> SABR 모델의 원전. 내재변동성 근사식 (eq. 2.17a-c) 과 ATM 극한식 (eq. 2.18)이 이 파일의
       `_sabr_vol_hagan`에 그대로 구현되어 있다.
[2] Oblój, J. (2008), "Fine-tune your smile: Correction to Hagan et al.", Wilmott Magazine
    (arXiv:0708.0998).
    -> beta < 1 일 때 Hagan 식의 z 정의에 오류가 있음을 지적하고 수정식을 제안.
       `_sabr_vol_obloj`로 구현했고, beta = 1 에서는 [1]과 정확히 일치한다.
[3] West, G. (2005), "Calibration of the SABR Model in Illiquid Markets",
    Applied Mathematical Finance 12(4), pp. 371-385.
    -> beta를 사전에 고정하고 (alpha, rho, nu)만 보정하는 절차, 다중 초기값 최적화 근거.
[4] Hagan, P. S., Kumar, D., Lesniewski, A. S., Woodward, D. E. (2014),
    "Arbitrage Free SABR", Wilmott Magazine, January 2014.
    -> 근사식은 저행사가/장기만기에서 음의 확률밀도(차익거래 가능성)를 만든다는 한계.
       이 프로젝트는 dupire.py에서 이를 진단하고 국소변동성을 클리핑하는 방식으로 대응한다.
[5] Gatheral, J. (2006), The Volatility Surface: A Practitioner's Guide, Wiley.
    -> 스마일 모형화의 일반 배경 (SABR 장 포함).
[6] Rebonato, R., McKay, K., White, R. (2009), The SABR/LIBOR Market Model, Wiley.
    -> SABR 파라미터의 해석과 beta 선택에 대한 교과서적 설명.
"""
from __future__ import annotations  # 타입 힌트를 문자열로 취급해 순환/전방참조 문제를 피한다

from dataclasses import dataclass  # 파라미터/결과를 담는 불변 데이터 클래스
from typing import Literal  # correction 인자의 허용값을 명시하기 위한 타입

import numpy as np  # 벡터화 수치 계산
from scipy.optimize import least_squares  # 경계 조건이 있는 비선형 최소제곱


@dataclass(frozen=True)  # frozen: 한 번 만들면 값이 바뀌지 않도록 (해시/공유 안전)
class SABRParams:
    """한 만기 슬라이스의 SABR 파라미터 묶음."""

    alpha: float  # 초기 변동성 수준
    beta: float  # 백본 지수 (고정값)
    rho: float  # 선도가격-변동성 상관
    nu: float  # vol-of-vol


@dataclass(frozen=True)
class SABRSliceFit:
    """한 만기 슬라이스의 캘리브레이션 결과."""

    expiry: float  # 만기 T (연 단위)
    forward: float  # 해당 만기의 선도가격 F_T
    params: SABRParams  # 보정된 SABR 파라미터
    rmse: float  # 시장 변동성 대비 RMSE (변동성 단위, 0.001 = 10bp)
    success: bool  # 최적화기가 수렴했는지 여부


def _x_of_z(z: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Hagan 식의 x(z) = ln{ [sqrt(1 - 2 rho z + z^2) + z - rho] / (1 - rho) } 를 계산한다."""
    root = np.sqrt(1.0 - 2.0 * rho * z + z * z)  # 제곱근 항 (|rho|<1 이면 항상 양수)
    return np.log((root + z - rho) / (1.0 - rho))  # 로그 항: z=0 에서 0 이 된다


def _z_over_x(z: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """z / x(z) 를 계산한다. z -> 0 에서는 0/0 이므로 극한 1 - rho*z/2 로 대체한다."""
    small = np.abs(z) < 1e-7  # 극한식을 쓸 만큼 z가 충분히 작은 위치
    z_safe = np.where(small, 1.0, z)  # 0 나눗셈 경고를 피하기 위해 임시 대입
    x_safe = _x_of_z(z_safe, rho)  # 안전한 z에서 x(z) 계산
    ratio = z_safe / x_safe  # 일반 영역의 z/x(z)
    limit = 1.0 - 0.5 * rho * z  # z->0 테일러 전개 (x ~ z + rho z^2 / 2 로부터)
    return np.where(small, limit, ratio)  # 작은 z에서는 극한식, 그 외에는 정확식


def _time_correction(
    F: np.ndarray, K: np.ndarray, T: np.ndarray,
    alpha: np.ndarray, beta: float, rho: np.ndarray, nu: np.ndarray,
) -> np.ndarray:
    """공식 끝의 {1 + [...] T} 보정 항 (Hagan 2002, eq. 2.17a). [1], [2] 모두 동일하다."""
    one_b = 1.0 - beta  # (1 - beta) 자주 쓰이므로 변수화
    fk = F * K  # F*K 의 곱
    term1 = one_b ** 2 / 24.0 * alpha ** 2 / fk ** one_b  # (1-b)^2 alpha^2 / (24 (FK)^(1-b))
    term2 = 0.25 * rho * beta * nu * alpha / fk ** (one_b / 2.0)  # rho b nu alpha / (4 (FK)^((1-b)/2))
    term3 = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2  # (2 - 3 rho^2) nu^2 / 24
    return 1.0 + (term1 + term2 + term3) * T  # 시간에 비례하는 2차 보정


def _sabr_vol_hagan(F, K, T, alpha, beta, rho, nu) -> np.ndarray:
    """Hagan et al. (2002) eq. (2.17a) 내재변동성. F=K(ATM)에서는 eq. (2.18)로 연속 연결된다."""
    one_b = 1.0 - beta  # (1 - beta)
    log_fk = np.log(F / K)  # ln(F/K): 로그 선도 머니니스
    fk_mid = (F * K) ** (one_b / 2.0)  # (FK)^((1-beta)/2)
    z = nu / alpha * fk_mid * log_fk  # z = (nu/alpha) (FK)^((1-b)/2) ln(F/K)
    # 분모의 로그 머니니스 급수 보정: 1 + (1-b)^2/24 ln^2 + (1-b)^4/1920 ln^4
    denom = fk_mid * (1.0 + one_b ** 2 / 24.0 * log_fk ** 2 + one_b ** 4 / 1920.0 * log_fk ** 4)
    corr = _time_correction(F, K, T, alpha, beta, rho, nu)  # 시간 보정 항
    return alpha / denom * _z_over_x(z, rho) * corr  # 세 인자의 곱이 최종 내재변동성


def _sabr_vol_obloj(F, K, T, alpha, beta, rho, nu) -> np.ndarray:
    """Oblój (2008) 수정식. z를 F^(1-b) - K^(1-b) 로 정의해 beta < 1 에서의 오차를 줄인다."""
    one_b = 1.0 - beta  # (1 - beta)
    log_fk = np.log(F / K)  # ln(F/K)
    if one_b < 1e-12:  # beta = 1: F^(1-b) - K^(1-b) 가 (1-b)로 나뉘면 ln(F/K) 로 수렴
        zeta = nu / alpha * log_fk  # 이 경우 Hagan 식과 동일한 z
    else:  # 일반 beta
        zeta = nu * (F ** one_b - K ** one_b) / (alpha * one_b)  # Oblój의 zeta
    corr = _time_correction(F, K, T, alpha, beta, rho, nu)  # 시간 보정 항 ([1]과 동일)
    small = np.abs(log_fk) < 1e-8  # ATM: nu*ln(F/K)/x(zeta) 가 0/0 이 되는 지점
    zeta_safe = np.where(small, 1.0, zeta)  # 임시값으로 나눗셈 경고 방지
    log_safe = np.where(small, 1.0, log_fk)  # 마찬가지로 임시값
    generic = nu * log_safe / _x_of_z(zeta_safe, rho) * corr  # 일반 영역: nu ln(F/K) / x(zeta) * corr
    atm = alpha / F ** one_b * corr  # ATM 극한: alpha / F^(1-b) * corr
    return np.where(small, atm, generic)  # ATM 근방에서는 극한식 사용


def sabr_implied_vol(
    F, K, T, alpha, beta, rho, nu,
    correction: Literal["hagan", "obloj"] = "hagan",
) -> np.ndarray:
    """SABR 블랙 내재변동성. 모든 인자는 numpy 브로드캐스팅이 가능하다.

    F: 선도가격, K: 행사가, T: 만기(연). correction: 'hagan'(원전 [1]) 또는 'obloj'(수정 [2]).
    """
    F, K, T, alpha, rho, nu = (np.asarray(v, dtype=float) for v in (F, K, T, alpha, rho, nu))  # 배열화
    if correction == "hagan":  # 원전 공식 선택
        return _sabr_vol_hagan(F, K, T, alpha, beta, rho, nu)  # [1] eq. (2.17a)
    if correction == "obloj":  # 수정 공식 선택
        return _sabr_vol_obloj(F, K, T, alpha, beta, rho, nu)  # [2] 수정식
    raise ValueError(f"unknown correction: {correction}")  # 잘못된 옵션 방어


def calibrate_sabr_slice(
    forward: float,
    expiry: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    beta: float = 1.0,
    correction: Literal["hagan", "obloj"] = "hagan",
    weights: np.ndarray | None = None,
) -> SABRSliceFit:
    """한 만기의 시장 스마일에 (alpha, rho, nu)를 최소제곱으로 맞춘다 (beta는 고정, [3]).

    목적함수는 sum_i w_i (sigma_SABR(K_i) - sigma_mkt(K_i))^2 이며, 국소해를 피하려고
    여러 초기값에서 시작해 가장 비용이 낮은 해를 고른다.
    """
    strikes = np.asarray(strikes, dtype=float)  # 행사가 배열
    market_vols = np.asarray(market_vols, dtype=float)  # 시장 내재변동성 배열
    w = np.ones_like(market_vols) if weights is None else np.asarray(weights, dtype=float)  # 가중치 (기본 동일)
    atm_vol = float(np.interp(forward, strikes, market_vols))  # 선도가격 부근의 ATM 변동성 (선형보간)

    def residuals(p: np.ndarray) -> np.ndarray:  # 최적화기가 최소화할 잔차 벡터
        a, r, n = p  # alpha, rho, nu
        model = sabr_implied_vol(forward, strikes, expiry, a, beta, r, n, correction)  # 모형 변동성
        return w * (model - market_vols)  # 가중 잔차

    alpha0 = atm_vol * forward ** (1.0 - beta)  # beta 백본에서 ATM과 맞는 alpha의 대략적 초기값
    lower = [1e-4, -0.999, 1e-3]  # alpha>0, |rho|<1, nu>0 경계
    upper = [10.0 * max(alpha0, 1e-3), 0.999, 10.0]  # 상한 (비현실적인 해 방지)
    best = None  # 지금까지의 최적 결과
    for rho0 in (-0.6, -0.2, 0.3):  # rho 초기값 후보 (주식 스큐는 음수 위주)
        for nu0 in (0.3, 0.8, 1.5):  # nu 초기값 후보
            x0 = np.array([alpha0, rho0, nu0])  # 이번 시작점
            try:  # 일부 시작점에서 수치 오류가 나도 전체가 죽지 않게
                sol = least_squares(residuals, x0, bounds=(lower, upper), method="trf")  # 경계형 TRF
            except (ValueError, FloatingPointError):  # 이 시작점은 건너뛴다
                continue
            if best is None or sol.cost < best.cost:  # 더 낮은 비용이면 갱신
                best = sol
    if best is None:  # 모든 시작점이 실패한 경우
        raise RuntimeError("SABR calibration failed for every starting point")
    a, r, n = best.x  # 최적 파라미터
    fitted = sabr_implied_vol(forward, strikes, expiry, a, beta, r, n, correction)  # 보정된 모형 변동성
    rmse = float(np.sqrt(np.mean((fitted - market_vols) ** 2)))  # 단순 RMSE (가중치 미적용)
    return SABRSliceFit(expiry, forward, SABRParams(float(a), beta, float(r), float(n)), rmse, bool(best.success))
