"""Dupire 국소변동성(local volatility): SABR 변동성 곡면으로부터 sigma_loc(S, t)를 구한다.

## 아이디어
Dupire (1994)에 따르면, 모든 유러피언 콜 가격 C(K, T)가 주어졌을 때 이를 정확히 재현하는 단일 확산과정

    dS_t / S_t = mu dt + sigma_loc(S_t, t) dW_t

가 (유일하게) 존재하고, 국소변동성은 가격의 편미분으로 결정된다.

    sigma_loc^2(K, T) = [ C_T + mu K C_K + q C ] / [ (1/2) K^2 C_KK ],        q = r - mu

사용자가 말한 "Dupire로 구하는 forward vol"이 바로 이 값이다. 즉, 시점 T에 기초자산이 K일 때의
*순간 선도 변동성(instantaneous forward volatility)* 이다. (두 만기 사이의 선도 "내재"변동성과는 다른 개념이다.)

## 구현 방식 두 가지
  A. `local_variance_from_implied` (기본): Gatheral (2006) eq. (1.10)의 "내재 총분산" 표현.
        y = ln(K / F_T), w(y, T) = sigma_imp^2 * T 라 하면
            sigma_loc^2 = w_T / [ 1 - (y/w) w_y + (1/4)(-1/4 - 1/w + y^2/w^2) w_y^2 + (1/2) w_yy ]
     가격을 두 번 미분하는 것보다 수치적으로 훨씬 안정적이다. SABR 곡면 w(y, T)를 중심차분으로 미분한다.
     분모는 곧 Breeden-Litzenberger 확률밀도에 비례하는 g(y, T)이며, g <= 0 이면 버터플라이 차익거래다.
  B. `local_vol_from_prices` (검증용): 위 Dupire 원식을 블랙 가격의 유한차분으로 직접 계산한다.
     두 방식이 일치하는지를 테스트에서 확인해 구현 오류를 잡는다.

## 격자와 조회
몬테카를로가 매 스텝, 매 경로마다 sigma_loc(S, t)를 필요로 하므로 (t x ln(S/F_t)) 격자에 미리 계산해 두고
선형보간으로 조회한다. 격자 밖의 S는 가장자리 값으로 고정한다. 차익거래 영역 때문에 g가 0 이하이거나 w_T가
0 이하인 곳은 [min_vol, max_vol] 안으로 클리핑하며, 그 비율을 `diagnostics()`로 보고한다.

## 참고문헌
[1] Dupire, B. (1994), "Pricing with a Smile", Risk 7(1), 18-20.
[2] Derman, E., Kani, I. (1994), "Riding on a Smile", Risk 7(2), 32-39.
    (이산 이항트리 버전의 같은 아이디어)
[3] Gatheral, J. (2006), The Volatility Surface: A Practitioner's Guide, Wiley, Ch.1
    (eq. 1.10: 내재 총분산으로 표현한 Dupire 식, 무차익 조건 g >= 0)
[4] Andersen, L., Brotherton-Ratcliffe, R. (1997/98), "The equity option volatility smile: an implicit
    finite difference approach", Journal of Computational Finance 1(2), 5-37. (국소변동성 수치 안정성)
[5] Breeden, D., Litzenberger, R. (1978), "Prices of state-contingent claims implicit in option prices",
    Journal of Business 51(4), 621-651. (C_KK = 확률밀도 관계)
"""
from __future__ import annotations  # 전방참조 타입 힌트

from typing import Protocol  # 국소변동성 모형의 공통 인터페이스 정의

import numpy as np  # 배열 연산

from .black import black_call  # 가격식 검증용 블랙 콜 가격
from .vol_surface import SABRVolSurface  # SABR 변동성 곡면


class LocalVolModel(Protocol):
    """시뮬레이터가 요구하는 최소 인터페이스: 시점 t와 현물 배열 S -> 국소변동성 배열."""

    def local_vol(self, t: float, S: np.ndarray) -> np.ndarray: ...  # 구현체는 이 시그니처만 맞추면 된다


class FlatLocalVol:
    """상수 변동성 모형(Black-Scholes). 투자설명서식 단일 변동성 벤치마크 / 테스트용."""

    def __init__(self, vol: float) -> None:
        self.vol = vol  # 상수 변동성

    def local_vol(self, t: float, S: np.ndarray) -> np.ndarray:
        return np.full_like(S, self.vol, dtype=float)  # 모든 경로에 같은 값


def local_variance_from_implied(
    surface: SABRVolSurface, y: np.ndarray, T: np.ndarray, hy: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Gatheral (2006) eq. (1.10)로 국소분산과 분모 g를 계산한다 (y, T는 브로드캐스팅되는 배열).

    반환: (국소분산 w_T / g,  g). 클리핑은 호출자가 한다.
    """
    y = np.asarray(y, dtype=float)  # 선도 로그머니니스
    T = np.asarray(T, dtype=float)  # 만기
    w = surface.total_variance(y, T)  # w(y, T)
    w_up = surface.total_variance(y + hy, T)  # w(y + h, T)
    w_dn = surface.total_variance(y - hy, T)  # w(y - h, T)
    w_y = (w_up - w_dn) / (2.0 * hy)  # dw/dy 중심차분
    w_yy = (w_up - 2.0 * w + w_dn) / hy ** 2  # d2w/dy2 중심차분
    kT = np.minimum(0.01, 0.5 * T)  # 만기 방향 차분 폭 (T가 작아도 T-kT > 0 보장)
    w_T = (surface.total_variance(y, T + kT) - surface.total_variance(y, T - kT)) / (2.0 * kT)  # dw/dT (y 고정)
    g = (  # 분모: 1 - (y/w) w_y + 1/4 (-1/4 - 1/w + y^2/w^2) w_y^2 + 1/2 w_yy
        1.0 - y / w * w_y
        + 0.25 * (-0.25 - 1.0 / w + y ** 2 / w ** 2) * w_y ** 2
        + 0.5 * w_yy
    )
    return w_T / np.where(np.abs(g) < 1e-12, 1e-12, g), g  # 0 나눗셈 방지 후 국소분산 반환


def local_vol_from_prices(
    surface: SABRVolSurface, K: np.ndarray, T: np.ndarray, rate: float,
) -> np.ndarray:
    """Dupire 원식 sigma^2 = (C_T + mu K C_K + q C) / (K^2 C_KK / 2)를 블랙 가격의 유한차분으로 계산 (검증용)."""
    K = np.asarray(K, dtype=float)  # 행사가
    T = np.asarray(T, dtype=float)  # 만기
    mu = surface.drift  # 위험중립 드리프트 r - q
    q = rate - mu  # 배당수익률 q = r - mu

    def C(k, t):  # 곡면의 SABR 변동성을 블랙 공식에 넣은 콜 가격
        return black_call(surface.forward(t), k, t, surface.implied_vol(k, t), rate)

    hK = 0.01 * K  # 행사가 방향 차분 폭 (1%)
    hT = 0.005  # 만기 방향 차분 폭 (약 1.3 영업일)
    c0 = C(K, T)  # 중심 가격
    c_K = (C(K + hK, T) - C(K - hK, T)) / (2.0 * hK)  # dC/dK
    c_KK = (C(K + hK, T) - 2.0 * c0 + C(K - hK, T)) / hK ** 2  # d2C/dK2
    c_T = (C(K, T + hT) - C(K, T - hT)) / (2.0 * hT)  # dC/dT
    return np.sqrt((c_T + mu * K * c_K + q * c0) / (0.5 * K ** 2 * c_KK))  # Dupire 국소변동성


class DupireLocalVol:
    """SABR 곡면에서 만든 Dupire 국소변동성 격자. `local_vol(t, S)`로 조회한다."""

    def __init__(
        self,
        surface: SABRVolSurface,
        t_max: float,
        n_t: int = 121,
        x_min: float = -1.6,
        x_max: float = 1.0,
        n_x: int = 161,
        min_vol: float = 0.05,
        max_vol: float = 2.0,
        t_floor: float = 0.02,
    ) -> None:
        self.surface = surface  # 원천 곡면
        self.t_grid = np.linspace(0.0, t_max, n_t)  # 시간 격자 (t=0 포함)
        self.x_grid = np.linspace(x_min, x_max, n_x)  # x = ln(S / F_t) 격자 (S/F 약 0.2 ~ 2.7)
        self._dt = self.t_grid[1] - self.t_grid[0]  # 균등 시간 간격
        T_eval = np.maximum(self.t_grid, t_floor)[:, None]  # t -> 0 에서 w -> 0 이므로 하한을 두고 평가
        lv2, g = local_variance_from_implied(surface, self.x_grid[None, :], T_eval)  # (n_t, n_x) 국소분산, g
        self._frac_low = float(np.mean(lv2 < min_vol ** 2))  # 하한으로 올려진 노드 비율
        self._frac_high = float(np.mean(lv2 > max_vol ** 2))  # 상한으로 눌린 노드 비율
        self._frac_g_neg = float(np.mean(g <= 0.0))  # 밀도가 음수인(차익거래) 노드 비율
        lv2 = np.where(g <= 0.0, max_vol ** 2, lv2)  # 음의 밀도 영역은 안전하게 상한 변동성으로 대체
        self.lv = np.sqrt(np.clip(lv2, min_vol ** 2, max_vol ** 2))  # 클리핑된 국소변동성 격자

    def local_vol(self, t: float, S: np.ndarray) -> np.ndarray:
        """시점 t, 현물 S(배열)에서의 국소변동성. 시간은 선형보간, x 방향도 선형보간 (경계 밖은 flat)."""
        pos = float(np.clip(t, 0.0, self.t_grid[-1]) / self._dt)  # 시간 격자상의 실수 위치
        i = min(int(pos), self.t_grid.size - 2)  # 아래쪽 격자 인덱스 (마지막 구간 보호)
        frac = pos - i  # 두 격자 사이의 보간 가중치
        row = (1.0 - frac) * self.lv[i] + frac * self.lv[i + 1]  # 시점 t의 x-단면
        x = np.log(S) - np.log(self.surface.forward(t))  # x = ln(S / F_t)
        return np.interp(x, self.x_grid, row)  # np.interp는 범위 밖에서 가장자리 값을 유지

    def diagnostics(self) -> dict:
        """클리핑/차익거래 비율 진단 (0에 가까울수록 곡면이 깨끗하다)."""
        return {
            "frac_below_min_vol": self._frac_low,  # 하한 클리핑 비율
            "frac_above_max_vol": self._frac_high,  # 상한 클리핑 비율
            "frac_negative_density": self._frac_g_neg,  # g <= 0 비율 (버터플라이 차익거래)
        }
