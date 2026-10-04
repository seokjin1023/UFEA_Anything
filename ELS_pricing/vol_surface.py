"""SABR 변동성 곡면: 만기별 슬라이스 보정 -> 파라미터 기간구조 보간 -> 임의의 (K, T) 내재변동성.

## 구성 방법
  1. 호가 격자의 각 만기를 `calibrate_sabr_slice`로 독립 보정해 (alpha_i, rho_i, nu_i)를 얻는다.
     (beta는 전 만기에서 동일하게 고정)
  2. 보정되지 않은 만기 T에서는 (alpha, rho, nu)를 만기에 대해 단조 PCHIP 보간(Fritsch-Carlson 1980)하고,
     범위 밖은 양 끝 값을 유지(flat extrapolation)한다. 그런 다음 그 T로 Hagan 식을 다시 평가한다.
     (식 안의 시간 보정 항이 T에 의존하므로 변동성 자체는 T에서 연속적으로 변한다.)
  3. Dupire가 요구하는 총분산 w(y, T) = sigma^2 T (y = ln(K/F_T), 선도 로그머니니스)를 제공한다.

## 한계 (정직한 주의)
  Hagan 근사식은 저행사가 / 장기만기에서 확률밀도가 음수가 되는 차익거래를 허용할 수 있다 ([2]).
  만기별 독립 보정 + 파라미터 보간은 달력 스프레드 차익거래도 보장하지 않는다.
  그래서 `calendar_arbitrage_fraction`을 진단용으로 제공하고, 버터플라이 쪽은 dupire.py가 진단한다.

## 참고문헌
[1] Hagan, Kumar, Lesniewski, Woodward (2002), "Managing Smile Risk", Wilmott Magazine. (sabr.py 참조)
[2] Hagan, Kumar, Lesniewski, Woodward (2014), "Arbitrage Free SABR", Wilmott Magazine.
[3] Gatheral, J. (2006), The Volatility Surface, Wiley, Ch.1 (총분산 w, 달력/버터플라이 무차익 조건).
[4] Fritsch, F. N., Carlson, R. E. (1980), "Monotone piecewise cubic interpolation",
    SIAM Journal on Numerical Analysis 17(2), 238-246. (PCHIP 보간의 근거)
"""
from __future__ import annotations  # 전방참조 타입 힌트

from typing import Literal  # correction 옵션 타입

import numpy as np  # 배열 연산
from scipy.interpolate import PchipInterpolator  # 단조 3차 보간

from .market_data import MarketQuotes  # 입력 호가 형식
from .sabr import SABRSliceFit, calibrate_sabr_slice, sabr_implied_vol  # SABR 공식/보정


class SABRVolSurface:
    """한 자산의 SABR 변동성 곡면 sigma(K, T)."""

    def __init__(
        self,
        name: str,
        spot: float,
        drift: float,
        beta: float,
        correction: Literal["hagan", "obloj"],
        slices: list[SABRSliceFit],
    ) -> None:
        if len(slices) < 2:  # PCHIP 보간에는 최소 두 개의 만기가 필요하다
            raise ValueError("need at least two calibrated expiries")
        self.name = name  # 자산 이름
        self.spot = spot  # 현물
        self.drift = drift  # 위험중립 드리프트 (선도가격 계산용)
        self.beta = beta  # 고정된 beta
        self.correction = correction  # 사용한 SABR 공식 종류
        self.slices = slices  # 만기별 보정 결과 (진단/표 출력용)
        T = np.array([s.expiry for s in slices])  # 만기 벡터
        self._t_min, self._t_max = float(T[0]), float(T[-1])  # 보간 가능 범위 (밖은 flat)
        self._alpha = PchipInterpolator(T, [s.params.alpha for s in slices])  # alpha(T)
        self._rho = PchipInterpolator(T, [s.params.rho for s in slices])  # rho(T)
        self._nu = PchipInterpolator(T, [s.params.nu for s in slices])  # nu(T)

    @classmethod
    def calibrate(
        cls,
        quotes: MarketQuotes,
        beta: float = 1.0,
        correction: Literal["hagan", "obloj"] = "hagan",
    ) -> "SABRVolSurface":
        """호가 격자의 모든 만기를 보정해 곡면 객체를 만든다."""
        slices = []  # 보정 결과 누적
        for i, T in enumerate(quotes.expiries):  # 만기별로
            F = float(quotes.forward(T))  # 이 만기 선도가격
            slices.append(  # 슬라이스 보정 결과 추가
                calibrate_sabr_slice(F, float(T), quotes.strikes(i), quotes.vols[i], beta, correction)
            )
        return cls(quotes.name, quotes.spot, quotes.drift, beta, correction, slices)

    def forward(self, T) -> np.ndarray:
        """만기 T의 선도가격 F_T = S0 exp(mu T)."""
        return self.spot * np.exp(self.drift * np.asarray(T, dtype=float))  # 연속복리 선도

    def params_at(self, T) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """만기 T의 (alpha, rho, nu). 범위 밖 T는 가장 가까운 보정 만기 값으로 고정한다."""
        Tc = np.clip(np.asarray(T, dtype=float), self._t_min, self._t_max)  # 보간 범위로 클립
        return self._alpha(Tc), self._rho(Tc), self._nu(Tc)  # 세 파라미터 보간값

    def implied_vol(self, K, T) -> np.ndarray:
        """절대 행사가 K, 만기 T의 SABR 내재변동성 (브로드캐스팅 가능)."""
        T = np.asarray(T, dtype=float)  # 배열화
        a, r, n = self.params_at(T)  # 만기 T의 파라미터
        return sabr_implied_vol(self.forward(T), K, T, a, self.beta, r, n, self.correction)  # Hagan/Oblój 식

    def implied_vol_fwd(self, y, T) -> np.ndarray:
        """선도 로그머니니스 y = ln(K/F_T) 로 표현한 내재변동성 (Dupire 식에서 사용)."""
        F = self.forward(T)  # 선도가격
        return self.implied_vol(F * np.exp(np.asarray(y, dtype=float)), T)  # K = F e^y

    def total_variance(self, y, T) -> np.ndarray:
        """총분산 w(y, T) = sigma(y, T)^2 * T."""
        T = np.asarray(T, dtype=float)  # 배열화
        return self.implied_vol_fwd(y, T) ** 2 * T  # sigma^2 T

    def calendar_arbitrage_fraction(self, y_grid: np.ndarray, T_grid: np.ndarray) -> float:
        """w(y, T)가 T에 대해 감소하는 격자 셀의 비율 (0이면 달력 스프레드 차익거래 없음)."""
        w = np.array([self.total_variance(y_grid, T) for T in T_grid])  # shape (n_T, n_y)
        return float(np.mean(np.diff(w, axis=0) < -1e-12))  # 인접 만기 간 감소 비율

    def calibration_table(self) -> list[dict]:
        """만기별 보정 파라미터와 RMSE를 사전 목록으로 반환 (출력용)."""
        return [  # 슬라이스마다 한 행
            {
                "T": s.expiry, "alpha": s.params.alpha, "beta": s.params.beta,
                "rho": s.params.rho, "nu": s.params.nu, "rmse_vol_pts": s.rmse * 100.0, "ok": s.success,
            }
            for s in self.slices
        ]
