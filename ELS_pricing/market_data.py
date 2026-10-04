"""시장 데이터 구조와 (데모용) 합성 내재변동성 호가 생성기.

이 프로젝트에는 실제 삼성전자 / KOSPI200 옵션 호가가 제공되지 않았다. 그래서

  * `MarketQuotes`  : "만기 x 선도머니니스" 격자의 내재변동성 호가를 담는 표준 입력 형식.
                      실제 데이터가 생기면 이 클래스에 배열만 채워 넣으면 이후 파이프라인
                      (SABR 보정 -> Dupire -> 몬테카를로)이 그대로 동작한다.
  * `make_synthetic_quotes` : 투자설명서의 ATM 변동성(KOSPI200 36.29%, 삼성전자 47.99%)에
                      수준을 맞춘 가상의 스마일을 만들어 내는 함수. 실제 시장 스큐가 아니라
                      "파이프라인을 끝까지 돌려보기 위한 대체 입력"이다.
  * `mirae_38078_market` : 투자설명서에 명시된 시장변수와 데모용 가정치를 묶은 팩토리.

## 용어 / 가정
  * drift  : 위험중립 드리프트 mu = r - q (이 프로젝트의 요구사항대로 "이미 주어진 값"으로 취급).
             선도가격은 F_T = S0 * exp(mu * T) 가 된다.
  * rate   : 할인에만 쓰이는 무위험 이자율 r (연속복리).
  * 투자설명서 p.144: 변동성 KOSPI200 36.29% / 삼성전자 47.99%, 일별수익률 상관계수 0.9558.
  * 투자설명서 p.148: 최초기준가격 예시 KOSPI200 1,088.61pt / 삼성전자 266,000원.
  * rate, 배당수익률(drift)은 투자설명서에 없으므로 아래 값은 "자리표시자(placeholder)" 가정이다.
"""
from __future__ import annotations  # 전방참조 타입 힌트

from dataclasses import dataclass  # 불변 데이터 컨테이너

import numpy as np  # 배열 연산
from scipy.optimize import brentq  # 1차원 근 찾기 (합성 호가의 alpha를 ATM에 맞출 때)

from .sabr import sabr_implied_vol  # 합성 스마일의 "진짜" SABR 곡선


@dataclass(frozen=True)
class AssetSpec:
    """기초자산 하나의 기본 정보."""

    name: str  # 자산 이름 (그래프/표 라벨)
    spot: float  # 최초기준가격 S0
    drift: float  # 위험중립 드리프트 mu = r - q (주어진 값)
    atm_vol: float  # 만기 부근 ATM 변동성 (합성 호가의 수준을 정하는 용도)


@dataclass(frozen=True)
class MarketQuotes:
    """한 자산의 내재변동성 호가 격자 (행: 만기, 열: 선도머니니스 K/F_T)."""

    name: str  # 자산 이름
    spot: float  # 현물 S0
    drift: float  # 위험중립 드리프트 mu
    expiries: np.ndarray  # 만기 벡터 (연), 오름차순, shape (n_T,)
    fwd_moneyness: np.ndarray  # K / F_T 벡터, 오름차순, shape (n_K,)
    vols: np.ndarray  # 내재변동성 행렬, shape (n_T, n_K)

    def forward(self, T) -> np.ndarray:
        """만기 T의 선도가격 F_T = S0 exp(mu T)."""
        return self.spot * np.exp(self.drift * np.asarray(T, dtype=float))  # 연속복리 선도

    def strikes(self, i: int) -> np.ndarray:
        """i번째 만기의 절대 행사가 K = (K/F_T) * F_T."""
        return self.fwd_moneyness * self.forward(self.expiries[i])  # 머니니스 -> 행사가


def make_synthetic_quotes(
    asset: AssetSpec,
    expiries: tuple[float, ...] = (1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0),
    fwd_moneyness: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5),
    short_end_ratio: float = 1.15,
    rho: float = -0.55,
    nu_short: float = 1.2,
    nu_long: float = 0.5,
    noise: float = 0.002,
    seed: int = 7,
) -> MarketQuotes:
    """ATM 수준이 `asset.atm_vol`로 수렴하는 가상의 SABR 스마일 호가를 만든다.

    * ATM 기간구조: 단기 = atm_vol * short_end_ratio -> 장기 = atm_vol (지수 수렴, 시간상수 1년)
    * vol-of-vol  : nu_short -> nu_long 으로 지수 수렴 (스마일이 장기로 갈수록 평탄해지는 전형적 모양)
    * 각 만기에서 ATM이 정확히 맞도록 alpha를 brentq로 역산한 뒤, 호가에 N(0, noise^2) 노이즈를 더한다.
    """
    rng = np.random.default_rng(seed)  # 재현 가능한 난수
    T_arr = np.asarray(expiries, dtype=float)  # 만기 배열
    m_arr = np.asarray(fwd_moneyness, dtype=float)  # 머니니스 배열
    vols = np.empty((T_arr.size, m_arr.size))  # 결과 행렬
    long_v = asset.atm_vol  # 장기 ATM
    short_v = asset.atm_vol * short_end_ratio  # 단기 ATM
    for i, T in enumerate(T_arr):  # 만기별로
        atm_T = long_v + (short_v - long_v) * np.exp(-T)  # 이 만기의 목표 ATM 변동성
        nu_T = nu_long + (nu_short - nu_long) * np.exp(-T)  # 이 만기의 vol-of-vol
        F = asset.spot * np.exp(asset.drift * T)  # 이 만기 선도가격
        # beta=1 에서 alpha를 바꿔가며 ATM 변동성이 목표와 같아지는 지점을 찾는다.
        # (상한 1.5: alpha가 너무 크면 장기 만기에서 시간보정항이 음수가 되어 alpha에 대해 비단조가 된다)
        alpha_T = brentq(lambda a: float(sabr_implied_vol(F, F, T, a, 1.0, rho, nu_T)) - atm_T, 1e-3, 1.5)
        smile = sabr_implied_vol(F, m_arr * F, T, alpha_T, 1.0, rho, nu_T)  # 이 만기의 스마일
        vols[i] = smile + rng.normal(0.0, noise, size=m_arr.size)  # 호가 노이즈 추가
    return MarketQuotes(asset.name, asset.spot, asset.drift, T_arr, m_arr, vols)


@dataclass(frozen=True)
class MarketAssumptions:
    """가격결정에 필요한 시장 가정 일체."""

    rate: float  # 할인 이자율 r (연속복리)
    assets: tuple[AssetSpec, ...]  # 기초자산 목록 (순서가 상관행렬의 순서)
    correlation: np.ndarray  # 일별수익률 상관행렬 (n x n)


def mirae_38078_market() -> MarketAssumptions:
    """투자설명서 수치 + 데모용 가정 (rate 3.0%, 배당수익률 KOSPI200 1.5% / 삼성전자 2.0%)."""
    rate = 0.03  # [가정] 무위험 이자율 (투자설명서에 없음)
    kospi = AssetSpec("KOSPI200", 1088.61, rate - 0.015, 0.3629)  # S0, drift=r-q(q 가정 1.5%), 설명서 변동성
    samsung = AssetSpec("Samsung Electronics", 266000.0, rate - 0.020, 0.4799)  # drift(q 가정 2.0%), 설명서 변동성
    rho = 0.9558  # 설명서 p.144 상관계수 (180영업일 역사적)
    corr = np.array([[1.0, rho], [rho, 1.0]])  # 2x2 상관행렬
    return MarketAssumptions(rate, (kospi, samsung), corr)
