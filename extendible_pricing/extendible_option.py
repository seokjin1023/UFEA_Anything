"""Holder-extendible call option (Longstaff, 1990) 가격결정 모듈.

## 상품 구조

    최초 만기 T1, 행사가 K1인 콜옵션. T1 시점에 S1 = S(T1)이 관측되면:

        S1 >= K1  ->  그냥 행사, payoff = S1 - K1
        S1 <  K1  ->  추가 프리미엄 a를 내고 만기 T2(>T1), 행사가 K2인
                       새 콜옵션으로 "연장"할 권리 (행사 안 해도 됨)
                       payoff = max( C_BS(S1, K2, T2-T1) - a , 0 )

    즉 OTM 분기(S1<K1)의 payoff는 정확히 "콜옵션(K2, T2)을 기초자산으로
    하는 콜옵션(행사가 a, 만기 T1)"이다. 이게 이 상품이 compound option인
    이유다 (자세한 설명은 docs/extendible_option.md 참고).

## Forward vs Backward: 왜 이 모듈은 "정방향(forward) 시뮬레이션"만 쓰는가

    일반적으로 중도 시점에 "연장/행사/포기"를 선택하는 옵션(버뮤단식,
    아메리칸식)을 몬테카를로로 풀 때는 각 중도 시점에서 "계속 보유했을 때의
    기대가치(continuation value)"를 알아야 하는데, 이 값은 대개 폐형식이
    없어서 Longstaff-Schwartz Least-Squares Monte Carlo(LSM)처럼 미래
    경로를 먼저 다 뽑아놓고 회귀분석으로 "뒤에서 앞으로(backward)" 값을
    복원하는 절차가 필요하다.

    그런데 이 상품은 특수한 경우다: T1 시점에서 "연장했을 때 받는 것"이
    바로 표준 유러피언 콜옵션(K2, T2)이고, 기초자산이 GBM을 따른다는
    가정 하에서는 그 값이 블랙-숄즈 폐형식으로 *이미* 알려져 있다.
    다시 말해 "backward induction"이 필요한 부분(=continuation value 계산)을
    시뮬레이션이 아니라 수학적으로 미리 풀어서 공식으로 만들어 둔 것이다.

    따라서:
      1) S1을 T1까지 딱 한 번 "forward"로 시뮬레이션하고,
      2) 그 노드에서 블랙-숄즈 폐형식(=이미 backward로 풀린 해)을 대입해
         "연장할지/그만둘지"를 결정한 뒤,
      3) T1에서 0시점으로 할인해서 평균을 낸다.

    이 방식이 `price_forward_analytic_mc` 이다. 회귀나 backward 시간축
    시뮬레이션이 전혀 필요 없으므로 표준 몬테카를로(순수 forward MC)만으로
    충분하고, 분산도 훨씬 작다(노이즈원이 T1 시점 시뮬레이션 하나뿐이므로).

    비교/검증을 위해 `price_nested_backward_mc`도 함께 제공한다. 이건 만약
    "2단계 옵션에 폐형식이 없었다면"(예: 아메리칸형, 배리어형, 확률변동성
    등) 반드시 해야 했을 방식 -- T1까지 forward로 뽑은 각 경로마다 다시
    T1->T2 구간을 안쪽(inner)에서 시뮬레이션해 continuation value 자체를
    시뮬레이션으로 근사(=backward 값을 시뮬레이션으로 복원)하는 nested MC다.
    두 방법의 가격이 서로 수렴하는 것을 확인함으로써 "폐형식을 forward
    시뮬레이션에 대입하는 지름길"이 정당함을 검증할 수 있다.

    마지막으로 `price_lognormal_quad`는 GBM 하에서 S1이 로그정규분포를
    따른다는 사실을 이용한 수치적분(quadrature) 버전으로, 몬테카를로가
    아닌 결정론적 방법으로 "정답"에 가장 가까운 값을 준다 (그릭스 계산 시
    유한차분 노이즈가 실제 곡률 때문인지 몬테카를로 노이즈 때문인지
    구분하기 위한 기준점 역할).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from .black_scholes import bs_price


@dataclass
class ExtendibleParams:
    """Extendible call의 계약 조건 + 시장 파라미터를 한 곳에 모은 컨테이너."""

    S0: float      # 현재 기초자산 가격
    K1: float      # 1단계(최초) 만기 T1의 행사가
    K2: float      # 연장 시 2단계 옵션의 행사가 (보통 K1과 같거나 약간 높게 재설정)
    a: float       # 연장 프리미엄 (extension premium) - T1에서 지불
    r: float       # 무위험이자율 (연속복리)
    q: float       # 배당수익률 (연속복리)
    sigma: float   # 변동성 (연율)
    T1: float      # 최초 만기까지 남은 시간 (년)
    T2: float      # 연장 시 최종 만기까지 남은 시간 (년), T2 > T1

    @property
    def tau2(self) -> float:
        """연장 구간의 길이 (T1 -> T2)."""
        return self.T2 - self.T1


def _simulate_S1(p: ExtendibleParams, Z: np.ndarray) -> np.ndarray:
    """위험중립 GBM으로 T1 시점의 기초자산 가격을 '정방향(forward)'으로 시뮬레이션.

    S(T1) = S0 * exp( (r - q - 0.5*sigma^2)*T1 + sigma*sqrt(T1)*Z ),  Z ~ N(0,1)

    Z를 인자로 받는 이유: 그릭스를 유한차분으로 계산할 때 파라미터(S0, sigma
    등)만 바꾸고 난수 Z는 그대로 재사용하는 "공통난수법(Common Random
    Numbers, CRN)"을 쓰기 위함이다. CRN을 쓰지 않으면 감마/바나/보마처럼
    2차 차분으로 얻는 그릭스는 몬테카를로 노이즈에 완전히 파묻힌다.
    """
    return p.S0 * np.exp((p.r - p.q - 0.5 * p.sigma ** 2) * p.T1 + p.sigma * np.sqrt(p.T1) * Z)


def extendible_payoff_at_T1(S1: np.ndarray, p: ExtendibleParams) -> np.ndarray:
    """T1 시점의 payoff (할인 전).

    S1 >= K1 이면 즉시행사 payoff, S1 < K1 이면 "연장권"의 payoff
    (= 2단계 콜의 블랙-숄즈 가치 - 연장프리미엄, 단 음수면 0으로 포기).
    """
    S1 = np.asarray(S1, dtype=float)
    continuation_value = bs_price(S1, p.K2, p.r, p.q, p.sigma, p.tau2, is_call=True)
    extend_payoff = np.maximum(continuation_value - p.a, 0.0)
    exercise_payoff = S1 - p.K1
    return np.where(S1 >= p.K1, exercise_payoff, extend_payoff)


def extend_abandon_boundary(p: ExtendibleParams) -> float:
    """T1 시점 payoff 함수의 '두 번째 kink' S* 를 찾는다.

    T1의 payoff는 사실 두 군데에서 꺾인다.

      1) S1 = K1        : 즉시행사 <-> "연장을 고려" 분기의 경계
      2) S1 = S* (<K1)  : OTM 구간(S1<K1) 안에서, "연장" <-> "포기(0)" 분기의
                           경계. S*는 C_BS(S*, K2, r, q, sigma, T2-T1) = a를
                           만족하는 스팟 (연장해서 받는 새 콜옵션의 블랙-숄즈
                           가치가 딱 연장프리미엄 a와 같아지는 지점).

    두 kink 모두에서 payoff의 기울기가 불연속적으로 바뀌므로, 오늘
    시점(t=0)의 가격함수는 K1 근방뿐 아니라 S* 근방에서도 국소적으로 큰
    곡률(고차 그릭스)을 가질 수 있다. `scripts/demo_taleb_greeks.py`의
    스팟 스캔은 이 두 지점을 모두 그리드에 포함시켜 이 효과를 보여준다.
    """
    def f(s: float) -> float:
        c2 = float(bs_price(np.array([s]), p.K2, p.r, p.q, p.sigma, p.tau2, is_call=True)[0])
        return c2 - p.a

    lo, hi = 1e-6, p.K1
    if f(lo) > 0:  # a가 아주 작아 S->0에서도 연장가치가 a를 넘는 극단적인 경우
        return lo
    if f(hi) < 0:  # K1에서도 연장가치가 a에 못 미치면 (거의) 항상 포기
        return hi
    return brentq(f, lo, hi)


def price_forward_analytic_mc(
    p: ExtendibleParams,
    n_paths: int = 200_000,
    seed: int | None = None,
    antithetic: bool = True,
    Z: np.ndarray | None = None,
):
    """[주 방법] Forward 시뮬레이션 + 해석적(블랙-숄즈) continuation value.

    절차:
      1) T1 시점까지 GBM 경로를 forward로 시뮬레이션 (Z가 주어지면 그대로
         재사용 -> Greeks 계산 시 CRN 용도).
      2) 각 경로의 S1에서 `extendible_payoff_at_T1`로 payoff 계산
         (내부적으로 블랙-숄즈 폐형식을 이용해 "연장" 분기의 가치를 계산 -
         즉 backward로 미리 풀어둔 해를 forward 경로 위에 대입).
      3) e^{-r T1}으로 할인 후 경로 평균 = 몬테카를로 가격 추정치.

    antithetic=True 이면 분산감소를 위해 대칭난수(Z, -Z) 쌍을 사용한다.
    """
    if Z is None:
        rng = np.random.default_rng(seed)
        if antithetic:
            half = n_paths // 2
            z_half = rng.standard_normal(half)
            Z = np.concatenate([z_half, -z_half])
        else:
            Z = rng.standard_normal(n_paths)

    S1 = _simulate_S1(p, Z)
    payoff = extendible_payoff_at_T1(S1, p)
    discounted = np.exp(-p.r * p.T1) * payoff

    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(discounted.size))
    return price, stderr, Z


def price_nested_backward_mc(
    p: ExtendibleParams,
    n_outer: int = 20_000,
    n_inner: int = 2_000,
    seed: int | None = None,
):
    """[검증/대조용] Nested Monte Carlo - "폐형식이 없다고 가정"했을 때 필요한 방식.

    2단계 옵션(K2, T2)에 대해 블랙-숄즈 공식을 쓰지 않고, T1에서 각
    외부(outer) 경로마다 다시 내부(inner) 경로들을 T1->T2 구간에서
    forward 시뮬레이션해 continuation value를

        C2_hat(S1) ≈ e^{-r*(T2-T1)} * mean( max(S2 - K2, 0) )

    로 '시뮬레이션으로' 근사한다. 이는 실무에서 2단계 옵션이 아메리칸형,
    배리어형, 확률변동성 모형 등 폐형식이 없는 경우에 실제로 사용해야 하는
    절차(=Longstaff-Schwartz류 backward induction의 사촌)이며, 여기서는
    `price_forward_analytic_mc`가 정확한지 교차검증하는 용도로만 쓴다.

    비용: outer x inner 만큼의 경로가 필요해 계산량이 훨씬 크고, 추정치의
    분산도 (해석적 continuation value를 쓰는 경우보다) 훨씬 크다 -- 이는
    "폐형식이 있을 때는 절대 nested MC/backward induction을 쓸 필요가
    없다"는 본 모듈 설계 근거를 수치적으로 뒷받침한다.
    """
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal(n_outer)
    S1 = _simulate_S1(p, Z1)

    tau2 = p.tau2
    Z2 = rng.standard_normal((n_outer, n_inner))
    S2 = S1[:, None] * np.exp(
        (p.r - p.q - 0.5 * p.sigma ** 2) * tau2 + p.sigma * np.sqrt(tau2) * Z2
    )
    inner_payoff = np.maximum(S2 - p.K2, 0.0)
    C2_hat = np.exp(-p.r * tau2) * inner_payoff.mean(axis=1)

    extend_payoff = np.maximum(C2_hat - p.a, 0.0)
    exercise_payoff = S1 - p.K1
    payoff = np.where(S1 >= p.K1, exercise_payoff, extend_payoff)
    discounted = np.exp(-p.r * p.T1) * payoff

    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(n_outer))
    return price, stderr


def price_lognormal_quad(p: ExtendibleParams) -> float:
    """[결정론적 벤치마크] S1의 로그정규분포에 대한 수치적분(quadrature).

    몬테카를로가 아니라 scipy.integrate.quad로 payoff의 위험중립 기대값을
    직접 적분한다. 노이즈가 전혀 없으므로 (a) forward-analytic MC와
    (b) nested backward MC가 둘 다 같은 참값으로 수렴하는지 확인하는
    기준점(ground truth)으로 쓴다.
    """
    mu = np.log(p.S0) + (p.r - p.q - 0.5 * p.sigma ** 2) * p.T1
    sd = p.sigma * np.sqrt(p.T1)

    def integrand(s1: float) -> float:
        payoff = float(extendible_payoff_at_T1(np.array([s1]), p)[0])
        density = norm.pdf((np.log(s1) - mu) / sd) / (s1 * sd)
        return payoff * density

    lower = np.exp(mu - 10 * sd)
    upper = np.exp(mu + 10 * sd)
    integral, _ = quad(integrand, lower, upper, limit=400)
    return float(np.exp(-p.r * p.T1) * integral)
