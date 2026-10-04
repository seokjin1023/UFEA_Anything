"""ELS 몬테카를로 가격결정기: 시뮬레이션 경로 위에서 조기상환 / 낙인 / 만기 페이오프를 평가한다.

## 알고리즘 (경로별 상태 기계, 시간 순서로 한 번만 훑는다)
  상태: alive(아직 상환 안 됨), ki_hit(지금까지 KI 터치 여부)
  매 영업일 k마다
    1) worst = min_i S_i(t_k) / S_i(0)                      # Worst-of 성과
    2) ki_hit |= (worst < 30%)                              # 종가 기준 일별 낙인 관찰
    3) k가 j차 조기상환평가일이면: alive & (worst >= barrier_j) 경로는 payout_j 를 받고 종료
    4) k가 최종관찰일이면: 남은 경로는 `maturity_payout`(187% 또는 worst 성과)을 받고 종료
  각 경로의 현재가치 = 상환금액 x exp(-r x 지급일까지의 시간)

  가격 = 경로 평균, 표준오차 = 경로(또는 대조변수 짝) 표준편차 / sqrt(표본 수).

## 대조변수 짝을 고려한 표준오차
simulation.py는 경로 배열의 앞 절반(Z)과 뒤 절반(-Z)을 짝으로 만든다. 짝의 두 값은 서로 상관되어 있으므로
"짝 평균"을 독립 표본으로 보고 표준오차를 구해야 한다 (경로 하나하나를 독립으로 보면 SE를 과대평가한다).

## 할인
무위험 이자율 r 하나로 연속복리 할인한다 (발행사 신용스프레드 미반영). 지급은 조기상환 평가일 +3영업일,
만기는 설명서의 만기일이다.

## 참고문헌
[1] Glasserman, P. (2004), Monte Carlo Methods in Financial Engineering, Springer, Sec. 4.2(대조변수), Ch. 1(MC 추정과 SE).
[2] Hull, J. C., Options, Futures, and Other Derivatives (교과서), 장벽옵션 / 몬테카를로 장.
"""
from __future__ import annotations  # 전방참조 타입 힌트

from dataclasses import dataclass  # 결과 컨테이너

import numpy as np  # 배열 연산

from .product import ELSSchedule, ELSTerms, autocall_redeems, maturity_payout  # 상품 정의와 페이오프
from .simulation import CorrelatedLocalVolSimulator  # 경로 생성기


@dataclass(frozen=True)
class PricingResult:
    """가격결정 결과. 모든 금액은 액면 1.0 대비 비율이다 (1.0551 = 액면의 105.51% = 10,551원)."""

    price: float  # 이론가 (현재가치 평균)
    std_error: float  # 몬테카를로 표준오차
    autocall_probs: np.ndarray  # 차수별 조기상환 확률 (길이 11)
    maturity_full_prob: float  # 만기까지 가서 187%를 받는 확률
    loss_prob: float  # 원금 손실(상환금 < 100%) 확률
    mean_payout_given_loss: float  # 손실이 났을 때의 평균 상환금 (액면 대비)
    ki_touch_prob: float  # 전체 경로 중 "만기까지 살아남았고 KI도 터치한" 경로의 비율
    expected_life: float  # 평균 상환 시점 (연)
    n_paths: int  # 사용한 경로 수
    payouts: np.ndarray  # 경로별 (할인 전) 상환금액 (분포 그래프용)
    pay_times: np.ndarray  # 경로별 지급 시점 (연)

    def price_per_certificate(self, notional: float) -> float:
        """1증권 액면(원)으로 환산한 이론가."""
        return self.price * notional  # 비율 x 액면

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """정규근사 95% 신뢰구간."""
        return self.price - z * self.std_error, self.price + z * self.std_error  # 가격 +- z SE


class ELSPricer:
    """시뮬레이터 + 상품 + 할인율로 ELS 이론가를 계산한다."""

    def __init__(
        self,
        terms: ELSTerms,
        schedule: ELSSchedule,
        simulator: CorrelatedLocalVolSimulator,
        rate: float,
    ) -> None:
        if simulator.n_assets != len(terms.underlyings):  # 자산 수 일치 확인
            raise ValueError("simulator assets must match terms.underlyings")
        if simulator.times.size != schedule.times.size or not np.allclose(simulator.times, schedule.times):
            raise ValueError("simulator time grid must equal the schedule time grid")  # 같은 격자여야 이벤트 인덱스가 맞는다
        self.terms = terms  # 계약 조건
        self.schedule = schedule  # 격자와 이벤트 인덱스
        self.simulator = simulator  # 경로 생성기
        self.rate = rate  # 연속복리 할인율

    def price(self, n_paths: int = 100_000, seed: int = 0, antithetic: bool = True) -> PricingResult:
        """몬테카를로로 이론가와 상환 통계를 계산한다."""
        terms, sched = self.terms, self.schedule  # 짧게 쓰기 위한 별칭
        s0 = self.simulator.spots[:, None]  # 최초기준가격 (n_assets, 1) -> 성과 계산용
        alive = np.ones(n_paths, dtype=bool)  # 아직 상환되지 않은 경로
        ki_hit = np.zeros(n_paths, dtype=bool)  # KI(30% 미만) 터치 여부
        payout = np.zeros(n_paths)  # 경로별 상환금액 (할인 전)
        pay_time = np.zeros(n_paths)  # 경로별 지급 시점
        redeem_time = np.zeros(n_paths)  # 경로별 상환 결정(평가) 시점 (평균 수명 계산용)
        ac_index = {step: j for j, step in enumerate(sched.autocall_steps)}  # 격자 인덱스 -> 조기상환 차수(0부터)
        ac_counts = np.zeros(len(sched.autocall_steps))  # 차수별 조기상환 경로 수

        for k, t, S in self.simulator.iter_steps(n_paths, seed, antithetic):  # 영업일 순서로 상태 전이
            if k == 0:  # 초기 시점에는 관찰이 없다 (KI는 최초기준가격평가일 "익일"부터)
                continue
            worst = (S / s0).min(axis=0)  # Worst-of 성과: 자산별 S/S0 중 최솟값
            ki_hit |= worst < terms.ki_barrier  # 종가가 KI 배리어 "미만"이면 낙인 발생 (한번 켜지면 유지)
            if k in ac_index:  # 조기상환평가일인가
                j = ac_index[k]  # 몇 차인가
                redeem = alive & autocall_redeems(worst, terms.autocall_barriers[j])  # 조건 충족 + 아직 생존
                payout[redeem] = terms.autocall_payouts[j]  # 상환금액 기록
                pay_time[redeem] = sched.autocall_pay_times[j]  # 지급 시점(평가일 + 3영업일)
                redeem_time[redeem] = t  # 평가 시점
                ac_counts[j] = redeem.sum()  # 차수별 상환 경로 수
                alive &= ~redeem  # 상환된 경로는 이후 단계에서 제외
            if k == sched.final_step:  # 최종관찰일 (만기평가일)
                final = maturity_payout(worst, ki_hit, terms)  # (12)(13)(14) 규칙 적용
                payout[alive] = final[alive]  # 생존 경로에만 만기 상환금 기록
                pay_time[alive] = sched.maturity_pay_time  # 만기 지급일
                redeem_time[alive] = t  # 만기평가 시점
                survived_ki = alive & ki_hit  # 만기까지 생존 + KI 터치 경로 (통계용)
                alive = np.zeros(n_paths, dtype=bool)  # 모두 종료 처리
                break  # 마지막 관찰일 이후는 시뮬레이션할 필요 없다

        pv = payout * np.exp(-self.rate * pay_time)  # 경로별 현재가치 (연속복리 할인)
        if antithetic:  # 대조변수 짝 구조: 앞 절반(Z)과 뒤 절반(-Z)
            half = n_paths // 2  # 짝의 수
            pairs = 0.5 * (pv[:half] + pv[half:])  # 짝 평균을 독립 표본으로 취급
            std_error = float(pairs.std(ddof=1) / np.sqrt(half))  # 짝 기준 표준오차
        else:  # 일반 MC
            std_error = float(pv.std(ddof=1) / np.sqrt(n_paths))  # 경로 기준 표준오차
        lost = payout < 1.0 - 1e-12  # 원금 손실 경로
        mat_full = np.isclose(payout, terms.maturity_payout) & (pay_time == sched.maturity_pay_time)  # 만기 187% 경로
        return PricingResult(
            price=float(pv.mean()),
            std_error=std_error,
            autocall_probs=ac_counts / n_paths,
            maturity_full_prob=float(mat_full.mean()),
            loss_prob=float(lost.mean()),
            mean_payout_given_loss=float(payout[lost].mean()) if lost.any() else float("nan"),
            ki_touch_prob=float(survived_ki.mean()),
            expected_life=float(redeem_time.mean()),
            n_paths=n_paths,
            payouts=payout,
            pay_times=pay_time,
        )
