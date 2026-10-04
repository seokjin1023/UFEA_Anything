"""ELS 상품 정의: 미래에셋증권 제38078회 파생결합증권(KOSPI200 / 삼성전자 보통주, 스텝다운 Worst-of, KI 30%).

투자설명서(핵심상품설명서 + 간이투자설명서)의 "1. 상품개요 / 2. 권리의 내용 / (2) 손익구조"를 코드로 옮긴 모듈이다.

## 상품 구조 요약 (모든 숫자는 투자설명서 p.142, p.145-148 기준)
  * 기초자산: KOSPI200 지수, 삼성전자 보통주 (2개, Worst-of: 두 자산 중 *나쁜 쪽* 기준)
  * 발행일(= 최초기준가격평가일) 2026-09-10, 만기일 2029-09-10, 만기평가일(최종관찰일) 2029-09-05
  * 자동조기상환: 3개월마다 11회. 각 평가일에 *모든* 기초자산의 종가가 최초기준가격의 배리어 이상이면
    액면 x (1 + 7.25% x 차수)를 지급하고 종료.
        1-4차 90% / 5-8차 85% / 9-11차 80%  (스텝다운),  지급은 평가일 후 3영업일
  * 만기 (조기상환이 한 번도 안 된 경우):
        (12) 만기평가가격이 모든 기초자산에서 최초기준가격의 70% 이상            -> 액면 x 187%
        (13) (12)가 아니고, 최초기준가격평가일 익일 ~ 최종관찰일 사이 어느 기초자산도
             종가가 30% 미만으로 내려간 적이 없음 (낙인(KI) 미발생)            -> 액면 x 187%
        (14) (12)가 아니고, 어느 하나라도 30% 미만으로 하락한 적이 있음 (KI 발생)
             -> 액면 x (만기평가가격 / 최초기준가격)  [기준종목 = 만기 성과가 가장 낮은 자산]
  * KI는 종가 기준 일별 관찰이므로, 영업일 단위로 시뮬레이션하면 이산 관찰 그대로 정확히 재현된다
    (Brownian bridge 같은 연속관찰 보정이 필요 없다).

## 이 모듈의 구성
  * `ELSTerms`      : 계약 조건(날짜, 배리어, 쿠폰)을 담는 불변 데이터 클래스
  * `ELSSchedule`   : 영업일 시뮬레이션 격자, 조기상환일/지급일의 격자 인덱스와 연 단위 시간
  * `build_schedule`: ELSTerms -> ELSSchedule
  * `autocall_redeems`, `maturity_payout` : 순수 페이오프 함수 (벡터화, 상태 없음)
  * `mirae_38078_terms` : 위 계약을 채워 넣은 팩토리

## 가정과 단순화
  * 영업일 = 월~금 (한국 공휴일은 `holidays`에 YYYY-MM-DD 문자열로 넣으면 반영된다. 기본값은 공휴일 없음).
  * 연 단위 시간 = 달력일수 / 365 (ACT/365). 변동성과 할인 모두 이 시간축을 쓴다.
  * 조기상환 지급일 = 평가일 + 3영업일, 만기 지급일 = 설명서의 만기일(2029-09-10).
  * 발행사 신용위험, 헤지비용, 세금은 고려하지 않는다 (설명서도 이론가에 헤지비용을 제외한다고 명시).
"""
from __future__ import annotations  # 전방참조 타입 힌트

from dataclasses import dataclass, field  # 불변 계약/격자 컨테이너
from datetime import date  # 계약 날짜

import numpy as np  # 날짜 연산(datetime64, busday_*) 및 벡터화 페이오프


@dataclass(frozen=True)
class ELSTerms:
    """ELS 계약 조건. 금액은 액면 1.0 대비 비율로 표현한다 (예: 1.0725 = 액면의 107.25%)."""

    name: str  # 종목명
    underlyings: tuple[str, ...]  # 기초자산 이름 (시뮬레이터/시장 데이터의 자산 순서와 같아야 한다)
    notional: float  # 1증권당 액면가 (원) - 결과를 원화로 환산할 때만 사용
    initial_fixing_date: date  # 최초기준가격평가일 (시뮬레이션 t=0)
    autocall_dates: tuple[date, ...]  # 자동조기상환평가일 (오름차순)
    autocall_barriers: tuple[float, ...]  # 차수별 조기상환 배리어 (최초기준가격 대비 비율)
    autocall_payouts: tuple[float, ...]  # 차수별 상환금액 (액면 대비 비율)
    final_obs_date: date  # 만기평가일 = 최종관찰일
    maturity_payment_date: date  # 만기 지급일
    maturity_barrier: float  # 만기 상환 배리어 (70%)
    maturity_payout: float  # 만기 상환금액 (187%)
    ki_barrier: float  # 낙인(KI) 배리어 (30%)
    payment_lag_bd: int = 3  # 조기상환 지급 지연 (영업일)
    holidays: tuple[str, ...] = field(default_factory=tuple)  # 휴장일 (YYYY-MM-DD)


@dataclass(frozen=True)
class ELSSchedule:
    """시뮬레이션 격자와 이벤트 인덱스. 격자의 0번 원소가 최초기준가격평가일이다."""

    dates: np.ndarray  # 영업일 격자 (datetime64[D])
    times: np.ndarray  # 각 격자점의 연 단위 시간 (ACT/365), times[0] = 0
    autocall_steps: tuple[int, ...]  # 조기상환평가일의 격자 인덱스
    autocall_pay_times: np.ndarray  # 조기상환 지급일의 연 단위 시간 (할인용)
    final_step: int  # 최종관찰일의 격자 인덱스
    maturity_pay_time: float  # 만기 지급일의 연 단위 시간 (할인용)


def build_schedule(terms: ELSTerms) -> ELSSchedule:
    """계약 날짜들로부터 영업일 격자와 이벤트 인덱스를 만든다."""
    hol = list(terms.holidays)  # numpy busday 함수에 넘길 휴장일 목록
    d0 = np.datetime64(terms.initial_fixing_date, "D")  # t=0 날짜
    d_end = np.datetime64(terms.final_obs_date, "D")  # 마지막 관찰 날짜
    days = np.arange(d0, d_end + 1, dtype="datetime64[D]")  # 달력일 전체
    dates = days[np.is_busday(days, holidays=hol)]  # 영업일만 남긴 격자
    if dates[0] != d0 or dates[-1] != d_end:  # 시작/종료일이 영업일이 아니면 계약 날짜 오류
        raise ValueError("initial fixing date and final observation date must be business days")
    index_of = {d: i for i, d in enumerate(dates)}  # 날짜 -> 격자 인덱스 사전
    times = (dates - d0).astype(float) / 365.0  # ACT/365 연 단위 시간
    steps = []  # 조기상환일 인덱스
    pay_times = []  # 조기상환 지급일 시간
    for d in terms.autocall_dates:  # 조기상환일마다
        dn = np.datetime64(d, "D")  # numpy 날짜로 변환
        if dn not in index_of:  # 영업일이 아니면 계약 오류 (휴장일 설정 확인 필요)
            raise ValueError(f"autocall date {d} is not a business day")
        steps.append(index_of[dn])  # 격자 인덱스 저장
        pay = np.busday_offset(dn, terms.payment_lag_bd, roll="forward", holidays=hol)  # 평가일 + 3영업일
        pay_times.append(float((pay - d0).astype(float)) / 365.0)  # 지급일의 연 단위 시간
    mat = np.datetime64(terms.maturity_payment_date, "D")  # 만기 지급일
    return ELSSchedule(
        dates=dates,
        times=times,
        autocall_steps=tuple(steps),
        autocall_pay_times=np.array(pay_times),
        final_step=index_of[d_end],
        maturity_pay_time=float((mat - d0).astype(float)) / 365.0,
    )


def autocall_redeems(worst_perf: np.ndarray, barrier: float) -> np.ndarray:
    """조기상환 조건: 모든 기초자산이 배리어 이상 <=> 최악 성과(worst-of)가 배리어 이상."""
    return worst_perf >= barrier  # 불리언 배열


def maturity_payout(worst_perf: np.ndarray, ki_hit: np.ndarray, terms: ELSTerms) -> np.ndarray:
    """만기 상환금액(액면 대비 비율). 조기상환되지 않은 경로에 대해서만 의미가 있다.

    (12) 만기 성과 >= 70%                 -> 187%
    (13) 위가 아니고 KI 미발생            -> 187%
    (14) 위가 아니고 KI 발생              -> 만기 성과 (액면 x worst 성과, 손실률 = 성과 - 1)
    """
    full = (worst_perf >= terms.maturity_barrier) | (~ki_hit)  # (12) 또는 (13)에 해당하는 경로
    return np.where(full, terms.maturity_payout, worst_perf)  # 해당 없으면 (14): worst 성과 그대로 상환


def mirae_38078_terms(holidays: tuple[str, ...] = ()) -> ELSTerms:
    """미래에셋증권 제38078회 ELS의 계약 조건 (투자설명서 p.142, p.145-148)."""
    ac_dates = (  # 자동조기상환평가일 11회 (p.145)
        date(2026, 12, 7), date(2027, 3, 5), date(2027, 6, 7), date(2027, 9, 7),
        date(2027, 12, 7), date(2028, 3, 7), date(2028, 6, 5), date(2028, 9, 5),
        date(2028, 12, 5), date(2029, 3, 6), date(2029, 6, 4),
    )
    barriers = (0.90,) * 4 + (0.85,) * 4 + (0.80,) * 3  # 스텝다운 배리어 (p.146)
    payouts = tuple(1.0 + 0.0725 * k for k in range(1, 12))  # 차수 k의 상환금 = 1 + 7.25% x k
    return ELSTerms(
        name="Mirae Asset Securities ELS No.38078",
        underlyings=("KOSPI200", "Samsung Electronics"),
        notional=10_000.0,
        initial_fixing_date=date(2026, 9, 10),
        autocall_dates=ac_dates,
        autocall_barriers=barriers,
        autocall_payouts=payouts,
        final_obs_date=date(2029, 9, 5),
        maturity_payment_date=date(2029, 9, 10),
        maturity_barrier=0.70,
        maturity_payout=1.87,
        ki_barrier=0.30,
        payment_lag_bd=3,
        holidays=holidays,
    )
