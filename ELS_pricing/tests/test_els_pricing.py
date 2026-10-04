"""ELS 가격결정 패키지 검증 테스트 (실행: pytest ELS_pricing/tests -q).

각 테스트는 "정답을 아는 상황"을 만들어 구현이 그 정답을 재현하는지 본다.
"""
import numpy as np  # 배열 연산
import pytest  # 테스트 프레임워크

from ELS_pricing import (
    AssetSpec, CorrelatedLocalVolSimulator, DupireLocalVol, ELSPricer, FlatLocalVol, SABRVolSurface,
    build_schedule, calibrate_sabr_slice, cholesky_factor, local_vol_from_prices, make_synthetic_quotes,
    mirae_38078_market, mirae_38078_terms, sabr_implied_vol,
)
from ELS_pricing.black import black_call  # 블랙 콜 가격
from ELS_pricing.product import maturity_payout  # 만기 페이오프 규칙
from ELS_pricing.sabr import SABRParams, SABRSliceFit  # 수동 곡면 구성용


@pytest.fixture(scope="module")
def market():
    return mirae_38078_market()  # 설명서 수치 + 가정


@pytest.fixture(scope="module")
def schedule():
    return build_schedule(mirae_38078_terms())  # 영업일 격자


@pytest.fixture(scope="module")
def kospi_surface(market):
    return SABRVolSurface.calibrate(make_synthetic_quotes(market.assets[0]))  # KOSPI200 곡면


# ----------------------------- SABR -----------------------------

def test_sabr_atm_continuity_and_closed_form():
    F, T, a, b, r, n = 100.0, 1.0, 0.3, 0.7, -0.4, 0.8  # 임의의 파라미터
    atm = sabr_implied_vol(F, F, T, a, b, r, n)  # K = F
    near = sabr_implied_vol(F, F * (1 + 1e-6), T, a, b, r, n)  # K = F 바로 옆
    closed = a / F ** (1 - b) * (1 + ((1 - b) ** 2 / 24 * a ** 2 / F ** (2 - 2 * b)
                                     + 0.25 * r * b * n * a / F ** (1 - b) + (2 - 3 * r ** 2) / 24 * n ** 2) * T)  # Hagan eq. 2.18
    assert float(atm) == pytest.approx(closed, rel=1e-12)  # ATM 극한식과 정확히 일치
    assert float(near) == pytest.approx(float(atm), rel=1e-4)  # 연속


def test_hagan_and_obloj_agree_for_beta_one_and_are_close_otherwise():
    K = np.linspace(70, 140, 15)  # 행사가 격자
    h1 = sabr_implied_vol(100.0, K, 1.0, 0.3, 1.0, -0.5, 0.7, "hagan")
    o1 = sabr_implied_vol(100.0, K, 1.0, 0.3, 1.0, -0.5, 0.7, "obloj")
    assert np.allclose(h1, o1, atol=1e-10)  # beta = 1 에서는 동일한 식
    h5 = sabr_implied_vol(100.0, K, 1.0, 3.0, 0.5, -0.5, 0.7, "hagan")
    o5 = sabr_implied_vol(100.0, K, 1.0, 3.0, 0.5, -0.5, 0.7, "obloj")
    assert np.max(np.abs(h5 - o5)) < 0.01  # beta < 1 에서도 1 vol pt 이내


@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_calibration_recovers_noiseless_parameters(beta):
    F, T = 100.0, 1.5
    K = F * np.linspace(0.6, 1.5, 12)
    alpha = 0.35 * F ** (1 - beta)  # beta에 맞춘 alpha 스케일
    mkt = sabr_implied_vol(F, K, T, alpha, beta, -0.45, 0.9)  # 노이즈 없는 정답 스마일
    fit = calibrate_sabr_slice(F, T, K, mkt, beta=beta)
    assert fit.rmse < 1e-6  # 스마일을 사실상 완벽히 재현
    assert fit.params.rho == pytest.approx(-0.45, abs=1e-3)
    assert fit.params.nu == pytest.approx(0.9, abs=1e-3)
    assert fit.params.alpha == pytest.approx(alpha, rel=1e-3)


# ----------------------------- Dupire -----------------------------

def test_dupire_flat_surface_gives_flat_local_vol():
    # nu -> 0 이면 SABR(beta=1)은 상수 변동성 alpha (시간 보정항 무시 가능). 국소변동성도 alpha여야 한다.
    slices = [SABRSliceFit(T, 100.0, SABRParams(0.25, 1.0, 0.0, 1e-3), 0.0, True) for T in (0.25, 1.0, 3.0)]
    surf = SABRVolSurface("flat", 100.0, 0.0, 1.0, "hagan", slices)
    lv = DupireLocalVol(surf, 3.0)
    vals = lv.local_vol(1.0, np.array([60.0, 100.0, 150.0]))
    assert np.allclose(vals, 0.25, atol=2e-3)


def test_dupire_implied_variance_form_matches_price_form(market, kospi_surface):
    lv = DupireLocalVol(kospi_surface, 3.02)
    for T in (0.5, 1.0, 2.0):
        for m in (0.8, 1.0, 1.2):
            K = m * float(kospi_surface.forward(T))
            grid = float(lv.local_vol(T, np.array([K]))[0])  # 내재분산 방식 (격자 보간)
            px = float(local_vol_from_prices(kospi_surface, K, T, market.rate))  # 가격 유한차분 방식
            assert grid == pytest.approx(px, rel=0.01)  # 두 독립 구현이 1% 이내로 일치


def test_local_vol_monte_carlo_reprices_sabr_vanillas(market, kospi_surface, schedule):
    """국소변동성 시뮬레이션으로 유러피언 콜을 다시 가격하면 SABR 곡면의 블랙 가격이 나와야 한다 (Dupire 정의)."""
    a = market.assets[0]
    sim = CorrelatedLocalVolSimulator([DupireLocalVol(kospi_surface, 3.02), FlatLocalVol(0.3)],
                                      [a.spot, 100.0], [a.drift, 0.0], np.eye(2), schedule.times)
    k = int(np.argmin(np.abs(schedule.times - 1.0)))  # 약 1년 후 격자
    T = float(schedule.times[k])
    for kk, _, S in sim.iter_steps(100_000, seed=11):
        if kk == k:
            ST = S[0]
            break
    F = float(kospi_surface.forward(T))
    assert ST.mean() / F == pytest.approx(1.0, abs=0.01)  # 위험중립 하에서 선도가격은 마팅게일
    for m in (0.7, 1.0, 1.3):
        K = m * F
        mc = float(np.mean(np.maximum(ST - K, 0.0)) * np.exp(-market.rate * T))
        bs = float(black_call(F, K, T, kospi_surface.implied_vol(K, T), market.rate))
        assert abs(mc - bs) / a.spot < 0.0015  # 스팟의 15bp 이내 (이산화 편향 포함)


# ----------------------------- 시뮬레이션 -----------------------------

def test_cholesky_factor_and_validation():
    corr = np.array([[1.0, 0.9558], [0.9558, 1.0]])
    L = cholesky_factor(corr)
    assert np.allclose(L @ L.T, corr)  # R = L L^T
    assert np.allclose(L, np.tril(L))  # 하삼각
    with pytest.raises(ValueError):
        cholesky_factor(np.array([[1.0, 0.5], [0.4, 1.0]]))  # 비대칭
    with pytest.raises(np.linalg.LinAlgError):
        cholesky_factor(np.array([[1.0, 1.2], [1.2, 1.0]]))  # 양의 정부호 아님


def test_simulated_log_return_correlation_and_moments():
    times = np.array([0.0, 1.0])  # 한 스텝 = 1년 (상수 변동성이면 정확한 분포)
    rho = 0.9558
    sim = CorrelatedLocalVolSimulator([FlatLocalVol(0.4), FlatLocalVol(0.2)], [100.0, 50.0], [0.03, 0.01],
                                      np.array([[1, rho], [rho, 1.0]]), times)
    S = sim.simulate_paths(200_000, seed=5)[-1]  # (2, n) 1년 후
    lr = np.log(S / np.array([[100.0], [50.0]]))  # 로그수익률
    assert np.corrcoef(lr)[0, 1] == pytest.approx(rho, abs=0.005)  # 상관 재현
    assert lr[0].std() == pytest.approx(0.4, rel=0.01) and lr[1].std() == pytest.approx(0.2, rel=0.01)  # 변동성 재현
    assert S[0].mean() == pytest.approx(100.0 * np.exp(0.03), rel=0.005)  # E[S] = S0 e^{mu T}


def test_antithetic_requires_even_paths(market, schedule):
    sim = CorrelatedLocalVolSimulator([FlatLocalVol(0.3)] * 2, [1.0, 1.0], [0.0, 0.0], market.correlation, schedule.times)
    with pytest.raises(ValueError):
        next(sim.iter_steps(101, antithetic=True))


# ----------------------------- 상품 / 페이오프 -----------------------------

def test_schedule_matches_prospectus(schedule):
    terms = mirae_38078_terms()
    assert len(schedule.autocall_steps) == 11 and len(terms.autocall_barriers) == 11
    assert [round(p, 4) for p in terms.autocall_payouts[:2] + terms.autocall_payouts[-1:]] == [1.0725, 1.145, 1.7975]
    assert schedule.times[0] == 0.0 and np.all(np.diff(schedule.times) > 0)
    assert np.all(schedule.autocall_pay_times > schedule.times[list(schedule.autocall_steps)])  # 지급은 평가일 이후
    assert schedule.maturity_pay_time > schedule.times[schedule.final_step]  # 만기 지급일이 최종관찰일 이후


def test_maturity_payout_rules():
    terms = mirae_38078_terms()
    worst = np.array([0.80, 0.80, 0.50, 0.50, 0.10])  # 만기 worst 성과
    ki = np.array([False, True, False, True, True])  # KI 터치 여부
    out = maturity_payout(worst, ki, terms)
    assert np.allclose(out, [1.87, 1.87, 1.87, 0.50, 0.10])  # (12) / (12) / (13) / (14) / (14)


# ----------------------------- 가격결정기 (결정론적 시나리오) -----------------------------

def _pricer(schedule, market, mu, vol=1e-9):
    sim = CorrelatedLocalVolSimulator([FlatLocalVol(vol)] * 2, [1.0, 1.0], [mu, mu], market.correlation, schedule.times)
    return ELSPricer(mirae_38078_terms(), schedule, sim, market.rate)


def test_deterministic_first_round_autocall(market, schedule):
    res = _pricer(schedule, market, mu=0.30).price(1_000, seed=0)  # 변동성 0, 큰 양의 드리프트: 1차에서 전부 상환
    expected = 1.0725 * np.exp(-market.rate * schedule.autocall_pay_times[0])
    assert res.price == pytest.approx(expected, rel=1e-6)
    assert res.autocall_probs[0] == 1.0 and res.loss_prob == 0.0


def test_deterministic_knock_in_loss(market, schedule):
    res = _pricer(schedule, market, mu=-0.5).price(1_000, seed=0)  # 변동성 0, 큰 음의 드리프트: 상환 없이 KI, 만기 손실
    worst_T = np.exp(-0.5 * schedule.times[schedule.final_step])  # 만기 worst 성과 (약 22%)
    assert res.autocall_probs.sum() == 0.0 and res.loss_prob == 1.0
    assert res.price == pytest.approx(worst_T * np.exp(-market.rate * schedule.maturity_pay_time), rel=1e-6)


def test_flat_vol_price_close_to_prospectus_fair_value(market, schedule):
    """설명서 이론가(10,551.48)와 같은 전제(상수 변동성 + 상관 0.9558)에서 몇 % 이내로 맞는지 확인하는 현실성 점검."""
    sim = CorrelatedLocalVolSimulator([FlatLocalVol(a.atm_vol) for a in market.assets],
                                      [a.spot for a in market.assets], [a.drift for a in market.assets],
                                      market.correlation, schedule.times)
    res = ELSPricer(mirae_38078_terms(), schedule, sim, market.rate).price(50_000, seed=2)
    assert res.price * 10_000 == pytest.approx(10_551.48, rel=0.03)  # r, q는 가정치이므로 3% 허용
