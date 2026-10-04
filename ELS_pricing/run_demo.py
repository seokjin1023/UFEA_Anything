"""ELS 가격결정 데모: 투자설명서 제38078회 ELS를 SABR -> Dupire -> 2자산 Cholesky 몬테카를로로 평가한다.

실행: python ELS_pricing/run_demo.py [--paths 100000] [--seed 1] [--no-plots]

단계
  [1] 상품 조건 / 시장 가정 출력
  [2] 자산별 SABR 곡면 보정 (만기별 alpha, rho, nu와 RMSE)  -> 변동성 곡면 그림
  [3] Dupire 국소변동성 격자 생성 + 차익거래/클리핑 진단 + 가격식 기반 구현과의 교차검증 -> 국소변동성 그림
  [4] 몬테카를로 가격결정
        (a) 벤치마크: 설명서의 상수 변동성(36.29% / 47.99%) + 상관 0.9558 (Black-Scholes식 GBM)
        (b) 본 모형: SABR + Dupire 국소변동성 + 상관 0.9558
      그리고 설명서의 이론가 10,551.48원(2026-08-25 기준)과 비교
  [5] 그림 저장 (ELS_pricing/outputs/)

주의: 변동성 호가는 실제 시장 데이터가 아니라 설명서의 ATM 변동성에 맞춘 *합성* 스마일이다 (market_data.py 참고).
      따라서 (b)의 절대 가격은 가정한 스큐에 의존하며, 실제 호가로 바꾸면 달라진다.
"""
from __future__ import annotations  # 전방참조 타입 힌트

import argparse  # 명령행 인자
import os  # 경로 처리
import sys  # sys.path 조정

import numpy as np  # 배열 연산

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 저장소 루트를 import 경로에 추가

from ELS_pricing import (  # noqa: E402  (경로 설정 뒤에 import)
    CorrelatedLocalVolSimulator, DupireLocalVol, ELSPricer, FlatLocalVol, SABRVolSurface,
    build_schedule, local_vol_from_prices, make_synthetic_quotes, mirae_38078_market, mirae_38078_terms,
)
from ELS_pricing import plotting  # noqa: E402  (그림 함수 모음)

PROSPECTUS_FAIR_VALUE = 10_551.48  # 설명서 p.144: 2026-08-25 기준 이론가 (헤지비용 제외)


def main() -> None:
    parser = argparse.ArgumentParser(description="ELS pricing demo (SABR + Dupire + Cholesky MC)")  # 인자 파서
    parser.add_argument("--paths", type=int, default=100_000, help="Monte Carlo paths (even number)")  # 경로 수
    parser.add_argument("--seed", type=int, default=1, help="random seed")  # 난수 시드
    parser.add_argument("--beta", type=float, default=1.0, help="SABR beta (fixed, not calibrated)")  # 고정 beta
    parser.add_argument("--correction", choices=["hagan", "obloj"], default="hagan", help="SABR formula variant")  # 공식 종류
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")  # 그림 생략
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))  # 출력 폴더
    args = parser.parse_args()  # 인자 파싱
    os.makedirs(args.out, exist_ok=True)  # 출력 폴더 생성

    # [1] 상품 조건과 시장 가정 -------------------------------------------------------------------
    terms = mirae_38078_terms()  # 설명서 계약 조건
    sched = build_schedule(terms)  # 영업일 격자와 이벤트 인덱스
    mkt = mirae_38078_market()  # 시장 가정 (설명서 수치 + 데모용 가정)
    print("=" * 78)
    print(f"[1] {terms.name}")
    print(f"    grid: {sched.times.size - 1} business days, final obs t = {sched.times[-1]:.4f}y, "
          f"maturity payment t = {sched.maturity_pay_time:.4f}y")
    print(f"    auto-call barriers: {[int(b * 100) for b in terms.autocall_barriers]}% | maturity 70% -> 187% | KI 30%")
    print(f"    rate r = {mkt.rate:.2%} (assumed) | correlation = {mkt.correlation[0, 1]:.4f} (prospectus)")
    for a in mkt.assets:
        print(f"    {a.name:<22s} S0 = {a.spot:>10,.2f} | drift mu = {a.drift:.2%} (given) | doc ATM vol = {a.atm_vol:.2%}")

    # [2] SABR 곡면 보정 --------------------------------------------------------------------------
    print("=" * 78)
    print(f"[2] SABR calibration (beta = {args.beta} fixed, formula = {args.correction})")
    quotes, surfaces = [], []  # 자산별 호가 / 곡면
    for a in mkt.assets:
        q = make_synthetic_quotes(a)  # 합성 호가 (실제 호가가 있으면 MarketQuotes를 직접 만들어 넣는다)
        s = SABRVolSurface.calibrate(q, beta=args.beta, correction=args.correction)  # 만기별 보정
        quotes.append(q); surfaces.append(s)  # 보관
        print(f"    {a.name}")
        print("      T(y)    alpha     rho      nu    RMSE(vol pts)")
        for row in s.calibration_table():
            print(f"      {row['T']:5.2f}  {row['alpha']:7.4f}  {row['rho']:7.4f}  {row['nu']:7.4f}   {row['rmse_vol_pts']:6.3f}"
                  f"{'' if row['ok'] else '  (!) optimizer did not converge'}")
        cal = s.calendar_arbitrage_fraction(np.linspace(-1.0, 0.5, 40), np.linspace(0.1, 3.0, 40))  # 달력 차익 진단
        print(f"      calendar-spread arbitrage cells: {cal:.2%}")

    # [3] Dupire 국소변동성 ----------------------------------------------------------------------
    print("=" * 78)
    print("[3] Dupire local volatility")
    t_max = float(sched.times[-1]) + 0.02  # 시뮬레이션 마지막 시점보다 약간 길게
    local_vols = [DupireLocalVol(s, t_max) for s in surfaces]  # 자산별 국소변동성 격자
    for a, s, lv in zip(mkt.assets, surfaces, local_vols):
        d = lv.diagnostics()  # 클리핑/차익거래 진단
        print(f"    {a.name}: clipped(min) {d['frac_below_min_vol']:.2%} | clipped(max) {d['frac_above_max_vol']:.2%} | "
              f"negative-density nodes {d['frac_negative_density']:.2%}")
        T_chk, K_chk = 1.0, s.forward(1.0)  # 교차검증 지점: 1년, ATM-forward
        lv_grid = float(lv.local_vol(T_chk, np.array([K_chk]))[0])  # 격자 조회값 (내재분산 방식)
        lv_px = float(local_vol_from_prices(s, K_chk, T_chk, mkt.rate))  # Dupire 원식(가격 유한차분)
        print(f"      check @ T=1y, K=F: implied-variance form {lv_grid:.4f} vs price-based form {lv_px:.4f} "
              f"(SABR implied vol {float(s.implied_vol(K_chk, T_chk)):.4f})")

    # [4] 몬테카를로 가격결정 ---------------------------------------------------------------------
    print("=" * 78)
    print(f"[4] Monte Carlo pricing ({args.paths:,} paths, antithetic, daily steps, seed {args.seed})")
    spots = [a.spot for a in mkt.assets]  # 초기 현물
    drifts = [a.drift for a in mkt.assets]  # 주어진 드리프트
    models = {  # 비교할 두 모형
        "Flat vol (prospectus vols)": [FlatLocalVol(a.atm_vol) for a in mkt.assets],
        "SABR + Dupire local vol": local_vols,
    }
    results = {}  # 모형명 -> 결과
    for name, lvs in models.items():
        sim = CorrelatedLocalVolSimulator(lvs, spots, drifts, mkt.correlation, sched.times)  # 시뮬레이터
        res = ELSPricer(terms, sched, sim, mkt.rate).price(args.paths, args.seed)  # 가격결정
        results[name] = res  # 저장
        lo, hi = res.confidence_interval()  # 95% 신뢰구간
        print(f"    {name}")
        print(f"      price = {res.price_per_certificate(terms.notional):>9,.2f} KRW per 10,000 "
              f"(SE {res.std_error * terms.notional:.2f}, 95% CI [{lo * terms.notional:,.1f}, {hi * terms.notional:,.1f}])")
        print(f"      P(auto-call) = {res.autocall_probs.sum():.2%} | P(maturity 187%) = {res.maturity_full_prob:.2%} | "
              f"P(loss) = {res.loss_prob:.2%} | mean payout | loss = {res.mean_payout_given_loss:.1%} | "
              f"expected life = {res.expected_life:.2f}y")
        print("      auto-call prob by round: " + " ".join(f"{p:.1%}" for p in res.autocall_probs))
    print(f"    prospectus fair value (2026-08-25, ex hedging cost): {PROSPECTUS_FAIR_VALUE:,.2f} KRW per 10,000")

    # [5] 그림 저장 -------------------------------------------------------------------------------
    if not args.no_plots:
        print("=" * 78)
        print(f"[5] Saving figures to {args.out}")
        for q, s in zip(quotes, surfaces):  # 자산별 변동성 곡면
            fname = "vol_surface_" + q.name.lower().replace(" ", "_") + ".png"
            plotting.plot_vol_surface(s, q, os.path.join(args.out, fname))
        plotting.plot_local_vols(local_vols, [a.name for a in mkt.assets], os.path.join(args.out, "local_vol.png"))
        sim = CorrelatedLocalVolSimulator(local_vols, spots, drifts, mkt.correlation, sched.times)  # 그림용 소수 경로
        sample = sim.simulate_paths(40, seed=args.seed)  # 40개 경로 전체 저장
        plotting.plot_worst_of_paths(sched, terms, sample, np.array(spots), os.path.join(args.out, "worst_of_paths.png"))
        plotting.plot_redemption_distribution(results["SABR + Dupire local vol"], terms,
                                              os.path.join(args.out, "redemption_distribution.png"))
        print("    done")


if __name__ == "__main__":
    main()  # 스크립트로 실행될 때만 동작
