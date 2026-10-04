"""시각화: SABR 변동성 곡면, Dupire 국소변동성, Worst-of 경로, 상환 분포.

모든 그림은 같은 규칙을 따른다 (dataviz 원칙).
  * 자산 구분 색은 고정: KOSPI200 = 파랑(#2a78d6), 삼성전자 = 주황(#eb6834). 값의 크기(변동성)는 한 가지 색조의 명암.
  * 격자선/축은 연하게, 텍스트는 잉크 색(시리즈 색을 글자에 쓰지 않는다), 상/우 테두리 제거, 시리즈가 둘 이상이면 범례.
  * 라벨은 영어 (기본 matplotlib 폰트에는 한글 글리프가 없어 깨지므로).
  * `matplotlib.use("Agg")`로 화면 없이 PNG 파일만 저장한다.
"""
from __future__ import annotations  # 전방참조 타입 힌트

import matplotlib  # 백엔드 지정을 위해 먼저 import

matplotlib.use("Agg")  # GUI 없이 파일로만 그린다 (서버/CI 환경 대응)
import matplotlib.pyplot as plt  # 플로팅 API
import numpy as np  # 배열 연산
from matplotlib.colors import LinearSegmentedColormap  # 단일 색조 순차 컬러맵

from .dupire import DupireLocalVol  # 국소변동성 격자
from .market_data import MarketQuotes  # 시장 호가 (점으로 겹쳐 그림)
from .pricer import PricingResult  # 상환 분포 그림용 결과
from .product import ELSSchedule, ELSTerms  # 배리어 / 날짜
from .vol_surface import SABRVolSurface  # SABR 곡면

SURFACE = "#fcfcfb"  # 차트 면 색
INK = "#0b0b0b"  # 주 텍스트
INK_2 = "#52514e"  # 보조 텍스트
GRID = "#e4e3de"  # 연한 격자선
BLUE = "#2a78d6"  # 시리즈 1: KOSPI200
ORANGE = "#eb6834"  # 시리즈 2: Samsung Electronics
ASSET_COLORS = {"KOSPI200": BLUE, "Samsung Electronics": ORANGE}  # 이름 -> 고정 색
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#dbe9fb", "#7fb0ec", "#2a78d6", "#123f7c"])  # 밝음 -> 진함


def _style() -> None:
    """전역 matplotlib 스타일을 한 번에 설정한다."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,  # 배경
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,  # 글자색
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,  # 축/격자
        "axes.spines.top": False, "axes.spines.right": False,  # 상/우 테두리 제거
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,  # 글꼴
    })


def plot_vol_surface(surface: SABRVolSurface, quotes: MarketQuotes, path: str) -> None:
    """SABR 내재변동성 곡면(3D)과 만기별 스마일 적합(시장 호가 vs 모형)을 한 장에 저장한다."""
    _style()  # 스타일 적용
    m = np.linspace(quotes.fwd_moneyness[0], quotes.fwd_moneyness[-1], 60)  # K/F 격자
    T = np.linspace(quotes.expiries[0], quotes.expiries[-1], 40)  # 만기 격자
    MM, TT = np.meshgrid(m, T)  # 2D 격자
    F = surface.forward(TT)  # 만기별 선도가격
    vol = surface.implied_vol(MM * F, TT)  # SABR 내재변동성 (K = m F)
    fig = plt.figure(figsize=(12.5, 5.2))  # 큰 캔버스
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")  # 왼쪽: 3D 곡면
    ax3.plot_surface(MM, TT, vol * 100, cmap=SEQ_BLUE, linewidth=0, antialiased=True, alpha=0.95)  # 곡면
    mq, tq = np.meshgrid(quotes.fwd_moneyness, quotes.expiries)  # 호가 격자 좌표
    ax3.scatter(mq.ravel(), tq.ravel(), quotes.vols.ravel() * 100, color=INK, s=8, depthshade=False)  # 시장 호가 점
    ax3.set_xlabel("K / F_T"); ax3.set_ylabel("Maturity (yrs)"); ax3.set_zlabel("Implied vol (%)")  # 축 라벨
    ax3.set_title(f"{quotes.name}: SABR implied-vol surface (dots = quotes)")  # 제목
    ax3.view_init(elev=24, azim=-58)  # 보기 각도
    ax2 = fig.add_subplot(1, 2, 2)  # 오른쪽: 스마일 슬라이스
    shades = SEQ_BLUE(np.linspace(0.15, 1.0, len(quotes.expiries)))  # 만기가 길수록 진한 같은 색조
    for i, Ti in enumerate(quotes.expiries):  # 만기마다
        Fi = float(surface.forward(Ti))  # 선도가격
        line_vol = surface.implied_vol(m * Fi, Ti)  # 모형 스마일
        ax2.plot(m, line_vol * 100, color=shades[i], lw=2, label=f"T = {Ti:.2f}y")  # 모형 곡선
        ax2.scatter(quotes.fwd_moneyness, quotes.vols[i] * 100, color=shades[i], s=14, zorder=3,
                    edgecolor=SURFACE, linewidth=0.8)  # 시장 호가 점
    ax2.set_xlabel("K / F_T"); ax2.set_ylabel("Implied vol (%)")  # 축 라벨
    ax2.set_title("Smile fit by expiry (lines = SABR, dots = quotes)")  # 제목
    ax2.legend(ncol=2, fontsize=9)  # 만기 범례
    fig.tight_layout()  # 여백 정리
    fig.savefig(path, dpi=130)  # 저장
    plt.close(fig)  # 메모리 해제


def plot_local_vols(local_vols: list[DupireLocalVol], names: list[str], path: str) -> None:
    """자산별 Dupire 국소변동성 히트맵 (가로: 시간, 세로: S / F_t)."""
    _style()  # 스타일 적용
    fig, axes = plt.subplots(1, len(local_vols), figsize=(6.2 * len(local_vols), 4.8), sharey=True)  # 자산당 한 패널
    axes = np.atleast_1d(axes)  # 자산이 1개여도 반복 가능하도록
    for ax, lv, name in zip(axes, local_vols, names):  # 패널별로
        sel = (lv.x_grid >= np.log(0.3)) & (lv.x_grid <= np.log(2.0))  # 보여줄 구간: S/F in [0.3, 2.0]
        im = ax.pcolormesh(lv.t_grid, np.exp(lv.x_grid[sel]), lv.lv[:, sel].T * 100, cmap=SEQ_BLUE,
                           vmin=20, vmax=100, shading="auto")  # 변동성(%)을 명암으로
        ax.set_xlabel("Time t (yrs)"); ax.set_title(f"{name}: Dupire local vol")  # 라벨/제목
        fig.colorbar(im, ax=ax, label="Local vol (%), clipped at 100", extend="max")  # 컬러바
    axes[0].set_ylabel("S / F_t")  # 공통 y 라벨
    fig.tight_layout()  # 여백 정리
    fig.savefig(path, dpi=130)  # 저장
    plt.close(fig)  # 메모리 해제


def plot_worst_of_paths(
    schedule: ELSSchedule, terms: ELSTerms, paths: np.ndarray, spots: np.ndarray, path: str,
) -> None:
    """Worst-of 성과 경로 샘플과 조기상환 배리어(계단) / 만기 배리어 / KI 배리어."""
    _style()  # 스타일 적용
    worst = (paths / spots[None, :, None]).min(axis=1) * 100  # (n_steps+1, n_paths) 퍼센트 단위 worst 성과
    fig, ax = plt.subplots(figsize=(10, 5))  # 한 패널
    ax.plot(schedule.times, worst, color=BLUE, lw=0.7, alpha=0.35)  # 샘플 경로(얇고 반투명)
    ac_t = schedule.times[list(schedule.autocall_steps)]  # 조기상환평가일의 시간
    obs_t = np.append(ac_t, schedule.times[schedule.final_step])  # 관찰일 시간 (조기상환일 11개 + 만기평가일)
    obs_b = np.append(np.array(terms.autocall_barriers) * 100, terms.maturity_barrier * 100)  # 각 관찰일의 배리어(%)
    ax.step(obs_t, obs_b, where="post", color=INK, lw=1.6,  # 배리어는 해당 관찰일부터 다음 관찰일까지 유지되는 계단
            label="Redemption barrier (step-down, 70% at maturity)")
    ax.plot(obs_t, obs_b, "o", color=INK, ms=4)  # 실제 관찰일은 점으로 표시 (배리어는 이 날에만 적용된다)
    ax.axhline(terms.ki_barrier * 100, color=INK_2, lw=1.4, ls="--", label="Knock-in barrier (30%)")  # KI 선
    ax.set_xlabel("Time (yrs)"); ax.set_ylabel("Worst-of performance (% of initial)")  # 축 라벨
    ax.set_title("Simulated worst-of paths (local-vol model)")  # 제목
    ax.legend(loc="upper right")  # 범례
    fig.tight_layout()  # 여백 정리
    fig.savefig(path, dpi=130)  # 저장
    plt.close(fig)  # 메모리 해제


def plot_redemption_distribution(result: PricingResult, terms: ELSTerms, path: str) -> None:
    """상환 시점별 확률 막대 (조기상환 차수별 / 만기 187% / 원금 손실)."""
    _style()  # 스타일 적용
    labels = [f"#{k}" for k in range(1, len(result.autocall_probs) + 1)] + ["Mat.\n187%", "Loss\n(<100%)"]  # 막대 라벨
    probs = list(result.autocall_probs) + [result.maturity_full_prob, result.loss_prob]  # 막대 값
    fig, ax = plt.subplots(figsize=(10, 4.6))  # 한 패널
    bars = ax.bar(labels, np.array(probs) * 100, color=BLUE, width=0.62)  # 단일 시리즈 막대
    for b, p in zip(bars, probs):  # 막대 위에 값 직접 표기
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{p * 100:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Probability (%)"); ax.set_xlabel("Autocall round / maturity outcome")  # 축 라벨
    ax.set_title(f"Redemption distribution (price {result.price * terms.notional:,.0f} KRW per 10,000)")  # 제목
    ax.grid(axis="x", visible=False)  # x 격자 제거
    fig.tight_layout()  # 여백 정리
    fig.savefig(path, dpi=130)  # 저장
    plt.close(fig)  # 메모리 해제
