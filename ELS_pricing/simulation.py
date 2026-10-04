"""2자산 국소변동성 몬테카를로 시뮬레이터 (Cholesky 분해로 상관 반영).

## 모형
각 자산 i = 1, 2는 자신의 Dupire 국소변동성으로 움직이고, 브라운운동만 상관계수 rho로 묶인다.

    dS_i / S_i = mu_i dt + sigma_i(S_i, t) dW_i,        d<W_1, W_2> = rho dt

  * mu_i  : 위험중립 드리프트 (r - q_i). 이 프로젝트에서는 "이미 주어진 값".
  * sigma_i: dupire.py의 국소변동성. 자산별 SABR 곡면에서 따로 유도된다.
  * rho   : 설명서의 일별수익률 상관계수 0.9558. (SLV가 아닌 국소변동성 모형이므로 상관은 상수로 둔다.)

## 상관 난수 생성 (Cholesky 분해)
독립 표준정규 Z = (Z_1, Z_2)^T 와 상관행렬 R = L L^T (L은 하삼각)에 대해 W = L Z 는 상관행렬 R을 갖는다.
2자산의 경우 W_1 = Z_1,  W_2 = rho Z_1 + sqrt(1 - rho^2) Z_2 이다. 자산 수가 늘어도 코드는 그대로다.

## 이산화
로그 오일러(log-Euler) 스킴: 한 스텝 동안 sigma를 시작 시점 값으로 고정하면

    ln S_{t+dt} = ln S_t + (mu - sigma^2 / 2) dt + sigma sqrt(dt) (L Z)

이고, 이는 sigma가 일정하다면 정확한(exact) 전이이므로 S가 항상 양수이고 이산화 편향은 sigma의 변화에서만
생긴다. 스텝은 상품의 영업일 격자(일별)를 그대로 쓰므로 조기상환/낙인 관찰일과 정확히 맞는다.

## 메모리
전체 경로를 저장하지 않고 제너레이터로 한 스텝씩 내보낸다 (경로 10만 개 x 영업일 약 750일이어도 상태는
(자산 수 x 경로 수) 배열 하나뿐). 그림용으로 소수 경로 전체가 필요하면 `simulate_paths`를 쓴다.

## 분산 감소
대조변수(antithetic variates): 난수 Z와 -Z를 짝지어 쓰면 단조에 가까운 페이오프의 분산이 줄어든다.
경로 배열의 앞 절반이 Z, 뒤 절반이 -Z 이다 (pricer.py가 이 짝 구조를 이용해 표준오차를 계산한다).

## 참고문헌
[1] Glasserman, P. (2004), Monte Carlo Methods in Financial Engineering, Springer.
    - Sec. 2.3.3: Cholesky 분해로 다변량 정규 생성,  Ch. 3: 확산과정 이산화(오일러 / 로그-오일러),
      Sec. 4.2: 대조변수법.
[2] Shreve, S. E. (2004), Stochastic Calculus for Finance II, Springer, Ch. 4:
    상관된 브라운운동의 Cholesky 표현.
[3] Dupire, B. (1994), "Pricing with a Smile", Risk 7(1). (dupire.py 참조)
"""
from __future__ import annotations  # 전방참조 타입 힌트

from typing import Iterator, Sequence  # 제너레이터/시퀀스 타입

import numpy as np  # 난수와 선형대수

from .dupire import LocalVolModel  # 국소변동성 공통 인터페이스


def cholesky_factor(corr: np.ndarray) -> np.ndarray:
    """상관행렬 R의 하삼각 Cholesky 인자 L (R = L L^T). 대칭/양의 정부호가 아니면 오류."""
    corr = np.asarray(corr, dtype=float)  # 배열화
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:  # 정방행렬 확인
        raise ValueError("correlation matrix must be square")
    if not np.allclose(corr, corr.T):  # 대칭성 확인
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(corr), 1.0):  # 대각 성분이 1인지 확인
        raise ValueError("correlation matrix must have unit diagonal")
    return np.linalg.cholesky(corr)  # 양의 정부호가 아니면 LinAlgError 발생


class CorrelatedLocalVolSimulator:
    """자산별 국소변동성 + 상관 브라운운동으로 영업일 격자 위 경로를 생성한다."""

    def __init__(
        self,
        local_vols: Sequence[LocalVolModel],
        spots: Sequence[float],
        drifts: Sequence[float],
        corr: np.ndarray,
        times: np.ndarray,
    ) -> None:
        n = len(local_vols)  # 자산 수
        if not (len(spots) == len(drifts) == n and np.shape(corr) == (n, n)):  # 입력 차원 일관성 검사
            raise ValueError("local_vols, spots, drifts and corr must agree on the number of assets")
        self.local_vols = list(local_vols)  # 자산별 국소변동성 모형
        self.spots = np.asarray(spots, dtype=float)  # 초기 현물 (n,)
        self.drifts = np.asarray(drifts, dtype=float)  # 위험중립 드리프트 (n,)
        self.chol = cholesky_factor(corr)  # 상관 구조를 담은 Cholesky 인자
        self.times = np.asarray(times, dtype=float)  # 연 단위 시간 격자 (times[0] = 0)

    @property
    def n_assets(self) -> int:
        return self.spots.size  # 자산 수

    def iter_steps(
        self, n_paths: int, seed: int = 0, antithetic: bool = True,
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """(스텝 인덱스 k, 시간 t_k, 현물 배열 S[k] shape (n_assets, n_paths))를 순서대로 내보낸다."""
        if antithetic and n_paths % 2:  # 대조변수는 경로를 짝으로 쓰므로 짝수여야 한다
            raise ValueError("n_paths must be even when antithetic=True")
        rng = np.random.default_rng(seed)  # 재현 가능한 난수 발생기
        n_draw = n_paths // 2 if antithetic else n_paths  # 실제로 뽑을 독립 난수 열 수
        S = np.repeat(self.spots[:, None], n_paths, axis=1)  # 모든 경로가 S0에서 시작 (n_assets, n_paths)
        yield 0, float(self.times[0]), S  # k = 0: 초기 상태
        for k in range(1, self.times.size):  # 격자의 각 스텝
            t = float(self.times[k - 1])  # 스텝 시작 시각 (국소변동성 평가 시점)
            dt = float(self.times[k] - self.times[k - 1])  # 스텝 길이 (주말은 3일치)
            sigma = np.stack([lv.local_vol(t, S[i]) for i, lv in enumerate(self.local_vols)])  # sigma_i(S_i, t)
            z = rng.standard_normal((self.n_assets, n_draw))  # 독립 표준정규 Z
            if antithetic:  # 대조변수: Z 뒤에 -Z를 이어 붙인다
                z = np.concatenate([z, -z], axis=1)
            dW = (self.chol @ z) * np.sqrt(dt)  # 상관된 브라운 증분 L Z sqrt(dt)
            drift = (self.drifts[:, None] - 0.5 * sigma ** 2) * dt  # 로그 드리프트 (mu - sigma^2/2) dt
            S = S * np.exp(drift + sigma * dW)  # 로그-오일러: S_{k} = S_{k-1} exp(...)
            yield k, float(self.times[k]), S  # 새 상태를 내보낸다

    def simulate_paths(self, n_paths: int, seed: int = 0, antithetic: bool = True) -> np.ndarray:
        """전체 경로 배열 shape (n_steps + 1, n_assets, n_paths). 그림/디버깅용 (경로 수가 작을 때만)."""
        out = np.empty((self.times.size, self.n_assets, n_paths))  # 결과 버퍼
        for k, _, S in self.iter_steps(n_paths, seed, antithetic):  # 제너레이터를 소비하며
            out[k] = S  # 각 스텝을 저장
        return out
