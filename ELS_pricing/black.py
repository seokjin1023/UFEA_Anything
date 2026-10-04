"""블랙(Black-76 / 배당 있는 BSM) 유러피언 콜 가격.

SABR 내재변동성을 "가격"으로 바꿔야 하는 곳(Dupire 가격식 검증, 차익거래 진단)에서 쓰이는 빌딩 블록이다.
선도가격 F = S0 * exp(drift * T) 로 표현하면 배당수익률 q 는 drift = r - q 에 흡수된다.

## 참고문헌
[1] Black, F. (1976), "The pricing of commodity contracts", Journal of Financial Economics 3, 167-179.
[2] Hull, J. C., Options, Futures, and Other Derivatives (교과서), 블랙-숄즈-머튼 / 선도가격 기반 옵션 공식 장.
"""
from __future__ import annotations  # 전방참조 타입 힌트 허용

import numpy as np  # 벡터화 연산
from scipy.stats import norm  # 표준정규 누적분포 N(.)


def black_call(F, K, T, sigma, r) -> np.ndarray:
    """선도가격 F 기준 유러피언 콜: exp(-rT) [F N(d1) - K N(d2)]."""
    F, K, T, sigma = (np.asarray(v, dtype=float) for v in (F, K, T, sigma))  # 배열화
    sd = sigma * np.sqrt(T)  # 만기까지의 총 표준편차 sigma*sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd  # d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt T)
    d2 = d1 - sd  # d2 = d1 - sigma sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))  # 할인된 기대 페이오프
