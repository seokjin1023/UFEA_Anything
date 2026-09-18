# Extendible Option (연장형 옵션)이란 무엇인가

## 1. 정의

**Extendible option**은 만기(maturity)가 고정되어 있지 않고, 특정 조건이 충족되면
보유자(holder-extendible) 혹은 발행자(writer-extendible)가 옵션의 만기를
한 번(혹은 여러 번) 더 뒤로 늘릴 수 있는 권리가 내재된 옵션이다.
가장 널리 인용되는 형태는 Longstaff (1990, *Journal of Finance*, "Pricing of
Options with Extendible Maturities")가 정식화한 **holder-extendible call**로,
구조는 다음과 같다.

- 최초 만기 `T1`, 행사가 `K1`인 콜옵션을 보유한다.
- `T1` 시점에 기초자산 가격이 `S(T1) = S1`이라 하자.
  - `S1 >= K1` (내가격, ITM)이면: 보유자는 그냥 행사해서 `S1 - K1`을 받는다.
  - `S1 < K1` (외가격, OTM — 원래대로라면 옵션이 그냥 소멸)이면: 보유자는
    **추가 프리미엄 `a`를 지불**하고 만기를 `T2 (> T1)`, 행사가 `K2`
    (통상 `K2 = K1`이거나 조금 높게 재설정)인 **새로운 콜옵션으로 교체(연장)**할
    권리를 가진다. 물론 그 권리를 행사하지 않고 옵션을 포기(0)할 수도 있다.

수식으로 `T1` 시점의 가치(=payoff)는

```
V(T1, S1) =  S1 - K1                                             ,  S1 >= K1
          =  max( C_BS(S1, K2, r, q, σ, T2 - T1) - a ,  0 )       ,  S1 <  K1
```

여기서 `C_BS(S1, K2, r, q, σ, τ)`는 잔존만기 `τ = T2 - T1`, 행사가 `K2`인
블랙-숄즈 유러피언 콜의 시간 `T1` 시점 가치(=continuation value)다.

옵션의 현재가치는 이를 무위험이자율로 할인한 위험중립 기대값이다.

```
Price(0) = e^{-r T1} · E^Q[ V(T1, S1) ]
```

## 2. 왜 "복합 옵션(Compound Option)"인가

Compound option은 "옵션을 기초자산으로 하는 옵션"이다 (Geske, 1979).
가장 표준적인 형태는 **call-on-call**: 만기 `T1`, 행사가 `a`인 콜옵션이 있고,
그 콜의 행사 시 받는 것은 현금이 아니라 "만기 `T2`, 행사가 `K2`인 또 다른
콜옵션(`C_BS(S1, K2, ..., T2-T1)`)"인 구조다. 페이오프는

```
Compound payoff(T1) = max( C_BS(S1, K2, ..., T2-T1) - a , 0 )
```

Extendible option의 OTM 분기(`S1 < K1`)를 다시 보면 **정확히 이 compound
call의 페이오프와 동일한 형태**임을 알 수 있다. 즉,

- `S1 < K1`일 때: "잔존만기 `T2-T1`짜리 콜옵션(K2)을 프리미엄 `a`에 살 수
  있는 권리" = **콜옵션 위에 쓰여진 콜옵션 = compound call**.
- `S1 >= K1`일 때: 그냥 즉시 행사(vanilla call의 intrinsic value)이므로
  compound 구조가 필요 없다.

따라서 **extendible option = "바닐라 콜옵션" + "OTM 영역에서만 살아있는
compound call(연장권)"**로 분해할 수 있는 합성 상품이며, 학술적으로도
Longstaff(1990)의 extendible option 가격공식은 Geske(1979)의 compound
option 공식과 동일한 수학적 장치(2차원 표준정규 결합분포, bivariate normal
CDF)를 사용해 유도된다. 다시 말해:

> **Extendible option은 "언제 compound option이 되는가?"에 대한 답이 아니라,
> 그 자체가 (조건부로 활성화되는) compound option의 한 특수 사례다.**
> 연장 여부에 대한 결정 자체가 "옵션에 대한 옵션"이기 때문이다.

## 3. 왜 이 상품이 "고차 그릭스(Higher-order Greeks)"의 좋은 예시인가 (Taleb, *Dynamic Hedging* Ch.2 문맥)

Taleb의 *Dynamic Hedging* 2장은 옵션의 위험을 1차 민감도(Delta)만으로
설명하려는 접근이 얼마나 위험한 단순화인지를 다룬다. 표준 바닐라 옵션조차
델타만으로는 곡률(감마)에서 오는 손익을 설명하지 못하는데, extendible /
compound 옵션은 payoff 자체에 **이산적인 의사결정 경계**가 내재되어 있어
이 문제가 더 두드러진다. 실제로 이 상품의 `T1` 시점 payoff에는 경계가
두 개 있다 (자세한 계산은 `extend_abandon_boundary()` 참고).

- `K1` : 즉시행사 vs "연장을 고려" 사이의 경계
- `S*` (`< K1`): OTM 구간 안에서, "연장" vs "포기(0)" 사이의 경계
  (연장 시 받는 콜옵션의 블랙-숄즈 가치가 딱 연장프리미엄 `a`와 같아지는
  스팟)

`scripts/demo_taleb_greeks.py`로 실제 계산해보면 다음이 확인된다.

- **Gamma/Vomma의 절대 크기 자체는 바닐라 옵션과 비슷하게 "날개(wing)"
  쪽에서 커지는 경향을 보인다** (ATM 근처에서 작고 깊은 ITM/OTM에서
  커지는 흔한 패턴). 즉 "결정 경계에서만 특별히 폭발적으로 커진다"고
  단순화하기는 어렵다.
- 그러나 이 상품 고유의 뚜렷한 구조적 지문은 **Vanna의 부호 반전**이다:
  `K1` 바로 아래(연장을 고려하는 영역)에서는 Vanna가 양수(+)이지만, `K1`을
  넘어서는 순간 음수(-)로 뒤집힌다. 이는 "행사냐 연장이냐"라는 이산적
  선택이 만들어내는 kink가 실제로 오늘 시점 가격함수의 곡률 구조를
  바꾼다는 직접적 증거다.
- **가장 중요한 정량적 증거는 손익(P&L) 분해다.** 스팟/변동성에 작은
  충격을 준 뒤 "진짜 재평가 손익"과 "1차(Delta+Vega)만 쓴 테일러 근사",
  "2차(+Gamma/Vanna/Vomma)까지 쓴 테일러 근사"를 비교하면, 세 지점
  (두 경계에서 먼 지점, `S*` 부근, `K1` 부근) 모두에서 **1차 근사는
  실제 손익의 상당 부분(테스트한 예시에서 7~40%)을 설명하지 못하지만,
  2차 근사를 쓰면 오차가 90% 이상 줄어든다.**

즉 "델타를 0으로 맞추는 델타 헤지만으로는 extendible/compound 옵션의
리스크를 관리할 수 없고, 감마·바나·보마 같은 고차 그릭스를 능동적으로
관리해야 한다"는 Taleb의 논지가, 정성적으로는 Vanna의 부호 반전으로,
정량적으로는 손익 분해 실험으로 재현된다.

이 저장소의 코드(`extendible_pricing/`, `scripts/demo_taleb_greeks.py`)는
바로 이 실험을 몬테카를로 시뮬레이션으로 직접 수행한다: 두 결정 경계
(`S*`, `K1`) 부근에서 스팟을 스캔하며 Delta, Gamma, Vanna, Vomma를
계산하고, 실제 가격 변화(reprice)와 "1차만 사용한 예측" vs "2차까지
사용한 예측"의 오차를 비교한다.

## 4. 참고사항 — 가격결정 방법론 개요

자세한 구현 설명(Forward simulation을 쓰는 이유, backward/nested MC와의
비교)은 코드 주석과 `extendible_pricing/extendible_option.py` 모듈
docstring, 그리고 `scripts/demo_taleb_greeks.py`를 참고한다. 요약하면:

- 2단계(`T1`, `T2`)의 만기 구조를 갖지만, `T1`에서의 "연장 여부" 판단에
  필요한 **연속가치(continuation value)가 블랙-숄즈 폐형식으로 이미 알려져
  있으므로**, 별도의 회귀 기반 backward induction(예: Longstaff–Schwartz
  Least-Squares Monte Carlo)이 필요하지 않다.
- 대신 `S1`까지만 **forward로 한 번** 시뮬레이션하고, 그 노드에서
  블랙-숄즈 폐형식(=risk-neutral 하에서 backward로 미리 풀어놓은 해)을
  대입하는 "forward simulation + 해석적 continuation value" 방식이
  정확하면서도 계산 비용이 훨씬 낮다.
- 코드에는 이 지름길이 정말로 정확한지 검증하기 위해 (i) `T1→T2`까지도
  내부(nested) 시뮬레이션으로 continuation value를 직접 근사하는 완전
  시뮬레이션 버전과, (ii) 로그정규분포에 대한 수치적분(quadrature) 버전을
  함께 구현하여 세 방법의 가격이 서로 수렴함을 대조한다.
