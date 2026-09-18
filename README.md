# UFEA_Anything

## Extendible Option과 고차 그릭스(Higher-order Greeks)

Taleb, *Dynamic Hedging* Ch.2의 논지(복합/연장형 옵션에서는 1차 그릭스인
델타보다 고차 그릭스가 손익을 훨씬 크게 좌우한다)를 holder-extendible call
옵션을 예로 들어 몬테카를로 시뮬레이션으로 직접 확인하는 프로젝트다.

- 상품 설명 및 왜 compound option인지: [`docs/extendible_option.md`](docs/extendible_option.md)
- 가격결정/그릭스 구현: [`extendible_pricing/`](extendible_pricing)
  - `black_scholes.py`: 블랙-숄즈 폐형식 (continuation value 및 벤치마크용)
  - `extendible_option.py`: forward-analytic MC / nested-backward MC / lognormal quadrature 세 가지 가격결정 방법. **Forward vs Backward를 어떤 기준으로 선택했는지**는 이 파일의 모듈 docstring에 상세히 설명되어 있다.
  - `greeks.py`: 공통난수법(CRN)을 적용한 유한차분 그릭스 계산 (Delta/Gamma/Vega/Vanna/Vomma/Theta)
- 데모 스크립트: [`scripts/demo_taleb_greeks.py`](scripts/demo_taleb_greeks.py)

### 실행 방법

```bash
pip install -r requirements.txt
python scripts/demo_taleb_greeks.py
```

스크립트는 다음을 출력/생성한다.

1. 세 가지 가격결정 방법의 교차검증 (콘솔 출력)
2. 연장 결정 경계(K1) 부근에서의 Delta/Gamma/Vanna/Vomma 스캔 (콘솔 표 + `outputs/greeks_scan.png`)
3. 실제 스팟/변동성 충격에 대한 "1차 근사 vs 2차(고차) 근사" 손익 비교 (콘솔 출력)
