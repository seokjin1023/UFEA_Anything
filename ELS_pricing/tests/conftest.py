"""pytest 설정: 저장소 루트를 import 경로에 추가해 `import ELS_pricing`이 되도록 한다."""
import os  # 경로 처리
import sys  # import 경로 조정

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # tests -> ELS_pricing -> 저장소 루트
