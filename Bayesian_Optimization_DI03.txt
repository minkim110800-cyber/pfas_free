import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

# 💡 [원인 차단] 주피터 세션 꼬임 방지를 위해 딕셔너리 구조를 소문자 key로 명확히 고정
substances_info = {
    'Novec 4200': {'gamma': 17.0, 'theta': 115.0, 'viscosity': 1.02},
    'Triton BG-10': {'gamma': 28.0, 'theta': 75.0, 'viscosity': 1.15},
    'Triton CG-50': {'gamma': 27.0, 'theta': 78.0, 'viscosity': 1.20},
    'Brij S100':    {'gamma': 56.0, 'theta': 65.0, 'viscosity': 1.50},
    'DI-03':        {'gamma': 72.0, 'theta': 70.0, 'viscosity': 1.0}
}

target_substance = 'DI-03'
info = substances_info[target_substance]

# 1. 탐색 공간 정의 (Grid Search와 완벽히 동일한 DI-O3 제약 조건 설정)
search_space = [
    Integer(20, 40, name='Temp'),
    Integer(500, 6000, name='RPM')
]

# 2. 베이지안 최적화용 목적 함수 정의
@use_named_args(search_space)
def bayesian_objective(Temp, RPM):
    # AI 모델 투입용 6차원 입력 벡터 구축 (Worst-case 반지름 150mm 고정)
    input_data = pd.DataFrame([{
        'Temp': float(Temp),
        'RPM': float(RPM),
        'Gamma': info['gamma'],
        'Theta': info['theta'],
        'Viscosity': info['viscosity'], # 위에서 정의한 소문자 key와 완벽 매칭
        'Radius': 150.0
    }])

    # 앞서 학습 완료한 고도화된 AI 모델(Gradient Boosting)로 붕괴 예측 (0: Safe, 1: Collapse)
    pred = best_model.predict(input_data[['Temp', 'RPM', 'Gamma', 'Theta', 'Viscosity', 'Radius']])[0]

    if pred == 1:
        # 패턴이 붕괴하면 최악의 점수(0) 반환
        return 0.0
    else:
        # 안전하면 RPM을 극대화하기 위해 최소화 문제용 음수 변환
        return -float(RPM)

print(f"[진행 중] {target_substance} 기준 6개 변수 환경 베이지안 최적화 탐색 시작 (50회 샘플링)...")

# 3. 가우시안 프로세스 기반 베이지안 최적화 실행
res = gp_minimize(
    bayesian_objective,
    dimensions=search_space,
    n_calls=50,
    random_state=42,
    n_initial_points=10
)

# 4. 결과 도출 및 출력
opt_temp = res.x[0]
max_bayesian_rpm = -res.fun

print("\n=======================================================")
print("[✔] 동일 조건(6개 변수) 하의 베이지안 최적화 최종 결과")
print("=======================================================")
print(f"- 대상 약액 물질: {target_substance}")
print(f"- AI 도출 최적 온도: {opt_temp} °C")
print(f"- AI 도출 최대 안전 RPM: {int(max_bayesian_rpm)} RPM")
print("=======================================================")
