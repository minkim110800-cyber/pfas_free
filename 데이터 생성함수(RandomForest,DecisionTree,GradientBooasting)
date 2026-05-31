import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ========================================================
# 💡 [피드백 2, 3번 반영] 유효 점도(viscosity) 및 신규 변수 추가 정의
# ========================================================
substances_info = {
    'Novec 4200': {'gamma': 17.0, 'theta': 115.0, 'viscosity': 1.02},     # [cite: 61, 212]
    'Triton BG-10': {'gamma': 28.0, 'theta': 75.0, 'viscosity': 1.15},   # [cite: 61, 212]
    'Triton CG-50': {'gamma': 27.0, 'theta': 78.0, 'viscosity': 1.20},   # [cite: 61, 213]
    'Brij S100': {'gamma': 56.0, 'theta': 65.0, 'viscosity': 1.50},      # [cite: 61, 216]
    'DI-03': {'gamma': 72.0, 'theta': 70.0, 'viscosity': 1.0}            # [cite: 61, 217]
}

NUM_SAMPLES = 20000
np.random.seed(42)

temp = np.random.uniform(20, 90, NUM_SAMPLES) # [cite: 221]
rpm = np.random.uniform(500, 10000, NUM_SAMPLES) # [cite: 222]
sub_names = np.random.choice(list(substances_info.keys()), NUM_SAMPLES) # [cite: 223]

# 💡 신규 변수: 웨이퍼 반지름 위치(r)를 0~150mm 사이의 확률적 변수로 생성
radius = np.random.uniform(0, 150, NUM_SAMPLES)

gamma_base = np.array([substances_info[name]['gamma'] for name in sub_names]) # [cite: 225]
theta = np.array([substances_info[name]['theta'] for name in sub_names]) # [cite: 226]
# 💡 신규 변수: 물질별 유효 점도 데이터 매핑
viscosity_base = np.array([substances_info[name]['viscosity'] for name in sub_names])

# [물리 엔진 업그레이드]
gamma_eff = gamma_base - 0.1 * (temp - 25.0) # [cite: 232]
P_cap = np.abs((2 * (gamma_eff * 1e-3) * np.cos(np.radians(theta))) / 20e-9) * 1e-6 # [cite: 235]

# 💡 피드백 반영 수식 고도화: P_rpm에 실제 반지름 변수(r / r_max) 동역학 반영
P_rpm = (radius / 150.0) * 0.00000008 * (rpm**2) # [cite: 69, 236]

# 💡 피드백 반영 수식 고도화: 유효 점도에 따른 유체 전단 응력(Shear Stress) 패널티 추가
P_viscous = viscosity_base * (rpm / 10000.0) * (radius / 150.0) * 0.15

# 최종 통합 압력 연산
P_total = P_cap + P_rpm + P_viscous

damage_factor = 1.0 - 0.15 * np.random.randn(NUM_SAMPLES) # [cite: 239]
P_limit = 5.0 * damage_factor # [cite: 241]
collapse = (P_total > P_limit).astype(int) # [cite: 242]

# 💡 변수가 6개로 늘어난 고차원 공정 데이터프레임 구축
df = pd.DataFrame({
    'Temp': temp, 'RPM': rpm, 'Gamma': gamma_base, 'Theta': theta,
    'Viscosity': viscosity_base, 'Radius': radius, 'Collapse': collapse
})

# 독립변수(X)에 신규 변수 포함 총 6개 지정
X = df[['Temp', 'RPM', 'Gamma', 'Theta', 'Viscosity', 'Radius']]
y = df['Collapse']

# Train / Test 세트 8:2 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================================
# 💡 [피드백 4번 반영] 확장된 데이터셋 위에서 다중 모델 재검증
# ========================================================
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

benchmark_results = []

print("[진행 중] 고도화된 데이터셋 기반 모델별 학습 및 검증 수행...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    benchmark_results.append({
        'Model': name, 'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
        'Recall': round(rec, 4), 'F1-Score': round(f1, 4)
    })

perf_df = pd.DataFrame(benchmark_results)
print("\n=======================================================")
print("[✔] 최종 고도화 결과: 6개 변수 기준 모델 성능 비교 벤치마크")
print("=======================================================")
print(perf_df.to_string(index=False))
print("=======================================================")
