import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# [STEP 1] 데이터셋 생성 로직 (기존 부록 코드 기반 요약)
substances_info = {
    'Novec 4200': {'gamma': 17.0, 'theta': 115.0},
    'Triton BG-10': {'gamma': 28.0, 'theta': 75.0},
    'Triton CG-50': {'gamma': 27.0, 'theta': 78.0},
    'Brij S100': {'gamma': 56.0, 'theta': 65.0},
    'DI-03': {'gamma': 72.0, 'theta': 70.0}
} # [cite: 212, 213, 214, 215, 216, 217]

NUM_SAMPLES = 20000 # [cite: 218]
np.random.seed(42) # [cite: 219]

temp = np.random.uniform(20, 90, NUM_SAMPLES) # [cite: 221]
rpm = np.random.uniform(500, 10000, NUM_SAMPLES) # [cite: 222]
sub_names = np.random.choice(list(substances_info.keys()), NUM_SAMPLES) # [cite: 223]

gamma_base = np.array([substances_info[name]['gamma'] for name in sub_names]) # [cite: 225]
theta = np.array([substances_info[name]['theta'] for name in sub_names]) # [cite: 226, 228]

# 물리 엔진 연산 및 붕괴 판정 [cite: 229]
gamma_eff = gamma_base - 0.1 * (temp - 25.0) # [cite: 232]
P_cap = np.abs((2 * (gamma_eff * 1e-3) * np.cos(np.radians(theta))) / 20e-9) * 1e-6 # [cite: 235]
P_rpm = 0.00000008 * (rpm**2) # [cite: 236]
P_total = P_cap + P_rpm # [cite: 237]

damage_factor = 1.0 - 0.15 * np.random.randn(NUM_SAMPLES) # [cite: 239, 240]
P_limit = 5.0 * damage_factor # [cite: 241]
collapse = (P_total > P_limit).astype(int) # [cite: 242]

df = pd.DataFrame({
    'Temp': temp, 'RPM': rpm, 'Gamma': gamma_base, 'Theta': theta, 'Collapse': collapse
}) # [cite: 244, 245, 246, 247, 248, 249]

# ========================================================
# 💡 김민준 역할 핵심: 데이터 분리 및 다중 지표 검증 체계 도입
# ========================================================

# 1. 독립변수(X)와 종속변수(y) 설정 [cite: 254, 255]
X = df[['Temp', 'RPM', 'Gamma', 'Theta']] # [cite: 254]
y = df['Collapse'] # [cite: 255]

# 2. Train 데이터와 Test 데이터를 8:2 비율로 분리
# stratify=y 옵션은 학습과 테스트셋의 붕괴/안전 데이터 비율을 균일하게 맞춰줌
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"학습 데이터 개수: {X_train.shape[0]}개 | 테스트 데이터 개수: {X_test.shape[0]}개")

# 3. Random Forest 모델을 '학습 데이터(Train)'로만 학습 [cite: 256]
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # [cite: 256]
rf_model.fit(X_train, y_train)

# 4. '테스트 데이터(Test)'를 투입해 예측 성능 검증
y_pred = rf_model.predict(X_test)

# 5. 교수님 방어용 다각화 성능 지표 산출
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n[✔] Random Forest 검증 결과")
print(f"- Accuracy (전체 예측 정확도)  : {accuracy:.4f}")
print(f"- Precision (붕괴라 예측한 것 중 실제 붕괴) : {precision:.4f}")
print(f"- Recall (실제 붕괴 중 모델이 맞춘 비율)   : {recall:.4f}")
print(f"- F1-Score (정밀도와 재현율의 조화 평균)  : {f1:.4f}")
