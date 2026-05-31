import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

# [STEP 1] 실제 데이터 기반 물질 특성 정의
substances_info = {
    'Novec 4200': {'gamma': 17.0, 'theta': 115.0, 'viscosity': 1.02},
    'Triton BG-10': {'gamma': 28.0, 'theta': 75.0, 'viscosity': 1.15},
    'Triton CG-50': {'gamma': 27.0, 'theta': 78.0, 'viscosity': 1.20},
    'Brij S100':    {'gamma': 56.0, 'theta': 65.0, 'viscosity': 1.50},
    'DI-03':        {'gamma': 72.0, 'theta': 70.0, 'viscosity': 1.0}
}

NUM_SAMPLES = 20000
np.random.seed(42)

temp = np.random.uniform(20, 90, NUM_SAMPLES)
rpm = np.random.uniform(500, 6000, NUM_SAMPLES)
radius = np.random.uniform(0, 150, NUM_SAMPLES)

sub_names = np.random.choice(list(substances_info.keys()), NUM_SAMPLES)
gamma_base = np.array([substances_info[name]['gamma'] for name in sub_names])
theta = np.array([substances_info[name]['theta'] for name in sub_names])
viscosity_base = np.array([substances_info[name]['viscosity'] for name in sub_names])

# [물리 엔진 업그레이드]
gamma_eff = gamma_base - 0.1 * (temp - 25.0)
P_cap = np.abs((2 * (gamma_eff * 1e-3) * np.cos(np.radians(theta))) / 20e-9) * 1e-6
P_rpm = (radius / 150.0) * 0.00000008 * (rpm**2)
P_viscous = viscosity_base * (rpm / 6000.0) * (radius / 150.0) * 0.15

# ========================================================
# 💡 김민준의 칩샷: 실제 장비 기계적 진동 및 슬립 리스크 페널티 추가
# ========================================================
# 4,500 RPM 부근부터 기하급수적으로 증가하는 모터 구조 진동 응력 구현
P_vibration = 0.00000013 * (rpm ** 2) * (radius / 150.0)

# 최종 통합 압력 (패턴 붕괴 압력 + 유동 불균일 점도 응력 + 설비 진동 스트레스)
P_total = P_cap + P_rpm + P_viscous + P_vibration

damage_factor = 1.0 - 0.15 * np.random.randn(NUM_SAMPLES)
P_limit = 5.0 * damage_factor
collapse = (P_total > P_limit).astype(int)

df = pd.DataFrame({
    'Temp': temp, 'RPM': rpm, 'Gamma': gamma_base, 'Theta': theta,
    'Viscosity': viscosity_base, 'Radius': radius, 'Collapse': collapse
})

X = df[['Temp', 'RPM', 'Gamma', 'Theta', 'Viscosity', 'Radius']]
y = df['Collapse']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best_model = GradientBoostingClassifier(random_state=42)
best_model.fit(X_train, y_train)

# ========================================================
# 💡 [3~4단계] 장비 제약 반영된 AI 모델 기반 Grid Search 전수 탐색
# ========================================================
process_constraints = {
    'Novec 4200':   {'min_temp': 20, 'max_temp': 90},
    'Triton BG-10': {'min_temp': 20, 'max_temp': 90},
    'Triton CG-50': {'min_temp': 20, 'max_temp': 90},
    'Brij S100':    {'min_temp': 55, 'max_temp': 90},
    'DI-03':        {'min_temp': 20, 'max_temp': 40}
}

final_recipes = []

for name, info in substances_info.items():
    limits = process_constraints[name]

    temp_range = np.arange(limits['min_temp'], limits['max_temp'] + 1, 1)
    rpm_range = np.arange(500, 6001, 100)

    T, R = np.meshgrid(temp_range, rpm_range)

    grid_df = pd.DataFrame({
        'Temp': T.flatten(),
        'RPM': R.flatten(),
        'Gamma': info['gamma'],
        'Theta': info['theta'],
        'Viscosity': info['viscosity'],
        'Radius': 150.0
    })

    grid_df['Collapse_Pred'] = best_model.predict(grid_df[['Temp', 'RPM', 'Gamma', 'Theta', 'Viscosity', 'Radius']])
    safe_zone = grid_df[grid_df['Collapse_Pred'] == 0]

    if not safe_zone.empty:
        best_recipe = safe_zone.sort_values(by='RPM', ascending=False).iloc[0]
        final_recipes.append({
            'Substance': name,
            'Opt_Temp (°C)': int(best_recipe['Temp']),
            'Max_Safe_RPM': int(best_recipe['RPM']),
            'Status': 'Success'
        })
    else:
        final_recipes.append({
            'Substance': name, 'Opt_Temp (°C)': '-', 'Max_Safe_RPM': '-', 'Status': 'Failed'
        })

recipe_df = pd.DataFrame(final_recipes)
print("\n=======================================================")
print("[✔] 장비 기계적 제약(진동/슬립) 최종 반영 최적 공정 레시피")
print("=======================================================")
print(recipe_df.to_string(index=False))
print("=======================================================")
