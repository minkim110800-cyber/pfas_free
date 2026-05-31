% MATLAB 기반 최적화 알고리즘(Bayesian vs Grid) 검증 시각화 스크립트 (글자 겹침 수정본)
clear; clc; close all;

% 💡 한글 깨짐 방지를 위한 폰트 설정
set(0, 'DefaultAxesFontName', 'Malgun Gothic');

% 전체 도화지(Figure) 생성 및 크기 조절
fig = figure('Name', '알고리즘 신뢰성 검증 결과', 'Position', [100, 100, 1100, 480]);

%% 📈 그래프 1: 생산성(Max Safe RPM) 정량 비교 바 차트
subplot(1, 2, 1);

% 6개 변수 동일 조건 하의 실전 데이터 매핑
rpm_values = [3059, 3200]; 
labels = {'베이지안 최적화 (20℃)', '고도화된 전수조사 (40℃)'};

b = bar(rpm_values, 0.4, 'FaceColor', 'flat', 'EdgeColor', [0 0 0], 'LineWidth', 1.2);
b.CData(1, :) = [0.91, 0.30, 0.24]; % Red
b.CData(2, :) = [0.18, 0.80, 0.44]; % Green

% 💡 개선 1: 가독성을 위해 X축 폰트 크기를 10으로 살짝 조정하여 겹침 방지
set(gca, 'XTickLabel', labels, 'FontSize', 10, 'FontWeight', 'bold');
ylabel('최대 안전 운전 속도 (RPM)', 'FontSize', 12, 'FontWeight', 'bold');
title('6개 변수 통일 조건 하의 공정 생산성 비교', 'FontSize', 13, 'FontWeight', 'bold');

% 💡 개선 2: 상단 텍스트 박스가 들어갈 공간 확보를 위해 Y축 상한을 4,500으로 상향
ylim([0, 4500]); 
grid on;
ax = gca; ax.XGrid = 'off'; ax.YGrid = 'on'; ax.GridLineStyle = '--'; ax.GridAlpha = 0.4;

% 바 바로 위에 꽂히는 정량 수치 표시 (Y축 오프셋을 +80으로 슬림하게 조정)
text(1, rpm_values(1) + 80, [num2str(rpm_values(1)), ' RPM'], 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);
text(2, rpm_values(2) + 80, [num2str(rpm_values(2)), ' RPM'], 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11);

% 💡 개선 3: 무작위로 늘어나던 annotation을 제거하고, 두 바 사이(X=1.5, Y=3850)에 고정된 데이터 텍스트 박스 배치
improvement = ((3200 - 3059) / 3059) * 100;
imp_str = sprintf('공정 속도 약 %.1f%% 추가 향상', improvement);
text(1.5, 3850, imp_str, ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
     'FontWeight', 'bold', 'FontSize', 10.5, 'Color', [0.15, 0.68, 0.38], ...
     'EdgeColor', [0.18, 0.80, 0.44], 'BackgroundColor', [0.91, 0.97, 0.94], ...
     'LineWidth', 1.2, 'Margin', 6);


%% 📉 그래프 2: 알고리즘 탐색 매커니즘 거동 비교 (수렴 곡선)
subplot(1, 2, 2);
iterations = 1:50;

% 6개 변수 환경의 베이지안 수렴 특성 재모사 (3059 RPM 타겟)
rng(42); 
bayesian_exploration = 500 + 2559 * (1 - exp(-0.15 * iterations)) + randn(1, 50) * 12;
bayesian_exploration(bayesian_exploration > 3059) = 3059; 

grid_exploration = ones(1, 50) * 3200;

plot(iterations, bayesian_exploration, '--', 'Color', [0.91, 0.30, 0.24], 'LineWidth', 2.5);
hold on;
plot(iterations, grid_exploration, '-', 'Color', [0.18, 0.80, 0.44], 'LineWidth', 3);

% 수렴선 가이드 주입 (글자 겹침 방지를 위해 Y축 오프셋 하향 조정)
yline(3059, ':', 'Color', [0.75, 0.22, 0.17], 'LineWidth', 1.5);
text(30, 2750, 'Local Optimum (3,059 RPM)', 'Color', [0.75, 0.22, 0.17], 'FontWeight', 'bold', 'FontSize', 10);

title('최적점 탐색 알고리즘 거동 특성 비교 (수렴 곡선)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('탐색 횟수 / 격자 탐색 진행 (Iterations)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('발견된 최대 안전 RPM', 'FontSize', 12, 'FontWeight', 'bold');
ylim([500, 4000]);
grid on;
ax2 = gca; ax2.GridLineStyle = '--'; ax2.GridAlpha = 0.4;

legend({'Bayesian Optimization (3,059 RPM 수렴 한계)', 'Grid Search (3,200 RPM 글로벌 최적점)'}, ...
       'Location', 'southeast', 'FontSize', 10, 'Box', 'on');