% ==============================================================================
% [MATLAB] PFAS-Free 5종 약액 웨이퍼 반경별 패턴 응력 분포 (논리 오류 교정본)
% ==============================================================================
clear; clc; close all;

d_pattern = 20e-9;       
r_max = 0.15;            
radius_vec = linspace(0, r_max, 500); 

% [핵심 수정]: 장비 고유 상수를 최신 팹(Fab) 저진동 설비 스펙으로 현실화
k_constant = 1.2e-7;     % (기존 2.5e-6에서 수정)
vib_constant = 1.0e-8;   % 미세 진동 패널티 상수 신설

chem_names = {'Novec 4200', 'Triton BG-10', 'Triton CG-50', 'Brij S100', 'DI-O3'};
opt_temp   = [60.0, 62.0, 63.0, 70.0, 40.0];  
opt_rpm    = [4400, 4400, 4600, 3500, 3200];  
gamma_base = [17.0, 28.0, 27.0, 56.0, 72.0];  
theta_deg  = [115.0, 75.0, 78.0, 65.0, 70.0]; 
viscosity_base = [1.02, 1.15, 1.20, 1.50, 1.00];  % ◀ 이 줄을 똑같이 타이핑해서 넣어라!

f1 = figure('Name', 'Wafer Radial Stress Distribution', 'Position', [100, 100, 850, 550], 'Color', 'w');
hold on; grid on;

colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880];
line_styles = {'-', '--', '-.', ':', '-'};

for i = 1:length(chem_names)
    T = opt_temp(i);
    RPM = opt_rpm(i);
    g_base = gamma_base(i);
    th = theta_deg(i);
    v_base = viscosity_base(i);  % ◀ [1단계 복구]: 위에서 만든 배열을 순서대로 호출!
    
    gamma_eff = g_base - 0.1 * (T - 25.0);
    phi_viscosity = v_base * exp(100 / (T + 273.15)); % ◀ [2단계 복구]: 1.02 지우고 v_base로 연산!
    
    % 모세관 압력 (단위 변환 유지)
    p_cap_pa = abs((2 * (gamma_eff * 1e-3) * cos(deg2rad(th))) / d_pattern);
    p_cap = p_cap_pa * phi_viscosity * 1e-6; 
    
    % 원심 압력 및 진동 (현실화된 장비 상수 적용, 불필요한 1e-6 축소 제거)
    p_rpm = (radius_vec / r_max) * (k_constant * (RPM^2));
    p_vibration = (radius_vec / r_max) * (vib_constant * (RPM^2));
    
    p_total = p_cap + p_rpm + p_vibration;
    
    p_upper = p_total * 1.15;
    p_lower = p_total * 0.85;
    
    x_fill = [radius_vec*1000, fliplr(radius_vec*1000)];
    y_fill = [p_upper, fliplr(p_lower)];
    
    fill(x_fill, y_fill, colors(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(radius_vec * 1000, p_total, 'LineWidth', 2.5, 'Color', colors(i,:), 'LineStyle', line_styles{i}, 'DisplayName', chem_names{i});
end

limit_line = 5.0;
line([0, r_max*1000], [limit_line, limit_line], 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-', 'HandleVisibility', 'off');
text(5, limit_line + 0.3, 'Critical Collapse Limit (5.0 MPa)', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 11);

xlabel('Wafer Radius (mm)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Total Applied Stress, P_{total} (MPa)', 'FontSize', 12, 'FontWeight', 'bold');
title('Radial Pattern Stress Distribution (Physics Corrected)', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

xlim([0 150]);
ylim([0 6]);
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'TickDir', 'out', 'FontName', 'Arial');

lgd = legend('Location', 'northwest', 'FontSize', 11);
title(lgd, 'Chemical Solution');

hold off;
