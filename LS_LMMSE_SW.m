clear; clc; close all;
%% ========== 1. 参数设置 ==========
M       = 1;              % 波导数量 (Number of waveguides, KEEP FIXED for simplicity)
N_total_list = 8:4:32;    % <<< MODIFIED: List of total antenna numbers to test (8, 12, ..., 32)
M_test  = 1000;           % Monte Carlo 次数
SNR_dB_list = -20:5:20;   % SNR 范围 (SNR range)

% 城市环境 & 波导参数 (Urban environment & waveguide parameters)
c       = 3e8;
fc      = 28e9;
lambda  = c/fc;
lambda_g = lambda/1.4; % Wavelength inside waveguide (approximation)
d       = 5;           % Waveguide height/position component
Dx      = 20;          % Environment dimension X
Dy      = 20;          % Environment dimension Y (less relevant if M=1)
building_height = 10;
P_ref   = 3;           % Number of reflection paths considered
P_sca   = 3;           % Number of scattering paths considered

% Rician 因子 (Rician factor)
Ricain_K_dB = 10;
Ricain_K    = 10^(Ricain_K_dB/10);

% 天线间隔 (Antenna spacing)
delta   = lambda/2; % Increased spacing

% 存储 NMSE (2D Storage for N_total vs SNR) - Only LS and LMMSE
nmse_LS_avg    = zeros(length(N_total_list), length(SNR_dB_list));
nmse_LMMSE_avg = zeros(length(N_total_list), length(SNR_dB_list));
% In this version, OMP-based nmse_CS_avg is removed

%% ========== 2. 主循环 (OUTER: N_total, INNER: SNR) ==========
for iN_total = 1:length(N_total_list)
    N_total = N_total_list(iN_total);

    if mod(N_total, M) ~= 0
        error('N_total (%d) must be divisible by M (%d)', N_total, M);
    end
    N = N_total / M;      % Antennas per waveguide for this iteration

    T = N_total;          % Sequential probing: Time slots = Total antennas
    % K (Sparsity parameter) is removed

    fprintf('\n--- Running for N_total = %d (N=%d per WG, M=%d, T=%d) ---\n', N_total, N, M, T);

    % --- Generate the switching matrix W for this N_total ---
    W = eye(N_total); % Sequential switching matrix (N_total x T) where T=N_total
    W_H = W';         % Transpose is also identity (T x N_total)

    for iSNR = 1:length(SNR_dB_list)
        SNR_dB = SNR_dB_list(iSNR);
        SNR_linear = 10^(SNR_dB/10);
        sigma2 = 1 / SNR_linear; % Noise variance (assuming signal power is normalized)

        % --- Initialize storage for this specific N_total and SNR ---
        nmse_ls_all    = zeros(M_test, 1);
        nmse_lmmse_all = zeros(M_test, 1);
        % nmse_cs_all is removed

        % Store data for LMMSE calculation later
        h_k_collection = zeros(N_total, M_test); % True channel vectors
        y_collection   = zeros(M*T, M_test);     % Observed signals
        A_collection   = cell(M_test, 1);        % Measurement matrices

        % --- 第一阶段：生成数据并存储 (Phase 1: Generate data and preliminary estimates) ---
        for trial = 1:M_test
            % (a) 用户位置 (User position) - Random per trial
            xm = Dx * rand;
            ym = Dy * (rand - 0.5) * (M>1); % Y position only matters if M>1
            zm = 0; % Assuming user on the ground
            psi_m = [xm, ym, zm];

            % (b) 波导和天线位置 (Waveguide and antenna positions)
            y_waveguides = 0; % Only one waveguide at y=0 for M=1
            psi_p = cell(M, 1);
            antenna_indices_global = zeros(N_total, 1);
            current_idx = 1;

            for m = 1:M % Loop executes once if M=1
                loc = 0:delta:(Dx*1.5); % Potential antenna locations
                if length(loc) < N
                     error('N=%d, but only %d locations possible with delta=%.3f, Dx=%.1f', N, length(loc), delta, Dx);
                end
                possible_indices = 1:length(loc);
                if length(possible_indices) < N
                     error('Cannot select %d unique locations from %d options.', N, length(possible_indices));
                end
                random_int = randperm(length(loc), N);
                random_int = sort(random_int);
                x_p = loc(random_int)';
                y_p = y_waveguides(m) * ones(N, 1);
                z_p = d * ones(N, 1);
                psi_p{m} = [x_p, y_p, z_p];
                antenna_indices_global(current_idx : current_idx + N - 1) = current_idx : current_idx + N - 1;
                current_idx = current_idx + N;
            end

            % (c) 生成信道 h_k 和 g (Generate channel h_k and g)
            h_k = zeros(N_total, 1);
            g_prop = zeros(N_total, 1); % Waveguide propagation factor
            G = zeros(M, N_total);      % Matrix mapping antenna signals to waveguide outputs

            for m = 1:M % Loop executes once if M=1
                psi_p_m = psi_p{m};
                psi_0_p = [0, y_waveguides(m), d]; % Waveguide start (port)
                idx_m_start = (m-1)*N + 1;
                idx_m_end   = m*N;
                ant_indices_m = idx_m_start:idx_m_end;

                h_k_m = zeros(N, 1);
                for n = 1:N % Loop through antennas within the waveguide
                    psi_p_mn = psi_p_m(n,:);
                    % --- LoS Component ---
                    dist_los = norm(psi_m - psi_p_mn);
                    if dist_los < 1e-6, dist_los = 1e-6; end
                    h_LoS = (lambda / (4*pi*dist_los)) * exp(-1j*2*pi*dist_los / lambda);
                    % --- Reflected Component ---
                    h_ref = 0;
                    for p_ = 1:P_ref
                        psi_p_mn_img = psi_p_mn;
                        psi_p_mn_img(3) = -psi_p_mn_img(3);
                        dist_ref = norm(psi_m - psi_p_mn_img);
                        if dist_ref < 1e-6, dist_ref = 1e-6; end
                        reflection_coeff = -0.7 / p_;
                        h_ref = h_ref + reflection_coeff * (lambda / (4*pi*dist_ref)) * exp(-1j*2*pi*dist_ref / lambda);
                    end
                    % --- Scattered Component ---
                    h_sca = 0;
                    for p_ = 1:P_sca
                        scatter_psi = [Dx * rand, Dy * (rand - 0.5) * (M>1), building_height * rand];
                        dist1 = norm(scatter_psi - psi_p_mn);
                        dist2 = norm(scatter_psi - psi_m);
                        total_dist = dist1 + dist2;
                         if total_dist < 1e-6, total_dist = 1e-6; end
                         h_sca = h_sca + (1/(p_^0.5)) * (lambda / (4*pi*total_dist)) * exp(-1j*2*pi*total_dist / lambda) * exp(-1j*rand*2*pi);
                    end
                    % --- Rician Combination with LoS Probability (5G mmWave standard) ---
                    distance_LoS = dist_los;
                    if distance_LoS <= 1.2
                        p_LOS = 1;
                    elseif distance_LoS < 6.5 && distance_LoS > 1.2 
                        p_LOS = exp(-(distance_LoS-1.2)/4.7); % Example decay model
                    else
                        p_LOS = 0.32 * exp(-(distance_LoS-6.5)/32.6); % Example decay model
                    end
                    I_LOS = randsrc(1,1,[1,0; p_LOS, 1-p_LOS]); % Randomly include LoS path

                    h_LoS_part = sqrt(Ricain_K / (Ricain_K + 1)) * h_LoS;
                    h_NLOS_part = sqrt(1 / (Ricain_K + 1)) * (h_ref + h_sca);
                    h_k_m(n) = I_LOS * h_LoS_part + h_NLOS_part;

                     if abs(h_k_m(n)) < 1e-15 % Add noise if channel is zero
                         h_k_m(n) = (randn(1,1) + 1j*randn(1,1)) * 1e-9;
                     end
                end
                h_k(ant_indices_m) = h_k_m;

                % g_m (Waveguide propagation factor for waveguide m)
                g_m = zeros(N, 1);
                for n = 1:N
                    dist_n = norm(psi_p_m(n,:) - psi_0_p);
                    atten_dB_per_m = 0.1;
                    atten_linear = 10^(-atten_dB_per_m * dist_n / 10);
                    g_m(n) = atten_linear * exp(-1j*2*pi*dist_n / lambda_g);
                end
                g_prop(ant_indices_m) = g_m;
                G(m, ant_indices_m) = g_m.';
            end

            % Channel Normalization
            current_norm_sq = norm(h_k)^2;
            if current_norm_sq > 1e-12
                 h_k = h_k * sqrt(N_total / current_norm_sq);
            else
                 warning('Trial %d @ N_total=%d, SNR=%.1f: Channel norm near zero.', trial, N_total, SNR_dB);
                 h_k = (randn(N_total,1) + 1j*randn(N_total,1)) * sqrt(N_total * 1e-12);
            end
            h_k_collection(:, trial) = h_k;

            % (e) 多波导观测 (Multi-waveguide observation)
            y_ideal = zeros(M*T, 1);
            A = zeros(M*T, N_total); % Measurement matrix A
            for t = 1:T % T = N_total
                w_t = W(:, t);
                y_t_ideal = G * diag(w_t) * h_k;
                y_ideal((t-1)*M + 1 : t*M) = y_t_ideal;
                A_block_t = G * diag(w_t);
                A((t-1)*M + 1 : t*M, :) = A_block_t;
            end
            noise = sqrt(sigma2/2) * (randn(M*T, 1) + 1j*randn(M*T, 1));
            y = y_ideal + noise;
            y_collection(:, trial) = y;
            A_collection{trial} = A;

            % (f) LS 估计 (LS Estimation)
            try
                h_hat_LS = pinv(A) * y; % Use pseudo-inverse for robustness
                nmse_LS = norm(h_hat_LS - h_k)^2 / norm(h_k)^2;
                if isnan(nmse_LS) || isinf(nmse_LS), nmse_LS = 1; end
            catch ME_LS
                fprintf('Error in LS @ N_total=%d, SNR=%.1f, Trial=%d: %s\n', N_total, SNR_dB, trial, ME_LS.message);
                nmse_LS = 1;
            end
            nmse_ls_all(trial) = nmse_LS;

            % (g) CS Estimation REMOVED

        end % End of Monte Carlo trial loop (Phase 1)


        % --- 第二阶段：计算 R_h_MMSE 并重新估计 LMMSE ---
        % (Phase 2: Calculate R_h_MMSE and re-estimate LMMSE)

        R_h_MMSE = cov(h_k_collection.'); % Estimate covariance

        % Add regularization for stability
        reg_factor = 1e-9 * mean(abs(diag(R_h_MMSE))); % Use abs for potentially complex diags
        if isnan(reg_factor) || isinf(reg_factor) || reg_factor < 1e-12
            reg_factor = 1e-9;
        end
         if cond(R_h_MMSE) > 1e12 || rcond(R_h_MMSE) < 1e-12
              fprintf('Regularizing R_h_MMSE @ N_total=%d, SNR=%.1f dB\n', N_total, SNR_dB);
              R_h_MMSE = R_h_MMSE + eye(N_total) * reg_factor;
         end

        % Re-iterate through trials for LMMSE
        for trial = 1:M_test
            h_k = h_k_collection(:, trial);
            y = y_collection(:, trial);
            A = A_collection{trial};

            % LMMSE 估计 (LMMSE Estimation)
            try
                RhA_T = R_h_MMSE * A';
                ARhA_T = A * RhA_T;
                Ry = ARhA_T + sigma2 * eye(M*T);

                if rcond(Ry) < 1e-14
                     Ry_inv = pinv(Ry);
                else
                     Ry_inv = Ry \ eye(size(Ry));
                end
                h_hat_LMMSE = RhA_T * Ry_inv * y;
                nmse_LMMSE = norm(h_hat_LMMSE - h_k)^2 / norm(h_k)^2;
                if isnan(nmse_LMMSE) || isinf(nmse_LMMSE), nmse_LMMSE = 1; end

            catch ME_LMMSE
                 fprintf('Error in LMMSE @ N_total=%d, SNR=%.1f, Trial=%d: %s\n', N_total, SNR_dB, trial, ME_LMMSE.message);
                 nmse_LMMSE = 1;
            end
            nmse_lmmse_all(trial) = nmse_LMMSE;
        end % End LMMSE trial loop (Phase 2)

        % --- Average NMSE for this N_total and SNR ---
        nmse_LS_avg(iN_total, iSNR)    = mean(nmse_ls_all);
        nmse_LMMSE_avg(iN_total, iSNR) = mean(nmse_lmmse_all);
        % nmse_CS_avg is removed

        fprintf('   SNR= %3d dB: LS=%.4e, LMMSE=%.4e\n', ... % Updated printout
                SNR_dB, nmse_LS_avg(iN_total, iSNR), nmse_LMMSE_avg(iN_total, iSNR));

    end % End of SNR loop
end % End of N_total loop

%% ========== 3. 绘图 ========== (Plotting)

% Convert NMSE to dB for plotting
nmse_LS_avg_dB = 10*log10(nmse_LS_avg);
nmse_LMMSE_avg_dB = 10*log10(nmse_LMMSE_avg);
% nmse_CS_avg_dB is removed

% --- Plot 1: NMSE vs SNR (for a fixed N_total, e.g., the last one) ---
figure;
iN_plot = length(N_total_list); % Index of N_total to plot (e.g., last one: 32)
N_total_plot = N_total_list(iN_plot);

plot(SNR_dB_list, nmse_LS_avg_dB(iN_plot, :), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(SNR_dB_list, nmse_LMMSE_avg_dB(iN_plot, :), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 7);
% CS plot removed
grid on;
legend('LS (T=N_{total})', 'LMMSE (T=N_{total})', 'Location', 'southwest'); % Updated legend
xlabel('SNR (dB)');
ylabel('NMSE (dB)');
title(sprintf('NMSE vs SNR (Sequential Probing, N_{total} = %d)', N_total_plot));
% ylim([-30 10]); % Adjust ylim based on results

% --- Plot 2: NMSE vs N_total (for a fixed SNR, e.g., 0 dB) ---
figure;
snr_idx_plot = find(SNR_dB_list == 0); % Index for SNR = 0 dB
if isempty(snr_idx_plot)
    [~, snr_idx_plot] = min(abs(SNR_dB_list - 0)); % Find closest SNR to 0 dB
    fprintf('Using SNR = %.1f dB for NMSE vs N_total plot.\n', SNR_dB_list(snr_idx_plot));
end
SNR_plot_val = SNR_dB_list(snr_idx_plot);

plot(N_total_list, nmse_LS_avg_dB(:, snr_idx_plot), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 7); hold on;
plot(N_total_list, nmse_LMMSE_avg_dB(:, snr_idx_plot), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 7);
% CS plot removed
grid on;
legend('LS (T=N_{total})', 'LMMSE (T=N_{total})', 'Location', 'best'); % Updated legend
xlabel('Total Number of Antennas (N_{total})');
ylabel('NMSE (dB)');
xticks(N_total_list); % Set ticks to actual N_total values
title(sprintf('NMSE vs N_{total} (Sequential Probing, SNR = %.1f dB)', SNR_plot_val));
% ylim([-30 10]); % Adjust ylim based on results
% 
% save('NMSE_LS.mat', 'nmse_LS_avg_dB');
% save('NMSE_LMMSE.mat', 'nmse_LMMSE_avg_dB');


%% ========== 4. OMP 函数 ========== (OMP Function REMOVED)