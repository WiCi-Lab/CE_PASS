clear; clc;
close all;

%% Parameters
rng(2025)
c = 3e8;        % Speed of light (m/s)
fc = 28e9;      % Carrier frequency (Hz)
lambda = c/fc;  % Wavelength (m)

delta = 0.1;    % Minimum spacing (m)
eta = c / (4 * pi * fc); % Attenuation constant (公式 1 中的 η)
W = 1;          % Number of waveguides
N_ants = 32;    % Maximum number of pinching antennas
M = 100000;     % Number of samples in the dataset
Dx = 20;        % x-range of the area (m)
Dy = 20;        % y-range of the area (m) 
d = 5;          % Height of the waveguide (m)
sigma2 = 1e-12; % Noise power (W)
Pm = 1e-3;      % Transmit power (W)
P = 5;          % Number of paths (LoS + NLoS)
train_SNR = 0:5:20;  % Training SNR list
N_p = 8;        % Number of pilot symbols

Ricain_K_dB = 10 ;
Ricain_K = 10^(Ricain_K_dB/10);

% Waveguide feed point
psi_0_p = [0, 0, d];

% Device position 
psi_m = [5, 5, 0];

% Some environment parameters
building_height = 10;
P_ref = 3;   % reflection paths
P_sca = 3;   % scattering paths

scale = 100;

%% Generate training dataset
loc = 0:delta:Dx;  

%% Generate test dataset
SNR_test = -20:5:20;
M_test = 500;
for N = 4:4:N_ants
    inputs = zeros(M_test*length(SNR_test), N, 1 + N_p*4);
    outputs = zeros(M_test*length(SNR_test), N, 2);
    i_sam = 0;

    for i_SNR = 1:length(SNR_test)
        for sample = 1:M_test
            i_sam = i_sam + 1;
            SNR = SNR_test(i_SNR);

            random_int = randperm(length(loc), N);
            random_int = sort(random_int);
            x_p = loc(random_int)';

            y_p = zeros(N,1);
            y_p_ = zeros(N, N_p);

            h_m1 = zeros(N,1);
            for n = 1:N
                distance_LoS = norm(psi_m - [x_p(n), 0, d]);
                h_LoS = sqrt(eta)*exp(-1j * 2*pi/lambda * distance_LoS)/distance_LoS;

                % Reflection
                h_ref = 0;
                for p_ = 1:P_ref
                    mirror_psi_p = [x_p(n), 0, d];
                    mirror_psi_p(2) = -mirror_psi_p(2);
                    distance_ref = norm(psi_m - mirror_psi_p);
                    h_ref = h_ref + sqrt(eta/p_^2) * exp(-1j*2*pi/lambda*distance_ref)/distance_ref;
                end

                % Scattering
                h_sca = 0;
                for p_ = 1:P_sca
                    scatter_x = Dx * rand;
                    scatter_y = Dy * (rand - 0.5);
                    scatter_z = building_height * rand;
                    distance1 = norm([scatter_x, scatter_y, scatter_z] - [x_p(n),0,d]);
                    distance2 = norm([scatter_x, scatter_y, scatter_z] - psi_m);
                    total_distance = distance1 + distance2;
                    h_sca = h_sca + sqrt(eta/p_^2) ...
                                  * exp(-1j*2*pi/lambda*total_distance)/ (distance1 * distance2);
                end

                % p_LOS
                if distance_LoS <= 12
                    p_LOS = 1;
                elseif distance_LoS < 20 && distance_LoS > 12
                    p_LOS = exp(-(distance_LoS-1.2)/4.7);
                else
                    p_LOS = 0.32 * exp(-(distance_LoS-6.5)/32.6);
                end
                I_LOS = randsrc(1,1,[1,0; p_LOS, 1-p_LOS]);

                % combine
                total_h = I_LOS*sqrt(Ricain_K/(Ricain_K+1))*h_LoS + ...
                          sqrt(1/(Ricain_K+1))*(h_ref + h_sca)/P_ref;
                h_m1(n) = total_h;
            end

            % waveguide
            n_e = 1.4;
            lambda_g = lambda/n_e;
            h_2 = zeros(N,1);
            s_ = zeros(N,1);
            s_(1) = norm(psi_0_p - [x_p(1), 0, d]);
            for m_ = 2:N
                s_(m_) = s_(m_-1) + norm([x_p(m_),0,d] - [x_p(m_-1),0,d]);
            end
            for n = 1:N
                phase_ = exp(-1j*2*pi/lambda_g * s_(n));
                h_2(n) = sqrt(1/N)*phase_;
            end

            % pilot
            s_m_ = zeros(1, N_p);
            for t_ = 1:N_p
                s_m_(t_) = exp(-1j * 2*pi*(t_-1)^2/(2*N_p));
            end

            y_p_ = h_2' * h_m1 * s_m_;
            y_p_original = awgn(y_p_, SNR, 'measured');
            y_p_mat = h_2 * y_p_original;

            % y_p_mat = y_p_mat ./ abs(y_p_mat);
            % h_m1 = h_m1 ./ mean(abs(h_m1));

            y_p_mat = scale * y_p_mat;
            h_m1 = scale * h_m1;
            
            % Store
            x_p_norm = x_p / Dx;
            inputs(i_sam,:,:) = [ ...
                x_p_norm, ...
                real(y_p_mat), imag(y_p_mat), ...
                real(repmat(s_m_,N,1)), imag(repmat(s_m_,N,1)) ...
            ];
            outputs(i_sam,:,1) = real(h_m1);
            outputs(i_sam,:,2) = imag(h_m1);
        end
    end

    filename = [num2str(W),'waveguides_',num2str(N),'ants_', ...
                num2str(N_p),'pilots_',num2str(M),'samples_', ...
                num2str(train_SNR(end)),'snr'];
    save(['PinchingTest_R1', filename], 'inputs', 'outputs');
end

disp('Dataset generation complete without per-sample amplitude normalization.');

