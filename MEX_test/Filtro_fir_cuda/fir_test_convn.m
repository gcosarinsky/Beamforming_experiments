% Configuración
cfg.n_elementos = 128;
cfg.taps = 62;
cfg.fs = 40;       % kHz
cfg.f1 = 0.5;      % kHz
cfg.f2 = 7;        % kHz
cfg.n_samples = 512;

% Tiempo
t = (0:cfg.n_samples-1) / cfg.fs;

% Frecuencia central
f_central = (cfg.f1 + cfg.f2) / 2;

% Generar directamente en orden [samples, receptores, emisores]
matrix = zeros(cfg.n_samples, cfg.n_elementos, cfg.n_elementos, 'int16');
for e = 1:cfg.n_elementos
    for r = 1:cfg.n_elementos
        fase = 2 * pi * rand();
        senal = sin(2 * pi * f_central * t + fase);
        ruido = 0.5 * randn(size(senal));
        senal_ruidosa = senal + ruido;
        matrix(:, r, e) = int16(1000 * senal_ruidosa);
    end
end

% Coeficientes del filtro FIR pasabanda
bandpass_coef = single(fir1(cfg.taps, [2*cfg.f1/cfg.fs, 2*cfg.f2/cfg.fs], 'bandpass'));
% Reshape filter coefficients
kernel = reshape(bandpass_coef, [1, 1, cfg.taps + 1]);

% Apply convolution along the time dimension
matrix_filt = convn(double(matrix), kernel, 'same');

% Plot a trace
i = 10;
figure;
plot(squeeze(matrix(:,1,i)), 'DisplayName', 'raw'); hold on;
plot(squeeze(matrix_filt(:,1,i)), 'DisplayName', 'filt');
legend;
xlabel('Sample');
ylabel('Amplitude');
title('FIR Bandpass Filter Applied with convn');