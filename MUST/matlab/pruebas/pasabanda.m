% Especificaciones del filtro
Fs = 62.5e6;        % Frecuencia de muestreo (Hz) - 62.5 MHz
Fpass1 = 3e6;       % Inicio de la banda de paso (Hz) - 3 MHz
Fpass2 = 8e6;       % Fin de la banda de paso (Hz) - 8 MHz
Fstop1 = 2e6;       % Fin de la primera banda de rechazo (Hz) - Puedes ajustar
Fstop2 = 9e6;       % Inicio de la segunda banda de rechazo (Hz) - Puedes ajustar
Apass = 1;          % Rizado máximo en la banda de paso (dB)
Astop = 60;         % Atenuación mínima en la banda de la banda de rechazo (dB)

% Diseñar el filtro FIR de orden mínimo usando designfilt
d = designfilt('bandpassfir', ...
               'StopbandFrequency1', Fstop1, 'PassbandFrequency1', Fpass1, ...
               'PassbandFrequency2', Fpass2, 'StopbandFrequency2', Fstop2, ...
               'StopbandAttenuation1', Astop, 'PassbandRipple', Apass, ...
               'StopbandAttenuation2', Astop, 'SampleRate', Fs);

% Obtener los coeficientes del filtro (numerador 'b')
bp_coef = d.Numerator;
a = 1; % Para un filtro FIR, el denominador 'a' es 1

% Visualizar la respuesta en frecuencia del filtro
freqz(b, 1, 8192, Fs);
title('Respuesta en Frecuencia del Filtro FIR Pasabanda');
xlabel('Frecuencia (Hz)');
ylabel('Magnitud (dB)');
grid on;

% --- Aplicación del filtro (sin cambios) ---
% Generar una señal de ejemplo que contenga frecuencias dentro y fuera de la
% banda de paso
t = 0:1/Fs:1e-4; % Un corto periodo de tiempo
f1 = 1e6;         % Frecuencia fuera de la banda de paso
f2 = 5e6;         % Frecuencia dentro de la banda de paso
f3 = 10e6;        % Frecuencia fuera de la banda de paso
signal = sin(2*pi*f1*t) + sin(2*pi*f2*t) + sin(2*pi*f3*t) + 0.1*randn(size(t));

% Aplicar el filtro pasabanda sin delay usando filtfilt
filtered_signal = filtfilt(bp_coef, a, signal);
save('bp_coef', 'bp_coef');
% Graficar la señal original y la señal filtrada
figure;
subplot(2,1,1);
plot(t*1e6, signal);
title('Señal Original');
xlabel('Tiempo (µs)');
ylabel('Amplitud');

subplot(2,1,2);
plot(t*1e6, filtered_signal);
title('Señal Filtrada (sin delay)');
xlabel('Tiempo (µs)');
ylabel('Amplitud');