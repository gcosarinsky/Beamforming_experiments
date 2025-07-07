nel = 16;
dc = 0.05;
apod_0 = ones(nel, 1);
apod_zm = apod_triple_group(nel) ;              
apod_dc = apod_zm + dc ;
d_landa = 2 ;                   
sin_theta = linspace(-1, 1, 1000);

amplitude_0 = beam_pattern_1d(nel, apod_0, d_landa, sin_theta);
amplitude_zm = beam_pattern_1d(nel, apod_zm, d_landa, sin_theta);
amplitude_dc = beam_pattern_1d(nel, apod_dc, d_landa, sin_theta);  % deberia ser lo mismo que amplitude_zm + 0.05 * amplitude_0
nsi = abs(amplitude_dc) - abs(amplitude_zm) ;

% Normalizar e ir a dB
amplitude_0_db = 20*log10(abs(amplitude_0)/max(abs(amplitude_0)));
amplitude_zm_db = 20*log10(abs(amplitude_zm)/max(abs(amplitude_zm)));
amplitude_dc_db = 20*log10(abs(amplitude_dc)/max(abs(amplitude_dc)));
nsi_db = 20*log10(abs(nsi)/max(abs(nsi)));

% Plot
figure
stem(1:nel, apod_zm, 'filled', 'LineWidth', 1.5)
ylim([-1.5 1.5])
xlabel('Índice de elemento')
ylabel('Valor de apodización')
title('Apodización dividida en 3 grupos con valores ±1')
grid on

figure
plot(sin_theta, amplitude_0_db)
xlabel('sin(\theta)')
ylabel('Amplitud (dB)')
title('Apodización constante')
grid on
ylim([-80, 0])

figure
plot(sin_theta, amplitude_zm_db)
hold on
plot(sin_theta, amplitude_dc_db)
xlabel('sin(\theta)')
ylabel('Amplitud (dB)')
title('Partes de NSI')
grid on
ylim([-80, 0])

figure
plot(sin_theta, nsi_db)
hold on
plot(sin_theta, amplitude_0_db)
xlabel('sin(\theta)')
ylabel('Amplitud (dB)')
title('NSI')
grid on
ylim([-80, 0])
