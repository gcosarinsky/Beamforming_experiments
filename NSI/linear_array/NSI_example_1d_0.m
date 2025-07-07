nel = 16;
dc = 0.5
[apod_zm, apod_dc1, apod_dc2] = nsi_apods(nel, dc) ;              
d_landa = 2 ;                    
sin_theta = linspace(-1, 1, 1000);

amplitude_zm = beam_pattern_1d(nel, apod_zm, d_landa, sin_theta);
amplitude_dc1 = beam_pattern_1d(nel, apod_dc1, d_landa, sin_theta);
amplitude_dc2 = beam_pattern_1d(nel, apod_dc2, d_landa, sin_theta);
nsi = 0.5*(abs(amplitude_dc1) + abs(amplitude_dc2)) - abs(amplitude_zm) ;
%nsi = abs(amplitude_dc1) - abs(amplitude_zm) ;

% Normalizar e ir a dB
amplitude_zm_db = 20*log10(abs(amplitude_zm)/max(abs(amplitude_zm)));
amplitude_dc1_db = 20*log10(abs(amplitude_dc1)/max(abs(amplitude_dc1)));
amplitude_dc2_db = 20*log10(abs(amplitude_dc1)/max(abs(amplitude_dc2)));
nsi_db = 20*log10(abs(nsi)/max(abs(nsi)));

% Plot
figure
stem(1:nel, apod_zm, 'filled', 'LineWidth', 1.5)
ylim([-1.5 1.5])
xlabel('Índice de elemento')
ylabel('Valor de apodización')
grid on

figure
%plot(sin_theta, amplitude_zm_db)
hold on
plot(sin_theta, amplitude_dc1_db)
plot(sin_theta, amplitude_dc2_db)
xlabel('sin(\theta)')
ylabel('Amplitud (dB)')
title('Patrón de haz en campo lejano')
grid on
ylim([-80, 0])

figure
plot(sin_theta, nsi_db)
xlabel('sin(\theta)')
ylabel('Amplitud (dB)')
title('Patrón de haz en campo lejano')
grid on
ylim([-60, 0])
