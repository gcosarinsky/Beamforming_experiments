% Parámetros del array
ny = 16;
nx = 16;
dx = 2;  % en lambda
dy = 2;

% Generar apodización en diagonales ±1
apod = apod_quadrant_diagonal(ny, nx);

% Crear malla de ángulos (rotación sobre X e Y)
angX_deg = linspace(-10, 10, 200);
angY_deg = linspace(-10, 10, 200);
[ang_X, ang_Y] = meshgrid(deg2rad(angX_deg), deg2rad(angY_deg));

% Calcular patrón de haz
dc = 0.05 ;
BP_zm = beam_pattern_2d(apod, dx, dy, ang_X, ang_Y);
BP_dc1 = beam_pattern_2d(apod + dc, dx, dy, ang_X, ang_Y);
BP_dc2 = beam_pattern_2d(-apod + dc, dx, dy, ang_X, ang_Y);
NSI = 0.5*(abs(BP_dc1) + abs(BP_dc2)) - abs(BP_zm) ;

BP_zm_dB  = 20*log10(abs(BP_zm)  / max(abs(BP_zm(:))));
BP_dc1_dB = 20*log10(abs(BP_dc1) / max(abs(BP_dc1(:))));
BP_dc2_dB = 20*log10(abs(BP_dc2) / max(abs(BP_dc2(:))));
NSI_dB = 20*log10(abs(NSI) / max(abs(NSI(:))));

% Mostrar resultados
figure;
imagesc(angY_deg, angX_deg, BP_zm_dB);
title('Apod original');
xlabel('Rotación Y (°)');
ylabel('Rotación X (°)');
caxis([-60 0]);
colorbar;
axis xy;
axis equal;
colormap jet;

figure;
imagesc(angY_deg, angX_deg, BP_dc1_dB);
title('Apod + dc');
xlabel('Rotación Y (°)');
ylabel('Rotación X (°)');
caxis([-60 0]);
colorbar;
axis xy;
axis equal;
colormap jet;

figure;
imagesc(angY_deg, angX_deg, BP_dc2_dB);
title('-Apod + dc');
xlabel('Rotación Y (°)');
ylabel('Rotación X (°)');
caxis([-60 0]);
colorbar;
axis xy;
axis equal;
colormap jet;

figure
imagesc(angY_deg, angX_deg, NSI_dB);
title('NSI');
xlabel('Rotación Y (°)');
ylabel('Rotación X (°)');
caxis([-60 0]);
colorbar;
axis xy;
axis equal;
colormap jet;
