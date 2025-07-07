% Parámetros del array
nel_x = 16;
nel_y = 16;
dx = 2;  % lambda
dy = 2;
apod = ones(nel_y, nel_x);  % sin apodización

% Grid de ángulos en radianes
angX_deg = linspace(-10, 10, 200);
angY_deg = linspace(-10, 10, 200);
[ang_X, ang_Y] = meshgrid(deg2rad(angX_deg), deg2rad(angY_deg));

% Beam pattern
BP = beam_pattern_2d(apod, dx, dy, ang_X, ang_Y);
BP_dB = 20 * log10(abs(BP)/max(abs(BP(:))));

% Visualización
imagesc(angY_deg, angX_deg, BP_dB)
xlabel('Rotación sobre eje Y (°)')
ylabel('Rotación sobre eje X (°)')
title('Beam pattern 2D (dB)')
colorbar
caxis([-60 0])
axis xy
colormap jet

