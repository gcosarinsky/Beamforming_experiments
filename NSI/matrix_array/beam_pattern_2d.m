function BP = beam_pattern_2d(apod, dx_lam, dy_lam, ang_X, ang_Y)
% BEAM_PATTERN_2D_EULER calcula el beam pattern en 2D usando �ngulos de rotaci�n sobre X e Y
%
% apod     : matriz de apodizaci�n [nel_y � nel_x]
% dx_lam   : espaciado entre elementos en x (en longitudes de onda)
% dy_lam   : espaciado entre elementos en y (en longitudes de onda)
% ang_X    : matriz de �ngulos (rad) de rotaci�n sobre el eje X
% ang_Y    : matriz de �ngulos (rad) de rotaci�n sobre el eje Y
%
% BP       : patr�n de haz (normalizado a 1), misma forma que ang_X

    [nel_y, nel_x] = size(apod);

    % Coordenadas f�sicas de los elementos del array
    x = ((0:nel_x-1) - (nel_x-1)/2) * dx_lam;
    y = ((0:nel_y-1) - (nel_y-1)/2) * dy_lam;

    % Componentes del vector de onda
    kx = sin(ang_Y);  % direcci�n X ? rotaci�n sobre Y
    ky = sin(ang_X);  % direcci�n Y ? rotaci�n sobre X

    % Inicializar patr�n
    BP = zeros(size(ang_X));

    for ix = 1:nel_x
        for iy = 1:nel_y
            phase = 2*pi * (kx * x(ix) + ky * y(iy));
            BP = BP + apod(iy, ix) .* exp(1j * phase);
        end
    end

end

