function apod = apod_quadrant_diagonal(ny, nx)
% APOG_QUADRANT_DIAGONAL genera una apodización 2D con diagonales opuestas en ±1
%
% ny : número de elementos en Y (filas)
% nx : número de elementos en X (columnas)

    apod = zeros(ny, nx);

    for iy = 1:ny
        for ix = 1:nx
            if (iy <= ny/2 && ix <= nx/2) || (iy > ny/2 && ix > nx/2)
                apod(iy, ix) = 1;  % diagonal principal
            else
                apod(iy, ix) = -1; % diagonal secundaria
            end
        end
    end
end
