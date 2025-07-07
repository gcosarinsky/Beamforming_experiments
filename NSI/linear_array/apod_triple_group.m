function apod = apod_triple_group(nel)
    % Divide el array en 3 grupos:
    % borde izquierdo (nel/4 elementos), centro (nel/2), borde derecho (nel/4)
    % Apodización: borde = -1, centro = +1

    if mod(nel, 4) ~= 0
        error('nel debe ser múltiplo de 4 para esta apodización');
    end

    nel_1 = nel / 4;  % borde izquierdo
    nel_2 = nel / 2;  % centro
    nel_3 = nel / 4;  % borde derecho

    apod = zeros(1, nel);

    % Asignar valores
    apod(1:nel_1) = -1;                         % borde izquierdo
    apod(nel_1+1 : nel_1+nel_2) = 1;            % centro
    apod(nel_1+nel_2+1 : end) = -1;              % borde derecho
end
