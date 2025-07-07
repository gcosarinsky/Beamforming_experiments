function [apod1, apod2, apod3] = nsi_apods(nel, dc_offset)
    % Verificar que nel es par para dividir en dos mitades iguales
    if mod(nel, 2) ~= 0
        error('nel debe ser un n�mero par');
    end

    half = nel / 2;

    % 1) Apodizaci�n 1: -1 izquierda, +1 derecha
    apod1 = [-ones(1, half), ones(1, half)];

    % 2) Apodizaci�n 2: apod1 + dc_offset
    apod2 = apod1 + dc_offset;

    % 3) Apodizaci�n 3: simetr�a (reflejo) de apod2
    apod3 = fliplr(apod2);
end
