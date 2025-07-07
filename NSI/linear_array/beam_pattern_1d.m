function amplitude = beam_pattern_1d(nel, apod, d_landa, sin_theta)
%BEAM_PATTERN_1D Calcula el patrón de haz (far field) de un array 1D
%
% Inputs:
%   nel        - número de elementos del array
%   apod       - vector de apodización (tamaño [nel, 1] o [1, nel])
%   d_landa    - espaciamiento entre elementos en unidades de lambda
%   sin_theta  - vector de valores de sin(?) donde se evalúa el patrón
%
% Output:
%   amplitude  - vector complejo con la respuesta del array en sin_theta

    % Validación
    if numel(apod) ~= nel
        error('La apodización debe tener %d elementos.', nel);
    end

    % Vector de posiciones relativas de los elementos (en múltiplos de lambda)
    n = 0:(nel-1);  % posición de cada elemento

    % Matriz de fase: [nel x num_angles]
    phase_matrix = exp(1j * 2 * pi * d_landa * (n.') * sin_theta); 

    % Suma ponderada de los elementos (producto apodización × fase)
    amplitude = apod(:).' * phase_matrix;  % tamaño [1 x num_angles]
end
