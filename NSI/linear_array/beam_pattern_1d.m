function amplitude = beam_pattern_1d(nel, apod, d_landa, sin_theta)
%BEAM_PATTERN_1D Calcula el patr�n de haz (far field) de un array 1D
%
% Inputs:
%   nel        - n�mero de elementos del array
%   apod       - vector de apodizaci�n (tama�o [nel, 1] o [1, nel])
%   d_landa    - espaciamiento entre elementos en unidades de lambda
%   sin_theta  - vector de valores de sin(?) donde se eval�a el patr�n
%
% Output:
%   amplitude  - vector complejo con la respuesta del array en sin_theta

    % Validaci�n
    if numel(apod) ~= nel
        error('La apodizaci�n debe tener %d elementos.', nel);
    end

    % Vector de posiciones relativas de los elementos (en m�ltiplos de lambda)
    n = 0:(nel-1);  % posici�n de cada elemento

    % Matriz de fase: [nel x num_angles]
    phase_matrix = exp(1j * 2 * pi * d_landa * (n.') * sin_theta); 

    % Suma ponderada de los elementos (producto apodizaci�n � fase)
    amplitude = apod(:).' * phase_matrix;  % tama�o [1 x num_angles]
end
