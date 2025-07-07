
% Reservar memoria para 100 enteros
ptr = allocate_memory_mex(100 * 4);

% Escribir datos en la memoria
process_memory_mex(ptr);

% Leer los primeros 10 enteros
data = read_memory_mex(ptr, 20);
disp(data);

% Liberar la memoria
free_memory_mex(ptr);
