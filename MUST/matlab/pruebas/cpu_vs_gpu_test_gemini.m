% --- Configuración ---
n = 10000;          % Dimensión de la matriz (n x n)
sparsity = 0.001;   % Factor de dispersión (ajusta según sea necesario)
num_trials = 5;     % Número de ejecuciones para promediar el tiempo

% --- Generación de datos ---
% Matriz sparse aleatoria en la CPU
rng('default'); % Para reproducibilidad
sparse_cpu = sprand(n, n, sparsity);

% Vector aleatorio en la CPU
vector_cpu = randn(n, 1);

% --- Ejecución en CPU ---
times_cpu = zeros(num_trials, 1);
disp('Ejecutando en CPU...');
for i = 1:num_trials
    tic;
    result_cpu = sparse_cpu * vector_cpu;
    times_cpu(i) = toc;
end
time_cpu_avg = mean(times_cpu);
fprintf('Tiempo promedio en CPU: %.6f segundos\n', time_cpu_avg);

% --- Ejecución en GPU (si la GPU está disponible) ---
if gpuDeviceCount > 0
    try
        % Transferir datos a la GPU
        sparse_gpu = gpuArray(sparse_cpu);
        vector_gpu = gpuArray(vector_cpu);

        times_gpu = zeros(num_trials, 1);
        disp('Ejecutando en GPU...');
        for i = 1:num_trials
            tic;
            result_gpu = sparse_gpu * vector_gpu;
            wait(gpuDevice); % Esperar a que terminen las operaciones en la GPU
            times_gpu(i) = toc;
        end
        time_gpu_avg = mean(times_gpu);
        fprintf('Tiempo promedio en GPU: %.6f segundos\n', time_gpu_avg);

        % Opcional: Verificar si los resultados son similares (con cierta tolerancia)
        tolerance = 1e-6;
        result_cpu_gpu = gather(result_gpu); % Transferir resultado de la GPU a la CPU
        if max(abs(result_cpu - result_cpu_gpu)) < tolerance
            disp('Resultados de CPU y GPU coinciden (dentro de la tolerancia).');
        else
            warning('Los resultados de CPU y GPU difieren significativamente.');
        end

    catch ME
        warning('Error al ejecutar en GPU: %s', ME.message);
    end
else
    disp('No se detectó ninguna GPU disponible.');
end