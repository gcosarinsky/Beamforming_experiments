typedef struct {
    // Parámetros float
    float fs;        // Frecuencia de muestreo
    float c1;        // Velocidad del sonido
    float pitch;     // Separación entre elementos
    float f1;        // Frecuencia de corte inferior
    float f2;        // Frecuencia de corte superior
    float bfd;       // Factor de apertura dinámica
    float x_step;    // Paso en x
    float z_step;    // Paso en z
    float x0_roi;    // Origen ROI en x
    float z0_roi;    // Origen ROI en z
    float t_start;   // Tiempo inicial
    float x_0;       // Centro del array

    // Parámetros int (como float)
    int taps;      // Taps del filtro
    int n_elementos; // Número de elementos
    int n_angles;  // Número de ángulos
    int nx;        // Puntos en x
    int nz;        // Puntos en z
    int n_samples; // Número de muestras
} kernel_params;


__device__ kernel_params build_params(const float *float_params, const int *int_params) {
    kernel_params params;
    
    // Copiar parámetros float
    params.fs = float_params[0];
    params.c1 = float_params[1];
    params.pitch = float_params[2];
    params.f1 = float_params[3];
    params.f2 = float_params[4];
    params.bfd = float_params[5];
    params.x_step = float_params[6];
    params.z_step = float_params[7];
    params.x0_roi = float_params[8];
    params.z0_roi = float_params[9];
    params.t_start = float_params[10];
    params.x_0 = float_params[11];
    
    // Copiar parámetros int como float
    params.taps = int_params[0];
    params.n_elementos = int_params[1];
    params.n_angles = int_params[2];
    params.nx = int_params[3];
    params.nz = int_params[4];
    params.n_samples = int_params[5];
    
    return params;
}