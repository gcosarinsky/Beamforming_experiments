__device__ void shift_izq(short *x) {
    for (int j = 0; j < TAPS; j++) {
        x[j] = x[j + 1];
    }
    x[TAPS] = 0;
}

__device__ float multisum(short *x, float *coef) {
    float q = 0;
    for (int j = 0; j < (TAPS + 1); j++) {
        q += coef[TAPS - j] * x[j];
    }
    return q;
}


extern "C" {

__global__ void filt_kernel_2(const short *datain, const float *coef_g, short *dataout) {
    // Calcular índices globales del hilo
    int e = blockIdx.x * blockDim.x + threadIdx.x;  // Índice de elemento
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // Índice de receptor

    // Verificar límites
    if (e >= N_ELEMENTOS || r >= N_ELEMENTOS) return;

    extern __shared__ float coef[];
    short x[TAPS + 1];

    // Copiar coeficientes a memoria compartida
    for (int l = threadIdx.x; l <= TAPS; l += blockDim.x) {
        coef[l] = coef_g[l];
    }
    __syncthreads();

    // Índice de primer sample del ascan en dataout
    int i = N_ELEMENTOS * N_SAMPLES * e + N_SAMPLES * r;

    // Calcular transitorio
    int l0 = TAPS / 2;
    for (int l = 0; l <= l0; l++) {
        x[l + l0] = datain[i + l];
    }

    // Calcular primera muestra de salida
    dataout[i] = (short)rintf(multisum(x, coef));

    // Continuar hasta traer la última muestra del A-scan
    int lmax = N_SAMPLES - l0;
    for (int l = 1; l < lmax; l++) {
        shift_izq(x);
        x[TAPS] = datain[i + l + l0];
        dataout[i + l] = (short)rintf(multisum(x, coef));
    }

    for (int l = lmax; l < N_SAMPLES; l++) {
        shift_izq(x);
        dataout[i + l] = (short)rintf(multisum(x, coef));
    }

    /* forzar esto para el tema de los bordes en el beamforming */
    dataout[i] = 0 ;
    dataout[i + N_SAMPLES - 2] = 0 ;
    dataout[i + N_SAMPLES - 1] = 0 ;
}

}