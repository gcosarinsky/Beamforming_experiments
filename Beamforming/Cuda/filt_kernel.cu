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
        // Limitar el valor acumulado dentro del rango de un short
        q = fmaxf(fminf(q, 32767.0f), -32768.0f);
    }
    return q;
}


extern "C" {

__global__ void filt_kernel(const short *datain, const float *coef_g, short *dataout) {
    // Calcular índices globales del hilo
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Índice de thread

    // Verificar límites
    if (tid >= N_ELEMENTOS * N_SAMPLES) return;

    //extern __shared__ float coef[];
    float coef[TAPS + 1] ;
    short x[TAPS + 1];
    // Índice de primer sample del ascan en dataout
    int i = N_SAMPLES * tid;

    /* copiar coeficientes del filtro en memoria privada */
    unsigned short l0 = TAPS/2 ;
    unsigned short lmax = TAPS + 1 ;
    for (unsigned short l=0; l < lmax; l++) {
        coef[l] = coef_g[l] ;
    }

    // Calcular transitorio
    for (unsigned short l = 0; l <= l0; l++) {
        x[l + l0] = datain[i + l];
    }

    // Calcular primera muestra de salida
    dataout[i] = (short)rintf(multisum(x, coef));

    // Continuar hasta traer la última muestra del A-scan
    lmax = N_SAMPLES - l0;
    for (unsigned short l = 1; l < lmax; l++) {
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