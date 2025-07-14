#define MAX_FIR_SIZE 64
#define FLT_EPSILON 1.1920929e-7f

enum FloatParams {
    FS,
    C1,
    PITCH,
    F1,
    F2,
    BFD,
    X_STEP,
    Z_STEP,
    X0_ROI,
    Z0_ROI,
    T_START,
    X_0,
    FLOAT_PARAMS_COUNT // Número total de parámetros float
};

enum IntParams {
    TAPS,
    N_ELEMENTOS,
    N_ANGLES,
    NX,
    NZ,
    N_SAMPLES,
    INT_PARAMS_COUNT // Número total de parámetros int
};


__device__ void shift_izq(short *x, int taps) {
    for (int j = 0; j < taps; j++) {
        x[j] = x[j + 1];
    }
    x[taps] = 0;
}


__device__ float multisum(short *x, float *coef, int taps) {
    float q = 0;
    for (int j = 0; j < (taps + 1); j++) {
        q += coef[taps - j] * x[j];
        // Limitar el valor acumulado dentro del rango de un short
    }
    q = fmaxf(fminf(q, 32767.0f), -32768.0f);
    return q;
}

__device__ void compute_sample_index(float *x_rx, float *xf, float *zf, float *c1, float *bfd,
                                     float *fs, int *ns, float *t1, float *t2, float *t,
                                     unsigned int *k, float *ap_dyn) {
    *t2 = hypotf(*x_rx - *xf, *zf) / *c1;
    *ap_dyn = fabsf(*x_rx - *xf) / *zf < *bfd;
    *ap_dyn = *zf/(fabsf(*x_rx - *xf) + FLT_EPSILON) > *bfd ;  // Apodización dinámica
    *t = *t1 + *t2;
    *t = *t * (*t > 0 ? 1 : 0);  /* First sample must be 0 !!! */
    *k = min((unsigned int)floorf(*t * (*fs)), *ns - 2); /* resto 2 para evitar que k+1 = ns */
}



extern "C" __global__ void filt_kernel(const int *int_params, const short *datain, const float *coef_g, short *dataout)
    {

    // Calcular índices globales del hilo
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Índice de thread

    int nel = int_params[N_ELEMENTOS];
    int ns = int_params[N_SAMPLES];
    int taps = int_params[TAPS];

    // Verificar límites
    if (tid >= nel * ns) return;

    //extern __shared__ float coef[];
    float coef[MAX_FIR_SIZE] ;
    short x[MAX_FIR_SIZE];

    // Índice de primer sample del ascan en dataout
    int i = ns * tid;

    /* copiar coeficientes del filtro en memoria privada */
    unsigned short l0 = taps/2 ;
    unsigned short lmax = taps + 1 ;
    for (unsigned short l=0; l < lmax; l++) {
        coef[l] = coef_g[l] ;
    }

    // Calcular transitorio
    for (unsigned short l = 0; l <= l0; l++) {
        x[l + l0] = datain[i + l];
    }

    // Calcular primera muestra de salida
    dataout[i] = (short)rintf(multisum(x, coef, taps));

    // Continuar hasta traer la última muestra del A-scan
    lmax = ns - l0;
    for (unsigned short l = 1; l < lmax; l++) {
        shift_izq(x, taps);
        x[taps] = datain[i + l + l0];
        dataout[i + l] = (short)rintf(multisum(x, coef, taps));
    }

    for (int l = lmax; l < ns; l++) {
        shift_izq(x, taps);
        dataout[i + l] = (short)rintf(multisum(x, coef, taps));
    }

    /* forzar esto para el tema de los bordes en el beamforming */
    dataout[i] = 0 ;
    dataout[i + ns - 2] = 0 ;
    dataout[i + ns - 1] = 0 ;
}

// cambiar el orden del loop, INCOMPLETO!!!
extern "C" __global__ void pwi_1pix_per_thread(
                               const int *int_params,
                               const float *float_params,
                               const float *angles,
                               const short *matrix,
                               const short *matrix_imag,
                               float *img,
                               float *img_imag,
                               float *cohe) {

    unsigned short iz = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short ix = blockIdx.y * blockDim.y + threadIdx.y;

    // Obtener los parámetros enteros y flotantes
    // int_params
    int nel = int_params[N_ELEMENTOS];
    int nang = int_params[N_ANGLES];
    int ns = int_params[N_SAMPLES];
    int nx = int_params[NX];
    int nz = int_params[NZ];
    if (iz >= nz || ix >= nx) return;  // Verificar límites de los índices

    // float params
    float fs = float_params[FS];
    float c1 = float_params[C1];
    float pitch = float_params[PITCH];
    float x0_roi = float_params[X0_ROI];
    float z0_roi = float_params[Z0_ROI];
    float x_step = float_params[X_STEP];
    float z_step = float_params[Z_STEP];
    float t_start = float_params[T_START];
    float x0 = float_params[X_0];
    float bfd = float_params[BFD];

    float xf = x0_roi + x_step * ix;
    float zf = z0_roi + z_step * iz;  // Z POSITIVE DOWNWARDS
    float x_rx = -x0, wave_source;
    float t1, t2;
    float t, dt, temp, theta, ap_dyn;
    unsigned int k, k0 = 0;
    float a, b, q = 0, q_imag = 0, w = 0, w_imag = 0;

    unsigned short f_idx = iz * nx + ix;

    for (unsigned short e = 0; e < nel; e++) {
        k0 = e * ns;  // Índice base para el A-scan
        x_rx += pitch;  // Incrementar x_rx para cada elemento
        t2 = hypotf(x_rx - xf, zf) / c1;
        ap_dyn = zf/(fabsf(x_rx - xf) + FLT_EPSILON) > bfd ;  // Apodización dinámica

        for (unsigned short i = 0; i < nang; i++) {

            theta = angles[i];
            wave_source = x0 * (theta < 0 ? 1 : -1);
            t1 = ((xf - wave_source) * sinf(theta) + zf * cosf(theta)) / c1 - t_start;
            t = t1 + t2;
            dt = t * fs - k;

            temp = (float)matrix[k0 + k];
            a = ((float)matrix[k0 + k + 1] - temp) * dt + temp;
            q += a * ap_dyn;

            temp = (float)matrix_imag[k0 + k];
            b = ((float)matrix_imag[k0 + k + 1] - temp) * dt + temp;
            q_imag += b * ap_dyn;

            temp = hypotf(a, b) + FLT_EPSILON;  /* módulo del "fasor" */
            /* se suman las componentes de los fasores para cada A-scan */
            w += a / temp;
            w_imag += b / temp;

            k0 += ns * nel; // saltar a la siguiente onda
        }
    }
    img[f_idx] = q ;
    img_imag[f_idx] = q_imag ;
    cohe[f_idx] = hypotf(w, w_imag) ;
}
