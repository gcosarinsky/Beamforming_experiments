extern "C" __global__ void filt_batch_kernel(const int *int_params, const short *datain, const float *coef_g, short *dataout)
    {

    // Calcular índices globales del hilo
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Índice de thread

    int nang = int_params[N_ANGLES];
    int nel = int_params[N_ELEMENTOS];
    int ns = int_params[N_SAMPLES];
    int taps = int_params[TAPS];
    int n_batch = int_params[N_BATCH];
    int batch_stride = ns * nel * nang; // stride para indexar los batches
    int i ;

    // Verificar límites
    if (tid >= nel * ns) return;

    //extern __shared__ float coef[];
    float coef[MAX_FIR_SIZE] ;
    short x[MAX_FIR_SIZE];

    /* copiar coeficientes del filtro en memoria privada */
    unsigned short l0 = taps/2 ;
    unsigned short lmax = taps + 1 ;
    for (unsigned short l=0; l < lmax; l++) {
        coef[l] = coef_g[l] ;
        }

    lmax = ns - l0;
    for (unsigned short bch = 0; bch < n_batch; bch++) {

        i = ns * tid + bch * batch_stride; //índice para el siguiente batch

        // Calcular transitorio
        for (unsigned short l = 0; l <= l0; l++) {
            x[l + l0] = datain[i + l];
        }

        // Calcular primera muestra de salida
        dataout[i] = (short)rintf(multisum(x, coef, taps));

        // Continuar hasta traer la última muestra del A-scan
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
}


extern "C" __global__ void pwi_batch_1pix_per_thread(
                               const int *int_params,
                               const float *float_params,
                               const float *angles,
                               const short *matrix,
                               const short *matrix_imag,
                               float *img,
                               float *img_imag) {

    unsigned short iz = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short ix = blockIdx.y * blockDim.y + threadIdx.y;

    // Obtener los parámetros enteros y flotantes
    // int_params
    int n_batch = int_params[N_BATCH];
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
    float x_rx, wave_source;
    float t1, t2;
    float t, dt, temp, theta, ap_dyn;
    float a, b; // para amacenar las samples localmente

    // variables para indexar
    unsigned int m_idx, k, k0 = 0, batch_offset; // para indexar las matrix
    unsigned int batch_stride = ns * nel * nang, img_stride = nx * nz ;
    unsigned short local_offset = (threadIdx.y * blockDim.x + threadIdx.x) * n_batch * 2; // Para indexar la memoria compartida
    unsigned short q_idx ; // valor inicial para indexar la memoria compartida

    unsigned short f_idx = iz * nx + ix; // para indexar la imagen

    extern __shared__ float q[]; // Variable compartida para almacenar resultados parciales, el tamaño
    // es (nro de pixeles por block) * n_batch * 2 (real e imaginaria)

    for (unsigned short i = 0; i < nang; i++) {
        theta = angles[i];
        wave_source = x0 * (theta < 0 ? 1 : -1);
        t1 = ((xf - wave_source) * sinf(theta) + zf * cosf(theta)) / c1 - t_start;
        x_rx = -x0;  // Inicializar x_rx para el primer elemento

        for (unsigned short e = 0; e < nel; e++) {
            x_rx += pitch;  // Incrementar x_rx para cada elemento
            compute_sample_index(&x_rx, &xf, &zf, &c1, &bfd, &fs, &ns, &t1, &t2, &t, &k, &ap_dyn);
            dt = t * fs - k;

            batch_offset = 0 ;
            q_idx = local_offset ;
            for (unsigned short bch = 0; bch < n_batch; bch++) {
                m_idx = batch_offset + k0 + k;
                temp = (float)matrix[m_idx];
                a = ((float)matrix[m_idx + 1] - temp) * dt + temp;
                q[q_idx] += a * ap_dyn;
                __syncthreads();

                temp = (float)matrix_imag[m_idx];
                b = ((float)matrix_imag[m_idx + 1] - temp) * dt + temp;
                q[n_batch + q_idx] += b * ap_dyn;
                __syncthreads();

                batch_offset += batch_stride ;
                q_idx += 1; // Incrementar el índice para la memoria compartida
            }
            k0 += ns;
        }
    }

    q_idx = local_offset; // Reiniciar el índice para la memoria compartida
    for (unsigned short bch = 0; bch < n_batch; bch++) {
        img[f_idx] = q[q_idx] ;
        __syncthreads();
        img_imag[f_idx] = q[q_idx + n_batch]; ;
        __syncthreads();
        f_idx += img_stride; // Incrementar el índice para la imagen
        q_idx += 1; // Incrementar el índice para la memoria compartida
        }
}