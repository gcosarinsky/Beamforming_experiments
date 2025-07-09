
extern "C" __global__ void pwi_with_params_struct(const short *matrix,
                                                  const short *matrix_imag,
                                                  float *img,
                                                  float *img_imag,
                                                  float *cohe,
                                                  const float *float_params,
                                                  const int *int_params,
                                                  const float *angles) {

    // Construir los parámetros usando build_params
    kernel_params params = build_params(float_params, int_params);

    unsigned short iz = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= params.nz || ix >= params.nx) return;

    unsigned short f_idx = iz * params.nx + ix;
    float xf = params.x0_roi + params.x_step * ix;
    float zf = params.z0_roi + params.z_step * iz; /* Z POSITIVE DOWNWARDS */
    float x_rx, wave_source;
    float t1, t2;
    float t, dt, temp, theta, ap_dyn;
    unsigned int k, k0;
    float a, b, q = 0, q_imag = 0, w = 0, w_imag = 0;

    for (unsigned short i = 0; i < params.n_angles; i++) {
        theta = angles[i];
        wave_source = params.x_0 * (theta < 0 ? 1 : -1);
        t1 = ((xf - wave_source) * sinf(theta) + zf * cosf(theta)) / params.c1;

        for (unsigned short e = 0; e < params.n_elementos; e++) {
            x_rx = e * params.pitch - params.x_0;
            t2 = hypotf(x_rx - xf, zf) / params.c1;
            ap_dyn = fabsf(x_rx - xf) / zf < params.bfd;
            t = t1 + t2 - params.t_start;
            t = t * (t > 0 ? 1 : 0);  /* First sample must be 0 !!! */
            k = min((int)floorf(t * params.fs), params.n_samples - 2); /* resto 2 para evitar que k+1 = N_SAMPLES */
            dt = t * params.fs - k;
            k0 = i * params.n_elementos * params.n_samples + e * params.n_samples;

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
        }

        img[f_idx] = q;
        img_imag[f_idx] = q_imag;
        cohe[f_idx] = hypotf(w, w_imag);
    }
}

}