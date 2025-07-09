extern "C" {

__global__ void pwi(const short *matrix,
                      const short *matrix_imag,
                      float *img,
                      float *img_imag,
                      float *cohe,
                      const float *angles) {

    unsigned short iz = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short ix = blockIdx.y * blockDim.y + threadIdx.y;

    if (iz >= NZ || ix >= NX) return;

    unsigned short f_idx = iz * NX + ix;
    float xf = X0_ROI + X_STEP * ix;
    float zf = Z0_ROI + Z_STEP * iz; /* Z POSITIVE DOWNWARDS */
    float x_rx, wave_source;
    float t1, t2;
    float t, dt, temp, theta, ap_dyn;
    unsigned int k, k0;
    float a, b, q = 0, q_imag = 0, w = 0, w_imag = 0;

    for (unsigned short i = 0; i < N_ANGLES; i++) {

        theta = angles[i];
        wave_source = X_0 * (theta < 0 ? 1 : -1);
        t1 = ((xf - wave_source) * sinf(theta) + zf * cosf(theta)) / C1;

        for (unsigned short e = 0; e < N_ELEMENTOS; e++) {
            x_rx = e * PITCH - X_0;
            t2 = hypotf(x_rx - xf, zf) / C1;
            ap_dyn = fabsf(x_rx - xf) / zf < BFD;
            t = t1 + t2 - T_START;
            t = t * (t > 0 ? 1 : 0);  /* First sample must be 0 !!! */
            k = min((unsigned int)floorf(t * FS), N_SAMPLES - 2); /* resto 2 para evitar que k+1 = N_SAMPLES */
            dt = t * FS - k;
            k0 = i * N_ELEMENTOS * N_SAMPLES + e * N_SAMPLES;

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

        img[f_idx] = q ;
        img_imag[f_idx] = q_imag ;
        cohe[f_idx] = hypotf(w, w_imag) ;
    }
}

}