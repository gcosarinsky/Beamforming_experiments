__kernel void pwi(__global short *matrix,
                  __global short *matrix_imag,
                  __global float *img,
                  __global float *img_imag,
                  __global float *cohe,
                  __global float *angles) {

    ushort iz = get_global_id(0);
    ushort ix = get_global_id(1);

    ushort f_idx = iz*NX + ix ;
    float xf = X0_ROI + X_STEP * ix ;
    float zf = Z0_ROI + Z_STEP * iz ; /* Z POSITIVE DOWNWARDS  */
    float x_rx, wave_source ;
    float t1, t2 ;
    float t, dt, temp, theta, ap_dyn ;
    uint k, k0 ;
    uint N = N_ELEMENTOS*N_ANGLES ;
    float a, b, q = 0, q_imag = 0, w = 0, w_imag = 0 ;

    for (ushort i=0; i < N_ANGLES; i++) {

        theta = angles[i] ;
        /*
        lo de wave_source lo copia de la funcion gen_mat en SMB
        */
        wave_source = X_0 * (theta < 0 ? 1 : -1) ;
        t1 = ((xf - wave_source) * sin(theta) + zf * cos(theta))/C1  ; /* X_0 is half array aperture */

        for (ushort e=0; e < N_ELEMENTOS; e++) {
            x_rx = e*PITCH - X_0;
            t2 = hypot(x_rx - xf, zf)/C1 ;
            ap_dyn = fabs(x_rx - xf)/zf < BFD ;
            t = t1 + t2 - T_START;
            t = t * step(0, t);  /* First sample must be 0 !!! */
            //k =  fmin(floor(t*FS) , N_SAMPLES - 2); /* resto 2 para evitar que k+1 = N_SAMPLES */
            k = min((int)floor(t * FS), N_SAMPLES - 2);
            dt = t*FS - k ;
            k0 = i*N_ELEMENTOS*N_SAMPLES + e*N_SAMPLES ;

            barrier(CLK_GLOBAL_MEM_FENCE) ;
            temp = (float) matrix[k0 + k] ;
            barrier(CLK_GLOBAL_MEM_FENCE) ;
            a = ((float) matrix[k0 + k + 1] - temp) * dt + temp ;
            q += a * ap_dyn ;

            barrier(CLK_GLOBAL_MEM_FENCE) ;
            temp = (float) matrix_imag[k0 + k] ;
            barrier(CLK_GLOBAL_MEM_FENCE) ;
            b = ((float) matrix_imag[k0 + k + 1] - temp) * dt + temp ;
            q_imag += b * ap_dyn ;

            temp = hypot(a, b) + FLT_EPSILON ;  /* módulo del "fasor" */
            /* se suman las componentes de los fasores para cada A-scan */
            w += a/temp ;
            w_imag += b/temp ;

        }

        barrier(CLK_GLOBAL_MEM_FENCE) ;
        img[iz*NX + ix] = q ;

        barrier(CLK_GLOBAL_MEM_FENCE) ;
        img_imag[iz*NX + ix] = q_imag/N ;

        barrier(CLK_GLOBAL_MEM_FENCE) ;
        cohe[iz*NX + ix] = hypot(w, w_imag)/N ;

    }
}
