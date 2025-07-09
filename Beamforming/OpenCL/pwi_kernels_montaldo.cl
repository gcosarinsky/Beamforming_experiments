__kernel void pwi_0(__global short *matrix,
                  __global short *matrix_imag,
                  __global float *img,
                  __global float *img_imag,
                  __global float *cohe,
                  __global float *angles,
                  __global float *ret_ini) {

    ushort iz = get_global_id(0);
    ushort ix = get_global_id(1);

    ushort f_idx = iz*NX + ix ;
    float xf = X0_ROI + X_STEP * ix ;
    float zf = Z0_ROI + Z_STEP * iz ; /* Z POSITIVE DOWNWARDS  */
    float x_rx ;
    float t1, t2 ;
    float t, dt, temp, theta, ap_dyn ;
    uint k, k0 ;
    uint N = N_ELEMENTOS*N_ANGLES ;
    float a, b, q = 0, q_imag = 0, w = 0, w_imag = 0 ;

    for (ushort i=0; i < N_ANGLES; i++) {

        theta = angles[i] ;
        /*
        t = 0 is when wavefront passes through array element 0, at -X_0. But SITAU starts acquiring after first
        firing, which is element 0 for angle >= 0, but element 128 (last) for angle < 0.
        ret_ini compensates this
        */
        t1 = ((xf + X_0) * sin(theta) + zf * cos(theta))/C1 - ret_ini[i] ; /* X_0 is half array aperture */

        for (ushort e=0; e < N_ELEMENTOS; e++) {
            x_rx = e*PITCH - X_0;
            t2 = hypot(x_rx - xf, zf)/C1 ;
            ap_dyn = zf/(fabs(x_rx - xf) + FLT_EPSILON) > BFD ;
            t = t1 + t2;
            t = t * step(0, t);  /* First sample must be 0 !!! */
            k = fmin(floor(t * FS), (float)(N_SAMPLES - 2)); /* resto 2 para evitar que k+1 = N_SAMPLES */
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
        img[iz*NX + ix] = q/N ;

        barrier(CLK_GLOBAL_MEM_FENCE) ;
        img_imag[iz*NX + ix] = q_imag/N ;

        barrier(CLK_GLOBAL_MEM_FENCE) ;
        cohe[iz*NX + ix] = hypot(w, w_imag)/N ;

    }
}

/* devuelve las partes real e imaginaria del vector coherencia*/
__kernel void pwi_1(__global short *matrix,
                  __global short *matrix_imag,
                  __global float *img,
                  __global float *img_imag,
                  __global float *angles,
                  __global float *ret_ini) {

    ushort iz = get_global_id(0);
    ushort ix = get_global_id(1);

    ushort f_idx = iz*NX + ix ;
    float xf = X0_ROI + X_STEP * ix ;
    float zf = Z0_ROI + Z_STEP * iz ; /* Z POSITIVE DOWNWARDS  */
    float x_rx ;
    float t1, t2 ;
    float t, dt, temp, theta, ap_dyn ;
    uint k, k0 ;
    uint N = N_ELEMENTOS*N_ANGLES ;
    float a, b, q = 0, q_imag = 0, w = 0, w_imag = 0 ;

    for (ushort i=0; i < N_ANGLES; i++) {

        theta = angles[i] ;
        t1 = ((xf + X_0) * sin(theta) + zf * cos(theta))/C1 - ret_ini[i] ; /* X_0 is half array aperture */

        for (ushort e=0; e < N_ELEMENTOS; e++) {
            x_rx = e*PITCH - X_0;
            t2 = hypot(x_rx - xf, zf)/C1 ;
            ap_dyn = zf/(fabs(x_rx - xf) + FLT_EPSILON) > BFD ;
            t = t1 + t2;
            t = t * step(0, t);  /* First sample must be 0 !!! */
            k = fmin(floor(t * FS), (float)(N_SAMPLES - 2)); /* resto 2 para evitar que k+1 = N_SAMPLES */
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
        img[iz*NX + ix] = w/N ;

        barrier(CLK_GLOBAL_MEM_FENCE) ;
        img_imag[iz*NX + ix] = w_imag/N ;
    }
}