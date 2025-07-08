/* funcion para correr (shiftear) los elementos de un array un lugar hacia la izquierda */
void shift_izq(short *x) {

    for (uint j=0; j < TAPS; j++) {

        x[j] = x[j + 1] ;
    }

    x[TAPS] = 0 ;
}

/* multiplica y suma "x" con "coef" flipeado */
float multisum(short *x, float *coef) {

    float q = 0 ;
    for (uint j=0; j < (TAPS + 1); j++) {

            q += coef[TAPS - j] * x[j] ;
    }

    return q ;
}


/* datain con tamaño distinto a dataout, para el caso en que hay menos elementos que canales*/
__kernel void filt_kernel_1(__global const short *datain,
                          __global const float *coef_g,
                          __global short *dataout) {

    ushort e = get_global_id(0) ;
    ushort r = get_global_id(1) ;

    float coef[TAPS + 1] ;
    short x[TAPS + 1] ;
    int i = N_CH*N_SAMPLES*e + N_SAMPLES*r ; /* índice de primer sample del ascan en datain*/
    int j = N_ELEMENTOS*N_SAMPLES*e + N_SAMPLES*r ; /* índice de primer sample del ascan en dataout*/

    /* copiar coeficientes del filtro en memoria privada */
    ushort l0 = TAPS/2 ;
    ushort lmax = TAPS + 1 ;
    for (ushort l=0; l < lmax; l++) {
        coef[l] = coef_g[l] ;
    }

    /* calcular transitorio:
    copiar las primeras muestras (desde 0 hasta TAPS/2) al array x, colocandolas en él desde TAPS/2 hasta el
    final */
    for (ushort l=0; l <= l0; l++) {
        x[l + l0] = datain[i + l] ;
    }

    /* calcula primera muestra de salida */
    dataout[j] = (short) rint(multisum(x, coef)) ;

    /* continua hasta traer la última muestra del A-scan*/
    lmax = N_SAMPLES - l0 ;
    for (ushort l=1; l < lmax; l++) {
        shift_izq(x) ;
        x[TAPS] = (float) datain[i + l + l0] ;
        dataout[j + l] = (short) rint(multisum(x, coef)) ;
    }

    for (ushort l=lmax; l < N_SAMPLES; l++) {
        shift_izq(x) ;
        dataout[j + l] = (short) rint(multisum(x, coef)) ;
    }
}


/* datain con el mismo tamaño que dataout */
__kernel void filt_kernel_2(__global const short *datain,
                          __global const float *coef_g,
                          __global short *dataout) {

    ushort e = get_global_id(0) ;
    ushort r = get_global_id(1) ;

    float coef[TAPS + 1] ;
    short x[TAPS + 1] ;
    int i = N_ELEMENTOS*N_SAMPLES*e + N_SAMPLES*r ; /* índice de primer sample del ascan en dataout*/

    /* copiar coeficientes del filtro en memoria privada */
    ushort l0 = TAPS/2 ;
    ushort lmax = TAPS + 1 ;
    for (ushort l=0; l < lmax; l++) {
        coef[l] = coef_g[l] ;
    }

    /* calcular transitorio:
    copiar las primeras muestras (desde 0 hasta TAPS/2) al array x, colocandolas en él desde TAPS/2 hasta el
    final */
    for (ushort l=0; l <= l0; l++) {
        x[l + l0] = datain[i + l] ;
    }

    /* calcula primera muestra de salida */
    dataout[i] = (short) rint(multisum(x, coef)) ;

    /* continua hasta traer la última muestra del A-scan*/
    lmax = N_SAMPLES - l0 ;
    for (ushort l=1; l < lmax; l++) {
        shift_izq(x) ;
        x[TAPS] = (float) datain[i + l + l0] ;
        dataout[i + l] = (short) rint(multisum(x, coef)) ;
    }

    for (ushort l=lmax; l < N_SAMPLES; l++) {
        shift_izq(x) ;
        dataout[i + l] = (short) rint(multisum(x, coef)) ;
    }

    /* forzar esto para el tema de los bordes en el beamforming */
    dataout[i] = 0 ;
    dataout[i + N_SAMPLES - 2] = 0 ;
    dataout[i + N_SAMPLES - 1] = 0 ;
}
