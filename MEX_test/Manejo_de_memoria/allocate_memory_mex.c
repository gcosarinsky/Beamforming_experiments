#include "mex.h"
#include <stdlib.h>
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1 || !mxIsDouble(prhs[0])) {
        mexErrMsgTxt("Se necesita un argumento: tamaño en bytes (double).");
    }

    size_t size = (size_t)mxGetScalar(prhs[0]);
    void* ptr = malloc(size);

    if (ptr == NULL) {
        mexErrMsgTxt("No se pudo reservar memoria.");
    }

    // Crear un uint64 para devolver la dirección
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)(uintptr_t)ptr;
}
