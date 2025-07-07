#include "mex.h"
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1 || !mxIsUint64(prhs[0])) {
        mexErrMsgTxt("Se necesita un puntero uint64 como entrada.");
    }

    uint64_t ptr_val = *((uint64_t*)mxGetData(prhs[0]));
    int* data = (int*)(uintptr_t)ptr_val;

    // Ejemplo: modificar los primeros 10 enteros
    for (int i = 0; i < 10; ++i) {
        data[i] = i * 10;
    }
}
