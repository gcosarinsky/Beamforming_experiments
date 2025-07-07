#include "mex.h"
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2 || !mxIsUint64(prhs[0]) || !mxIsDouble(prhs[1])) {
        mexErrMsgTxt("Uso: read_memory_mex(ptr, num_elements)");
    }

    // Obtener el puntero
    uint64_t ptr_val = *((uint64_t*)mxGetData(prhs[0]));
    int* data = (int*)(uintptr_t)ptr_val;

    // Obtener el número de elementos a leer
    size_t num_elements = (size_t)mxGetScalar(prhs[1]);

    // Crear array MATLAB para devolver los datos
    plhs[0] = mxCreateNumericMatrix(1, num_elements, mxINT32_CLASS, mxREAL);
    int* out = (int*)mxGetData(plhs[0]);

    // Copiar datos desde la memoria
    for (size_t i = 0; i < num_elements; ++i) {
        out[i] = data[i];
    }
}
