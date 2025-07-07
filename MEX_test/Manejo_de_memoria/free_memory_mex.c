#include "mex.h"
#include <stdlib.h>
#include <stdint.h>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1 || !mxIsUint64(prhs[0])) {
        mexErrMsgTxt("Se necesita un puntero uint64 como entrada.");
    }

    uint64_t ptr_val = *((uint64_t*)mxGetData(prhs[0]));
    void* ptr = (void*)(uintptr_t)ptr_val;

    free(ptr);
}
