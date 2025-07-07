#include "mex.h"
#include <cuda_runtime.h>
#include <math.h>

__device__ void shift_izq(short *x, int TAPS) {
    for (int j = 0; j < TAPS; j++) {
        x[j] = x[j + 1];
    }
    x[TAPS] = 0;
}

__device__ float multisum(short *x, float *coef, int TAPS) {
    float q = 0;
    for (int j = 0; j < (TAPS + 1); j++) {
        q += coef[TAPS - j] * x[j];
    }
    return q;
}

__global__ void filt_kernel_2(const short *datain, const float *coef_g, short *dataout,
                              int TAPS, int N_ELEMENTOS, int N_SAMPLES) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (e >= N_ELEMENTOS || r >= N_ELEMENTOS) return;

    extern __shared__ float coef[];
    short *x = new short[TAPS + 1];

    for (int l = threadIdx.x; l <= TAPS; l += blockDim.x) {
        coef[l] = coef_g[l];
    }
    __syncthreads();

    int i = N_ELEMENTOS * N_SAMPLES * e + N_SAMPLES * r;
    int l0 = TAPS / 2;

    for (int l = 0; l <= l0; l++) {
        x[l + l0] = datain[i + l];
    }

    dataout[i] = (short)rintf(multisum(x, coef, TAPS));

    int lmax = N_SAMPLES - l0;
    for (int l = 1; l < lmax; l++) {
        shift_izq(x, TAPS);
        x[TAPS] = datain[i + l + l0];
        dataout[i + l] = (short)rintf(multisum(x, coef, TAPS));
    }

    for (int l = lmax; l < N_SAMPLES; l++) {
        shift_izq(x, TAPS);
        dataout[i + l] = (short)rintf(multisum(x, coef, TAPS));
    }

    delete[] x;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    short *datain = (short *)mxGetData(prhs[0]);
    float *coef = (float *)mxGetData(prhs[1]);
    int TAPS = (int)mxGetScalar(prhs[2]);
    int N_ELEMENTOS = (int)mxGetScalar(prhs[3]);
    int N_SAMPLES = (int)mxGetScalar(prhs[4]);

    mwSize totalSize = N_ELEMENTOS * N_ELEMENTOS * N_SAMPLES;
    plhs[0] = mxCreateNumericMatrix(1, totalSize, mxINT16_CLASS, mxREAL);
    short *dataout = (short *)mxGetData(plhs[0]);

    short *d_datain, *d_dataout;
    float *d_coef;

    cudaMalloc(&d_datain, totalSize * sizeof(short));
    cudaMalloc(&d_dataout, totalSize * sizeof(short));
    cudaMalloc(&d_coef, (TAPS + 1) * sizeof(float));

    cudaMemcpy(d_datain, datain, totalSize * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coef, coef, (TAPS + 1) * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N_ELEMENTOS + 15) / 16, (N_ELEMENTOS + 15) / 16);
    filt_kernel_2<<<blocks, threads, (TAPS + 1) * sizeof(float)>>>(d_datain, d_coef, d_dataout,
                                                                   TAPS, N_ELEMENTOS, N_SAMPLES);

    cudaMemcpy(dataout, d_dataout, totalSize * sizeof(short), cudaMemcpyDeviceToHost);

    cudaFree(d_datain);
    cudaFree(d_dataout);
    cudaFree(d_coef);
}
