#define MAX_FIR_SIZE 64
#define FLT_EPSILON 1.1920929e-7f

enum FloatParams {
    FS,
    C1,
    PITCH,
    F1,
    F2,
    BFD,
    X_STEP,
    Z_STEP,
    X0_ROI,
    Z0_ROI,
    T_START,
    X_0,
    FLOAT_PARAMS_COUNT
};

enum IntParams {
    TAPS,
    N_ELEMENTOS,
    N_ANGLES,
    NX,
    NZ,
    N_SAMPLES,
    INT_PARAMS_COUNT
};

extern "C" __global__ void print_params(const float *float_params, const int *int_params) {
    int tid = threadIdx.x;

    if (tid == 0) {
        printf("Float Params:\\n");
        printf("FS = %f\\n", float_params[FS]);
        printf("C1 = %f\\n", float_params[C1]);
        printf("PITCH = %f\\n", float_params[PITCH]);
        printf("F1 = %f\\n", float_params[F1]);
        printf("F2 = %f\\n", float_params[F2]);
        printf("BFD = %f\\n", float_params[BFD]);
        printf("X_STEP = %f\\n", float_params[X_STEP]);
        printf("Z_STEP = %f\\n", float_params[Z_STEP]);
        printf("X0_ROI = %f\\n", float_params[X0_ROI]);
        printf("Z0_ROI = %f\\n", float_params[Z0_ROI]);
        printf("T_START = %f\\n", float_params[T_START]);
        printf("X_0 = %f\\n", float_params[X_0]);

        printf("Int Params:\\n");
        printf("TAPS = %d\\n", int_params[TAPS]);
        printf("N_ELEMENTOS = %d\\n", int_params[N_ELEMENTOS]);
        printf("N_ANGLES = %d\\n", int_params[N_ANGLES]);
        printf("NX = %d\\n", int_params[NX]);
        printf("NZ = %d\\n", int_params[NZ]);
        printf("N_SAMPLES = %d\\n", int_params[N_SAMPLES]);
    }
}