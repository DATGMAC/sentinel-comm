/**
 * SENTINEL COMM — Example user handlers.
 *
 * THIS FILE IS YOURS. Replace or extend the opcodes below with your own
 * GPU operations; this is the extension point of the bus. Keep exactly
 * one sc_user_dispatch() definition in your project.
 *
 * Every command with opcode >= SC_OP_USER_BASE lands here with all 256
 * sentinel threads. Use tid/num_threads to grid-stride over your data.
 */

#include "sentinel_comm_device.cuh"

/* Example opcodes — rename/replace freely */
#define OP_FILL_U32     (SC_OP_USER_BASE + 0)  /* arg0=dst, arg1=count, arg2=value */
#define OP_VEC_ADD_F32  (SC_OP_USER_BASE + 1)  /* arg0=a, arg1=b, arg2=out, arg3=count */

__device__ static int handle_fill_u32(const ScCommand* cmd, int tid, int nt) {
    uint32_t* dst = (uint32_t*)cmd->arg0;
    uint64_t count = cmd->arg1;
    uint32_t value = (uint32_t)cmd->arg2;
    if (!dst) return -1;
    for (uint64_t i = tid; i < count; i += nt)
        dst[i] = value;
    __threadfence();
    return SC_OK;
}

__device__ static int handle_vec_add_f32(const ScCommand* cmd, int tid, int nt) {
    const float* a = (const float*)cmd->arg0;
    const float* b = (const float*)cmd->arg1;
    float* out = (float*)cmd->arg2;
    uint64_t count = cmd->arg3;
    if (!a || !b || !out) return -1;
    for (uint64_t i = tid; i < count; i += nt)
        out[i] = a[i] + b[i];
    __threadfence();
    return SC_OK;
}

__device__ int sc_user_dispatch(const ScCommand* cmd, void* user_data,
                                int tid, int num_threads) {
    (void)user_data;  /* forward your device context here if you need one */

    switch (cmd->opcode) {
        case OP_FILL_U32:    return handle_fill_u32(cmd, tid, num_threads);
        case OP_VEC_ADD_F32: return handle_vec_add_f32(cmd, tid, num_threads);
        default:             return -1;  /* unknown user opcode */
    }
}
