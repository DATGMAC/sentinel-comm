/**
 * SENTINEL COMM — Device-side handler contract.
 *
 * To add your own GPU operations, implement sc_user_dispatch() in exactly
 * one .cu file of your project (see src/user_handlers.cu for a working
 * example) and compile it with CUDA separable compilation (-rdc=true /
 * CMake CUDA_SEPARABLE_COMPILATION ON) so the device linker can resolve
 * it against the sentinel kernel.
 *
 * The dispatcher routes every command with opcode >= SC_OP_USER_BASE to
 * this function. ALL 256 threads of the sentinel block enter it together:
 * thread 0 is the dispatcher, the rest are workers — use tid/num_threads
 * to grid-stride over your data. The block is synchronized before and
 * after the call; do NOT return early from a subset of threads without
 * ensuring all threads leave the function (no divergent __syncthreads).
 *
 * Contract:
 *   - Return SC_OK (0) for success, nonzero to bump the error counter.
 *     The return value of thread 0 is the one that counts.
 *   - cmd->arg0..arg3 are yours: device pointers, sizes, packed scalars.
 *   - user_data is the pointer given to sc_launch().
 *   - Fence signaling is handled by the bus (SC_FLAG_FENCE_SIGNAL);
 *     handlers never touch fence_values directly.
 *   - Keep handlers bounded: while a handler runs, the bus dispatches
 *     nothing else, and the host may be spinning on a fence.
 */

#ifndef SENTINEL_COMM_DEVICE_CUH
#define SENTINEL_COMM_DEVICE_CUH

#include "sentinel_comm.h"

__device__ int sc_user_dispatch(const ScCommand* cmd, void* user_data,
                                int tid, int num_threads);

#endif /* SENTINEL_COMM_DEVICE_CUH */
