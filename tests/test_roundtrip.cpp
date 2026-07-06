/**
 * SENTINEL COMM — Smoke test.
 *
 * Exercises the full lifecycle: init → launch → NOP heartbeats → user
 * opcodes with data verification → error counting for unknown opcodes →
 * fence semantics → clean shutdown. Exit 0 on success.
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "sentinel_comm.h"

#define OP_FILL_U32     (SC_OP_USER_BASE + 0)
#define OP_VEC_ADD_F32  (SC_OP_USER_BASE + 1)

static int g_failures = 0;

#define CHECK(cond, name) do { \
    bool _ok = (cond); \
    printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", name); \
    if (!_ok) g_failures++; \
} while (0)

int main() {
    printf("== sentinel-comm smoke test ==\n");

    CHECK(sc_init(0, 0, -1) == SC_OK, "init");

    /* All device allocations BEFORE launch (see README: the one rule) */
    const uint64_t N = 4096;
    uint32_t* d_buf;
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_buf, N * sizeof(uint32_t));
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    std::vector<float> h(N, 2.0f);
    cudaMemcpy(d_a, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(sc_launch(nullptr) == SC_OK, "launch");
    CHECK(sc_is_running() == 1, "is_running");

    /* Fenced NOP round-trips */
    for (int i = 0; i < 10; i++) {
        sc_submit(SC_OP_NOP, SC_FLAG_FENCE_SIGNAL, 0, 0, 0, 0, 1);
    }
    CHECK(sc_wait_fence(1, 10, 2000000000ULL) == SC_OK, "10 fenced NOPs");
    CHECK(sc_fence_value(1) == 10, "fence value == 10");

    /* FILL_U32 writes verifiable data */
    sc_submit(OP_FILL_U32, SC_FLAG_FENCE_SIGNAL,
              (uint64_t)d_buf, N, 0xDEADBEEF, 0, 2);
    CHECK(sc_wait_fence(2, 1, 2000000000ULL) == SC_OK, "fill_u32 fence");
    std::vector<uint32_t> out(N);
    cudaMemcpy(out.data(), d_buf, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    bool fill_ok = true;
    for (uint64_t i = 0; i < N; i++)
        if (out[i] != 0xDEADBEEF) { fill_ok = false; break; }
    CHECK(fill_ok, "fill_u32 data correct");

    /* VEC_ADD produces 4.0 everywhere */
    sc_submit(OP_VEC_ADD_F32, SC_FLAG_FENCE_SIGNAL,
              (uint64_t)d_a, (uint64_t)d_b, (uint64_t)d_out, N, 3);
    CHECK(sc_wait_fence(3, 1, 2000000000ULL) == SC_OK, "vec_add fence");
    std::vector<float> fout(N);
    cudaMemcpy(fout.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool add_ok = true;
    for (uint64_t i = 0; i < N; i++)
        if (fout[i] != 4.0f) { add_ok = false; break; }
    CHECK(add_ok, "vec_add data correct");

    /* Unknown user opcode must bump the error counter, not crash */
    ScStats before, after;
    sc_get_stats(&before);
    sc_submit(SC_OP_USER_BASE + 99, SC_FLAG_FENCE_SIGNAL, 0, 0, 0, 0, 4);
    CHECK(sc_wait_fence(4, 1, 2000000000ULL) == SC_OK, "unknown opcode still fences");
    sc_get_stats(&after);
    CHECK(after.commands_errors == before.commands_errors + 1,
          "unknown opcode counted as error");

    /* Stats sanity */
    CHECK(after.commands_processed >= 13, "commands_processed >= 13");
    CHECK(after.last_dispatch_ns > 0 && after.last_dispatch_ns < 1000000,
          "dispatch latency sane (<1ms)");

    CHECK(sc_shutdown() == SC_OK, "shutdown");
    CHECK(sc_is_running() == 0, "stopped after shutdown");

    cudaFree(d_buf); cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    printf("== %s (%d failures) ==\n",
           g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
