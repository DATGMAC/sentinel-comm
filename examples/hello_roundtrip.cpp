/**
 * SENTINEL COMM — Minimal usage example.
 *
 * Launches the bus, measures NOP dispatch latency, runs a GPU vector add
 * through the ring buffer (no kernel launches after boot!), and shuts
 * down cleanly.
 *
 * Build (part of the CMake project) or standalone:
 *   nvcc -rdc=true -I include src/sentinel_comm.cu src/user_handlers.cu \
 *        examples/hello_roundtrip.cpp -o hello_roundtrip
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "sentinel_comm.h"

#define OP_FILL_U32     (SC_OP_USER_BASE + 0)
#define OP_VEC_ADD_F32  (SC_OP_USER_BASE + 1)

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    /* ── Boot the bus ────────────────────────────────────────────────── */
    if (sc_init(/*device=*/0, /*ring_capacity=*/0, /*pin_cpu_core=*/-1) != SC_OK) {
        fprintf(stderr, "init failed: %s\n", sc_error());
        return 1;
    }

    /* IMPORTANT: allocate ALL device memory BEFORE sc_launch(). While the
     * sentinel is resident, cudaFree / graph capture / some cudaMalloc
     * paths deadlock (they wait for device idle — see README). */
    const uint64_t N = 1 << 20;
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    std::vector<float> h_a(N, 1.5f), h_b(N, 2.25f), h_out(N, 0.f);
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    if (sc_launch(nullptr) != SC_OK) {
        fprintf(stderr, "launch failed: %s\n", sc_error());
        return 1;
    }
    printf("sentinel resident: %d\n", sc_is_running());

    /* ── 1. Latency probe: fenced NOP round-trip ─────────────────────── */
    double t0 = now_us();
    sc_submit(SC_OP_NOP, SC_FLAG_FENCE_SIGNAL, 0, 0, 0, 0, /*fence=*/1);
    sc_wait_fence(1, 1, 1000000000ULL);
    printf("NOP submit→fence round-trip: %.1f us (host-observed)\n", now_us() - t0);

    ScStats st;
    sc_get_stats(&st);
    printf("GPU-side dispatch latency:   %llu ns\n",
           (unsigned long long)st.last_dispatch_ns);

    /* ── 2. Real GPU work through the ring: c = a + b ────────────────── */
    t0 = now_us();
    sc_submit(OP_VEC_ADD_F32, SC_FLAG_FENCE_SIGNAL,
              (uint64_t)d_a, (uint64_t)d_b, (uint64_t)d_out, N, /*fence=*/2);
    sc_wait_fence(2, 1, 1000000000ULL);
    printf("vec_add(%llu floats) via ring: %.1f us\n",
           (unsigned long long)N, now_us() - t0);

    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (uint64_t i = 0; i < N; i++)
        if (h_out[i] != 3.75f) { ok = false; break; }
    printf("vec_add result: %s\n", ok ? "CORRECT" : "WRONG");

    /* ── Shutdown ────────────────────────────────────────────────────── */
    sc_get_stats(&st);
    printf("stats: %llu processed, %llu errors\n",
           (unsigned long long)st.commands_processed,
           (unsigned long long)st.commands_errors);

    sc_shutdown();
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);  /* safe now: kernel gone */
    printf("clean shutdown\n");
    return ok ? 0 : 1;
}
