/**
 * SENTINEL COMM — Dispatch latency benchmark.
 *
 * Answers the first question every reviewer asks: "why not CUDA Graphs?"
 * with numbers. Measures host-observed round-trip latency (submit work,
 * wait until the GPU finished it) for three mechanisms doing the same
 * nothing:
 *
 *   1. plain kernel launch   : empty_kernel<<<>>> + cudaStreamSynchronize
 *   2. CUDA graph replay     : cudaGraphLaunch    + cudaStreamSynchronize
 *   3. sentinel-comm         : sc_submit(NOP)     + busy-wait on fence
 *
 * Plus the two one-way costs unique to the bus:
 *   - sc_submit() call cost (CPU side, no wait)
 *   - GPU-side dispatch latency (decode→complete, from kernel counters)
 *
 * NOTE ON ORDERING: graph capture/instantiation deadlocks while a
 * persistent kernel is resident (see README, The One Rule) — so all
 * baseline benchmarks run and release their resources BEFORE sc_launch().
 */

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#include "sentinel_comm.h"

__global__ void empty_kernel() {}

static inline double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(
        steady_clock::now().time_since_epoch()).count();
}

struct Dist { double p50, p99, min, mean; };

static Dist summarize(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    return {
        v[v.size() / 2],
        v[(size_t)(v.size() * 0.99)],
        v.front(),
        sum / v.size(),
    };
}

static void report(const char* name, Dist d) {
    printf("%-28s  p50 %8.2f us   p99 %8.2f us   min %8.2f us   mean %8.2f us\n",
           name, d.p50, d.p99, d.min, d.mean);
}

int main() {
    const int WARMUP = 200;
    const int ITERS = 2000;
    std::vector<double> samples;
    samples.reserve(ITERS);

    cudaFree(0);  /* force context creation */
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    printf("== dispatch latency: %d iterations (after %d warm-up) ==\n\n",
           ITERS, WARMUP);

    /* ── 1. Plain kernel launch + sync ──────────────────────────────── */
    for (int i = 0; i < WARMUP; i++) {
        empty_kernel<<<1, 256, 0, stream>>>();
        cudaStreamSynchronize(stream);
    }
    samples.clear();
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        empty_kernel<<<1, 256, 0, stream>>>();
        cudaStreamSynchronize(stream);
        samples.push_back(now_us() - t0);
    }
    report("kernel launch + sync", summarize(samples));

    /* ── 2. CUDA graph replay + sync ────────────────────────────────── */
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    empty_kernel<<<1, 256, 0, stream>>>();
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    for (int i = 0; i < WARMUP; i++) {
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
    }
    samples.clear();
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
        samples.push_back(now_us() - t0);
    }
    report("CUDA graph replay + sync", summarize(samples));

    /* ── Burst throughput: 1000 tiny ops pipelined, one final sync ────
     * This is the bus's home turf: the CPU cost of enqueueing dominates
     * when ops are small and frequent. */
    const int BURST = 1000;
    double t0 = now_us();
    for (int i = 0; i < BURST; i++)
        empty_kernel<<<1, 256, 0, stream>>>();
    cudaStreamSynchronize(stream);
    double launch_burst = now_us() - t0;

    t0 = now_us();
    for (int i = 0; i < BURST; i++)
        cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    double graph_burst = now_us() - t0;

    printf("\n== burst: %d pipelined ops, one final sync ==\n\n", BURST);
    printf("%-28s  total %8.0f us   per-op %6.2f us\n",
           "kernel launches", launch_burst, launch_burst / BURST);
    printf("%-28s  total %8.0f us   per-op %6.2f us\n",
           "CUDA graph replays", graph_burst, graph_burst / BURST);

    /* Release graph resources BEFORE the sentinel becomes resident */
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaDeviceSynchronize();

    /* ── 3. sentinel-comm: submit + fence round-trip ────────────────── */
    if (sc_init(0, 0, /*pin_cpu_core=*/-1) != SC_OK ||
        sc_launch(nullptr) != SC_OK) {
        fprintf(stderr, "sentinel boot failed: %s\n", sc_error());
        return 1;
    }

    uint64_t expected = 0;
    for (int i = 0; i < WARMUP; i++) {
        expected++;
        sc_submit(SC_OP_NOP, SC_FLAG_FENCE_SIGNAL, 0, 0, 0, 0, 1);
        while (sc_fence_value(1) < expected) { /* busy-wait */ }
    }
    samples.clear();
    for (int i = 0; i < ITERS; i++) {
        expected++;
        double t0 = now_us();
        sc_submit(SC_OP_NOP, SC_FLAG_FENCE_SIGNAL, 0, 0, 0, 0, 1);
        while (sc_fence_value(1) < expected) { /* busy-wait */ }
        samples.push_back(now_us() - t0);
    }
    report("sentinel submit + fence", summarize(samples));

    /* ── 3b. submit() call cost alone (fire-and-forget) ─────────────── */
    samples.clear();
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        sc_submit(SC_OP_NOP, SC_FLAG_NONE, 0, 0, 0, 0, 0);
        samples.push_back(now_us() - t0);
        if (sc_pending() > 2048) {           /* don't outrun the ring */
            while (sc_pending() > 0) { }
        }
    }
    while (sc_pending() > 0) { }
    report("sc_submit only (enqueue)", summarize(samples));

    /* ── 3c. GPU-side dispatch latency (from kernel counters) ───────── */
    ScStats st;
    sc_get_stats(&st);
    printf("%-28s  %llu ns (decode->complete, last command)\n",
           "GPU-side dispatch", (unsigned long long)st.last_dispatch_ns);

    /* ── 3d. Burst through the ring: submit flat-out, fence at the end */
    double tb = now_us();
    for (int i = 0; i < BURST - 1; i++) {
        while (sc_submit(SC_OP_NOP, SC_FLAG_NONE, 0, 0, 0, 0, 0) ==
               SC_ERR_RING_FULL) { /* GPU draining */ }
    }
    uint64_t base = sc_fence_value(2);
    sc_submit(SC_OP_NOP, SC_FLAG_FENCE_SIGNAL, 0, 0, 0, 0, 2);
    while (sc_fence_value(2) < base + 1) { }
    double ring_burst = now_us() - tb;
    printf("%-28s  total %8.0f us   per-op %6.2f us\n",
           "sentinel ring burst", ring_burst, ring_burst / BURST);

    sc_shutdown();
    printf("\ndone: %llu commands, %llu errors\n",
           (unsigned long long)st.commands_processed,
           (unsigned long long)st.commands_errors);
    return 0;
}
