/**
 * SENTINEL COMM — Persistent kernel + host bridge (single translation unit).
 *
 * The complete bus: a persistent GPU kernel that spin-polls a lock-free
 * SPSC ring buffer in pinned host memory, and the C host API that feeds
 * it. User operations live in a separate .cu implementing
 * sc_user_dispatch() (device-linked, see sentinel_comm_device.cuh).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <thread>
#include <chrono>

#include <sched.h>
#include <pthread.h>

#include "sentinel_comm.h"
#include "sentinel_comm_device.cuh"

/* ═══════════════════════════════════════════════════════════════════════════
 *  DEVICE SIDE — The persistent sentinel kernel
 * ═══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__ uint64_t sc_gpu_clock_ns() {
    uint64_t ticks;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks));
    return ticks;
}

__global__ void sc_sentinel_kernel(
    ScRingHeader*  ring_header,
    ScCommand*     command_slots,
    volatile int*  shutdown_flag,
    void*          user_data
) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_threads = blockDim.x * gridDim.x;
    const int is_dispatcher = (tid == 0);

    /* Batched dequeue: the ring lives in pinned HOST memory, so every
     * access is a PCIe transaction. Dequeuing one command at a time costs
     * several non-posted reads + atomics PER COMMAND (~10µs/op measured).
     * Instead: read the head once, cooperatively fetch up to SC_MAX_BATCH
     * commands into shared memory with coalesced 16-byte reads, dispatch
     * from shared, and touch tail/counters once per batch. */
    #define SC_MAX_BATCH 64
    __shared__ ScCommand batch_cmds[SC_MAX_BATCH];
    __shared__ uint32_t  batch_n;
    __shared__ uint32_t  batch_tail;
    __shared__ int       should_exit;
    __shared__ int       dispatch_err;

    /* Idle backoff: polling flat-out floods PCIe and starves co-resident
     * CUDA work. Back off exponentially while idle (200ns → 100µs) and
     * snap back to hot polling the moment a command arrives. */
    __shared__ uint32_t idle_backoff_ns;

    if (is_dispatcher) {
        should_exit = 0;
        idle_backoff_ns = 200;
    }
    __syncthreads();

    const uint64_t kernel_start = sc_gpu_clock_ns();

    while (!should_exit) {

        /* ── Poll once, claim a batch (dispatcher only) ─────────────── */
        if (is_dispatcher) {
            /* write_head is volatile mapped host memory: reads are
             * uncached PCIe transactions. The host fences before
             * advancing the head, so the command bytes are already
             * globally visible when we observe the new value. */
            uint32_t head = ring_header->write_head;
            uint32_t tail = ring_header->read_tail;
            uint32_t cap  = ring_header->capacity;
            uint32_t pending = (head - tail + cap) % cap;
            batch_n = pending < SC_MAX_BATCH ? pending : SC_MAX_BATCH;
            batch_tail = tail;
            if (batch_n > 0) idle_backoff_ns = 200;
        }
        __syncthreads();

        if (batch_n > 0) {
            /* ── Cooperative fetch: batch_n × 64B over PCIe, coalesced ─ */
            {
                uint32_t cap = ring_header->capacity;
                uint32_t total_vec = batch_n * 4;   /* 4 × uint4 per command */
                for (uint32_t v = tid; v < total_vec; v += num_threads) {
                    uint32_t j = v >> 2, k = v & 3;
                    const uint4* src =
                        (const uint4*)&command_slots[(batch_tail + j) % cap];
                    ((uint4*)&batch_cmds[j])[k] = src[k];
                }
            }
            __syncthreads();

            /* ── Dispatch each command from shared memory ───────────── */
            uint32_t processed = 0;
            for (uint32_t j = 0; j < batch_n && !should_exit; j++) {
                const ScCommand* cmd = &batch_cmds[j];
                uint64_t t0 = (j == batch_n - 1) ? sc_gpu_clock_ns() : 0;
                if (is_dispatcher) dispatch_err = 0;
                __syncthreads();

                switch (cmd->opcode) {
                    case SC_OP_NOP:
                        break;

                    case SC_OP_SHUTDOWN:
                        if (is_dispatcher) should_exit = 1;
                        break;

                    default: {
                        int err = SC_OK;
                        if (cmd->opcode >= SC_OP_USER_BASE) {
                            err = sc_user_dispatch(cmd, user_data,
                                                   tid, num_threads);
                        } else {
                            err = -1;  /* Reserved opcode with no handler */
                        }
                        if (is_dispatcher && err != SC_OK) dispatch_err = 1;
                        break;
                    }
                }
                __syncthreads();

                if (is_dispatcher) {
                    if (dispatch_err) {
                        atomicAdd((unsigned long long*)&ring_header->commands_errors, 1ULL);
                    }
                    if (cmd->flags & SC_FLAG_FENCE_SIGNAL) {
                        uint32_t fid = cmd->fence_id;
                        if (fid < SC_FENCE_SLOTS) {
                            atomicAdd((unsigned long long*)&ring_header->fence_values[fid], 1ULL);
                            __threadfence_system();
                        }
                    }
                    if (j == batch_n - 1) {
                        ring_header->last_dispatch_ns = sc_gpu_clock_ns() - t0;
                    }
                }
                processed++;
                __syncthreads();
            }

            /* ── Batch bookkeeping: one tail/counter update per batch ─ */
            if (is_dispatcher) {
                atomicAdd((unsigned long long*)&ring_header->commands_processed,
                          (unsigned long long)processed);
                ring_header->kernel_uptime_ns = sc_gpu_clock_ns() - kernel_start;
                ring_header->read_tail =
                    (batch_tail + processed) % ring_header->capacity;
                __threadfence_system();
            }
        } else {
            #if __CUDA_ARCH__ >= 700
                __nanosleep(idle_backoff_ns);
            #endif
            if (is_dispatcher && idle_backoff_ns < 100000) {
                idle_backoff_ns *= 2;   /* Cap idle latency at 100µs */
            }
        }
        __syncthreads();
    }

    if (is_dispatcher) {
        ring_header->kernel_uptime_ns = sc_gpu_clock_ns() - kernel_start;
        *shutdown_flag = 2;  /* Clean-exit signal for the host */
        __threadfence_system();
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  HOST SIDE — Bus lifecycle and submission
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct ScHostState {
    int            initialized;
    int            running;
    int            device_id;
    uint32_t       capacity;
    void*          control_host;     /* pinned mapped ring (host view)   */
    void*          control_dev;      /* same memory, device view         */
    ScRingHeader*  hdr;              /* host view                        */
    ScCommand*     slots;            /* host view                        */
    volatile int*  shutdown_host;
    int*           shutdown_dev;
    cudaStream_t   stream;
    uint16_t       seq;
} ScHostState;

static ScHostState g_sc = {};
static char        g_err[256] = {0};

#define SC_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        snprintf(g_err, sizeof(g_err), "CUDA error at %s:%d: %s", \
                 __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return SC_ERR_CUDA_FAILURE; \
    } \
} while (0)

static uint64_t sc_host_clock_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t sc_checksum(const ScCommand* cmd) {
    const uint8_t* data = (const uint8_t*)cmd;
    uint64_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < sizeof(ScCommand) - sizeof(uint64_t); i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320ULL & (-(crc & 1)));
    }
    return crc ^ 0xFFFFFFFF;
}

/* Optional: pin the submitting thread to one core with RT priority so
 * submission latency doesn't jitter with scheduler preemption. Off by
 * default — a library should not hijack the host's scheduling policy. */
static void sc_pin_host_thread(int core) {
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
        fprintf(stderr, "[sentinel-comm] warning: SCHED_FIFO unavailable "
                        "(needs CAP_SYS_NICE); submission may jitter\n");
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        fprintf(stderr, "[sentinel-comm] warning: could not pin to core %d\n", core);
    }
}

extern "C" int sc_init(int device_id, uint32_t ring_capacity, int pin_cpu_core) {
    if (g_sc.initialized) {
        snprintf(g_err, sizeof(g_err), "already initialized");
        return SC_ERR_ALREADY_RUNNING;
    }
    memset(&g_sc, 0, sizeof(g_sc));
    g_sc.device_id = device_id;
    g_sc.capacity = ring_capacity ? ring_capacity : SC_RING_CAPACITY_DEFAULT;

    if (pin_cpu_core >= 0) sc_pin_host_thread(pin_cpu_core);

    SC_CUDA_CHECK(cudaSetDevice(device_id));

    /* Pinned, mapped host memory — NOT cudaMallocManaged. Managed pages
     * ping-pong migrate between CPU and GPU while the sentinel spin-polls
     * them; the UVM fault storm holds driver locks and starves every
     * other CUDA client in the process. Pinned memory has no faults and
     * no migration: CPU access is plain memory, GPU polls over PCIe. */
    size_t ring_bytes = sizeof(ScRingHeader) + sizeof(ScCommand) * g_sc.capacity;
    SC_CUDA_CHECK(cudaHostAlloc(&g_sc.control_host, ring_bytes,
                                cudaHostAllocMapped | cudaHostAllocPortable));
    memset(g_sc.control_host, 0, ring_bytes);
    SC_CUDA_CHECK(cudaHostGetDevicePointer(&g_sc.control_dev, g_sc.control_host, 0));

    g_sc.hdr = (ScRingHeader*)g_sc.control_host;
    g_sc.slots = (ScCommand*)((uint8_t*)g_sc.control_host + sizeof(ScRingHeader));

    g_sc.hdr->magic = SC_RING_MAGIC;
    g_sc.hdr->write_head = 0;
    g_sc.hdr->read_tail = 0;
    g_sc.hdr->capacity = g_sc.capacity;
    g_sc.hdr->version = SC_PROTOCOL_VERSION;

    int* sd = nullptr;
    SC_CUDA_CHECK(cudaHostAlloc((void**)&sd, sizeof(int),
                                cudaHostAllocMapped | cudaHostAllocPortable));
    *sd = 0;
    g_sc.shutdown_host = (volatile int*)sd;
    SC_CUDA_CHECK(cudaHostGetDevicePointer((void**)&g_sc.shutdown_dev, sd, 0));

    /* Non-blocking stream: keeps the persistent kernel out of legacy-
     * stream synchronization. (This does NOT exempt it from device-wide
     * syncs — see README, "The one rule".) */
    SC_CUDA_CHECK(cudaStreamCreateWithFlags(&g_sc.stream, cudaStreamNonBlocking));

    g_sc.initialized = 1;
    return SC_OK;
}

extern "C" int sc_launch(void* user_data) {
    if (!g_sc.initialized) {
        snprintf(g_err, sizeof(g_err), "not initialized");
        return SC_ERR_NOT_INITIALIZED;
    }
    if (g_sc.running) {
        snprintf(g_err, sizeof(g_err), "sentinel already running");
        return SC_ERR_ALREADY_RUNNING;
    }

    *g_sc.shutdown_host = 0;
    SC_CUDA_CHECK(cudaDeviceSynchronize());

    ScRingHeader* hdr_dev = (ScRingHeader*)g_sc.control_dev;
    ScCommand* slots_dev =
        (ScCommand*)((uint8_t*)g_sc.control_dev + sizeof(ScRingHeader));

    /* 1 block × 256 threads: minimal SM footprint. Thread 0 dispatches,
     * the rest are workers for sc_user_dispatch(). */
    sc_sentinel_kernel<<<1, 256, 0, g_sc.stream>>>(
        hdr_dev, slots_dev, g_sc.shutdown_dev, user_data);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        snprintf(g_err, sizeof(g_err), "kernel launch failed: %s",
                 cudaGetErrorString(err));
        return SC_ERR_CUDA_FAILURE;
    }

    g_sc.running = 1;
    return SC_OK;
}

extern "C" int sc_submit(uint8_t opcode, uint8_t flags,
                         uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                         uint32_t fence_id) {
    if (!g_sc.initialized) return SC_ERR_NOT_INITIALIZED;

    ScRingHeader* hdr = g_sc.hdr;
    if ((hdr->write_head + 1) % hdr->capacity == hdr->read_tail) {
        snprintf(g_err, sizeof(g_err), "ring buffer full");
        return SC_ERR_RING_FULL;
    }

    ScCommand* cmd = &g_sc.slots[hdr->write_head % hdr->capacity];
    cmd->opcode = opcode;
    cmd->flags = flags;
    cmd->sequence_id = g_sc.seq++;
    cmd->payload_size = 0;
    cmd->arg0 = a0;
    cmd->arg1 = a1;
    cmd->arg2 = a2;
    cmd->arg3 = a3;
    cmd->fence_id = fence_id;
    cmd->reserved = 0;
    cmd->timestamp = sc_host_clock_ns();
    cmd->checksum = sc_checksum(cmd);

    /* Command bytes must be globally visible before the head advances. */
    __sync_synchronize();
    hdr->write_head = (hdr->write_head + 1) % hdr->capacity;
    __sync_synchronize();

    return cmd->sequence_id;
}

extern "C" int sc_wait_fence(uint32_t fence_id, uint64_t expected,
                             uint64_t timeout_ns) {
    if (!g_sc.initialized) return SC_ERR_NOT_INITIALIZED;
    if (fence_id >= SC_FENCE_SLOTS) return SC_ERR_INVALID_FENCE;

    uint64_t start = sc_host_clock_ns();
    while (g_sc.hdr->fence_values[fence_id] < expected) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        if (timeout_ns > 0 && (sc_host_clock_ns() - start) > timeout_ns) {
            snprintf(g_err, sizeof(g_err),
                     "fence %u timeout (current=%llu, expected=%llu)",
                     fence_id,
                     (unsigned long long)g_sc.hdr->fence_values[fence_id],
                     (unsigned long long)expected);
            return SC_ERR_TIMEOUT;
        }
        if (*g_sc.shutdown_host == 2 && !g_sc.running) return SC_ERR_KERNEL_DEAD;
    }
    return SC_OK;
}

extern "C" uint64_t sc_fence_value(uint32_t fence_id) {
    if (!g_sc.initialized || fence_id >= SC_FENCE_SLOTS) return 0;
    return g_sc.hdr->fence_values[fence_id];
}

extern "C" void sc_get_stats(ScStats* out) {
    if (!g_sc.initialized || !out) return;
    out->commands_processed = g_sc.hdr->commands_processed;
    out->commands_errors = g_sc.hdr->commands_errors;
    out->last_dispatch_ns = g_sc.hdr->last_dispatch_ns;
    out->kernel_uptime_ns = g_sc.hdr->kernel_uptime_ns;
}

extern "C" int sc_is_running(void) {
    return g_sc.initialized && g_sc.running && *g_sc.shutdown_host != 2;
}

extern "C" uint32_t sc_pending(void) {
    if (!g_sc.initialized) return 0;
    ScRingHeader* hdr = g_sc.hdr;
    return (hdr->write_head - hdr->read_tail + hdr->capacity) % hdr->capacity;
}

extern "C" int sc_shutdown(void) {
    if (!g_sc.initialized) return SC_ERR_NOT_INITIALIZED;

    if (g_sc.running) {
        sc_submit(SC_OP_SHUTDOWN, 0, 0, 0, 0, 0, 0);

        uint64_t start = sc_host_clock_ns();
        while (*g_sc.shutdown_host != 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (sc_host_clock_ns() - start > 5000000000ULL) {
                fprintf(stderr, "[sentinel-comm] shutdown timeout, forcing\n");
                break;
            }
        }
        cudaStreamSynchronize(g_sc.stream);
        g_sc.running = 0;
    }

    if (g_sc.control_host)  cudaFreeHost(g_sc.control_host);
    if (g_sc.shutdown_host) cudaFreeHost((void*)g_sc.shutdown_host);
    if (g_sc.stream)        cudaStreamDestroy(g_sc.stream);
    memset(&g_sc, 0, sizeof(g_sc));
    return SC_OK;
}

extern "C" const char* sc_error(void) {
    return g_err;
}
