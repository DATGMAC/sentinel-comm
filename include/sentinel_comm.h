/**
 * ============================================================================
 *  SENTINEL COMM — Persistent-Kernel CPU→GPU Command Bus
 * ============================================================================
 *
 *  A drop-in communication layer between a CPU host and a resident GPU
 *  kernel. One persistent kernel is launched once and spin-polls a
 *  lock-free ring buffer in pinned host memory; the host submits 64-byte
 *  command packets and the GPU dispatches them WITHOUT any kernel-launch
 *  overhead. Measured dispatch latency: ~100–600 ns.
 *
 *  Extracted from PROJECT DATGMAC's Sentinel architecture and stripped of
 *  everything inference-specific. Zero dependencies beyond the CUDA runtime.
 *
 *  Host API (C linkage — usable from C, C++, or FFI):
 *      sc_init()        allocate ring buffer + control state
 *      sc_launch()      start the persistent kernel
 *      sc_submit()      enqueue a command (returns immediately)
 *      sc_wait_fence()  block until the GPU signals completion
 *      sc_get_stats()   dispatch latency / throughput counters
 *      sc_shutdown()    graceful kernel exit + cleanup
 *
 *  Custom GPU operations: implement sc_user_dispatch() in your own .cu
 *  file (see sentinel_comm_device.cuh and src/user_handlers.cu).
 *
 * ============================================================================
 */

#ifndef SENTINEL_COMM_H
#define SENTINEL_COMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 *  OP-CODES
 *
 *  0x00–0x0F are reserved for the bus itself. Everything from
 *  SC_OP_USER_BASE up is routed to sc_user_dispatch() — your code.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SC_OP_NOP            0x00  /* Heartbeat / latency probe               */
#define SC_OP_SHUTDOWN       0x01  /* Graceful kernel exit                    */
#define SC_OP_USER_BASE      0x10  /* First user-defined opcode               */

/* ═══════════════════════════════════════════════════════════════════════════
 *  COMMAND FLAGS
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SC_FLAG_NONE         0x00
#define SC_FLAG_FENCE_SIGNAL 0x01  /* Bump fence_values[fence_id] when done   */

/* ═══════════════════════════════════════════════════════════════════════════
 *  COMMAND PACKET — exactly 64 bytes (one cache line)
 *
 *  arg0..arg3 are opaque to the bus: pass device pointers, sizes, packed
 *  floats — whatever your handlers expect.
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef __CUDACC__
typedef struct __align__(64) ScCommand {
#else
typedef struct __attribute__((aligned(64))) ScCommand {
#endif
    uint8_t   opcode;
    uint8_t   flags;
    uint16_t  sequence_id;    /* Monotonic, assigned by sc_submit()           */
    uint32_t  payload_size;   /* Reserved for side-channel payloads           */
    uint64_t  arg0;
    uint64_t  arg1;
    uint64_t  arg2;
    uint64_t  arg3;
    uint32_t  fence_id;       /* Fence slot to signal (with SC_FLAG_FENCE_SIGNAL) */
    uint32_t  reserved;
    uint64_t  timestamp;      /* Host CLOCK_MONOTONIC ns at submission        */
    uint64_t  checksum;       /* CRC32 of the packet (integrity, not verified
                                 on the hot path)                             */
} ScCommand;

#ifdef __cplusplus
static_assert(sizeof(ScCommand) == 64, "ScCommand must be exactly 64 bytes");
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 *  RING BUFFER HEADER
 *
 *  Lives in pinned, mapped host memory (NOT managed memory — see README
 *  "Why pinned memory"). Host advances write_head; the GPU dispatcher
 *  advances read_tail. Lock-free SPSC.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SC_RING_CAPACITY_DEFAULT 4096
#define SC_FENCE_SLOTS           256
#define SC_RING_MAGIC            0x53454E54434F4D00ULL  /* "SENTCOM\0" */

#ifdef __CUDACC__
typedef struct __align__(128) ScRingHeader {
#else
typedef struct __attribute__((aligned(128))) ScRingHeader {
#endif
    uint64_t  magic;
    volatile uint32_t write_head;      /* Host → GPU (producer)               */
    volatile uint32_t read_tail;       /* GPU → Host (consumer)               */
    uint32_t  capacity;
    uint32_t  version;

    uint8_t   _pad0[32];               /* Keep fences off the head/tail line  */

    volatile uint64_t fence_values[SC_FENCE_SLOTS];  /* GPU writes, host polls */

    /* Profiling counters (GPU-written) */
    volatile uint64_t commands_processed;
    volatile uint64_t commands_errors;
    volatile uint64_t last_dispatch_ns;
    volatile uint64_t kernel_uptime_ns;
} ScRingHeader;

/* ═══════════════════════════════════════════════════════════════════════════
 *  STATUS CODES
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SC_OK                     0
#define SC_ERR_NOT_INITIALIZED   -1
#define SC_ERR_ALREADY_RUNNING   -2
#define SC_ERR_RING_FULL         -3
#define SC_ERR_CUDA_FAILURE      -5
#define SC_ERR_TIMEOUT           -6
#define SC_ERR_INVALID_FENCE     -7
#define SC_ERR_KERNEL_DEAD      -10

#define SC_PROTOCOL_VERSION       1

/* ═══════════════════════════════════════════════════════════════════════════
 *  HOST API
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct ScStats {
    uint64_t commands_processed;
    uint64_t commands_errors;
    uint64_t last_dispatch_ns;   /* GPU-side decode→complete time of the last
                                    command (globaltimer ticks)               */
    uint64_t kernel_uptime_ns;
} ScStats;

/**
 * Initialize the bus: allocates the ring buffer (pinned mapped host
 * memory) and control state on `device_id`.
 *
 * @param device_id      CUDA device ordinal.
 * @param ring_capacity  Command slots (0 = SC_RING_CAPACITY_DEFAULT).
 * @param pin_cpu_core   Pin + RT-prioritize the calling thread to this
 *                       core for jitter-free submission latency, or -1 to
 *                       leave scheduling alone (recommended default for
 *                       library use).
 */
int sc_init(int device_id, uint32_t ring_capacity, int pin_cpu_core);

/**
 * Launch the persistent kernel (1 block × 256 threads on a non-blocking
 * stream). `user_data` is an opaque pointer (typically a device pointer
 * to your own context struct) forwarded to every sc_user_dispatch() call.
 */
int sc_launch(void* user_data);

/** Enqueue a command. Returns the sequence id (>= 0) or a negative error.
 *  Never blocks; returns SC_ERR_RING_FULL if the GPU has fallen behind.  */
int sc_submit(uint8_t opcode, uint8_t flags,
              uint64_t arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3,
              uint32_t fence_id);

/** Block until fence_values[fence_id] >= expected (1µs poll granularity).
 *  timeout_ns = 0 waits forever. */
int sc_wait_fence(uint32_t fence_id, uint64_t expected, uint64_t timeout_ns);

/** Current value of a fence register (0 if uninitialized/invalid). */
uint64_t sc_fence_value(uint32_t fence_id);

/** Snapshot the GPU-side counters. */
void sc_get_stats(ScStats* out);

/** 1 while the persistent kernel is resident. */
int sc_is_running(void);

/** Commands submitted but not yet dispatched. */
uint32_t sc_pending(void);

/** Send SC_OP_SHUTDOWN, join the kernel, free all resources. */
int sc_shutdown(void);

/** Human-readable message for the last error. */
const char* sc_error(void);

#ifdef __cplusplus
}
#endif

#endif /* SENTINEL_COMM_H */
