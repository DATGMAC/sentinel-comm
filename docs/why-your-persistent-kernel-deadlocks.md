# Your persistent kernel will deadlock llama.cpp — and it's not the reason you think

*The empirical bisection behind sentinel-comm's "One Rule". All numbers
measured on an RTX 5060 Ti (Blackwell, CUDA 12.8), 2026-07-06.*

## The setup

We built a persistent CUDA kernel — launched once, spinning forever on a
command ring buffer in pinned host memory — as the dispatch layer of a
custom LLM inference engine. The heavy math ran through llama.cpp
(llama-cpp-python) in the same process, same CUDA context. The persistent
kernel used **one block of 256 threads** on a **non-blocking stream**, so it
couldn't starve the SMs and couldn't entangle the legacy stream. We even
"knew" about boot-order problems, so the engine warmed up *before* the
kernel launched.

Then the smoke test froze at 100% GPU utilization. Forever. No error, no
timeout — one process, pinned at 100%, generating zero tokens.

## The bisection

We reduced it to a minimal repro — load model, generate, launch persistent
kernel, generate again — and toggled one variable at a time under a hard
timeout:

| Test | Config | Result |
|------|--------|--------|
| A | small ctx, small buffers, **8-token generation before launch** | ✅ passes |
| B | big buffers, 8-token pre-generation | ✅ passes |
| C | big ctx + big buffers, 8-token pre-generation | ✅ passes |
| D | identical to C but only a **1-token warm-up** before launch | ❌ hangs |
| E | D + `GGML_CUDA_DISABLE_GRAPHS=1` | ❌ hangs |

Test E killed our first hypothesis: it's not CUDA graph capture (alone).
Test D vs C looked like "warm-up depth matters" — until the full smoke test
hung *even with* the deep warm-up. The final tell: every passing repro
reused the **same prompt** before and after launch, so llama.cpp reused its
KV cache and never processed a new batch shape. The real test generated
with a *different* prompt after launch — new shape — hang.

## The actual rule

A persistent kernel **never completes**. Several innocuous CUDA operations
perform an **implicit device-wide synchronization** — they wait for every
kernel on the device to finish, including one that never will:

- `cudaFree` / `cudaFreeHost` — documented to synchronize
- CUDA graph capture / instantiation
- **first-touch lazy allocations inside libraries** — cuBLAS creating a
  workspace the first time it sees a new GEMM shape, cuDNN building a plan,
  an allocator servicing a miss

That last one is the killer, because you don't call it — a library does,
whenever a *new shape* shows up. Warm-ups only mask it: the first
unseen-shape mid-session (a longer prompt, a bigger batch, a defrag)
deadlocks you at a random moment. Non-blocking streams don't help;
device-wide means device-wide.

## What actually works

1. **Phase separation** (what we shipped): the persistent kernel and the
   alloc-happy library never run concurrently. Park the kernel (graceful
   shutdown) before the risky phase, relaunch after — a relaunch costs one
   ordinary kernel launch, microseconds.
2. **Own all allocation**: if everything on the device is allocated before
   launch and freed after shutdown — the discipline sentinel-comm's API
   pushes you toward — a resident kernel is perfectly safe.
3. **Separate process** for the other library: its own CUDA context; its
   device-wide syncs only wait for its own context's work.

## Two bonus traps we hit on the way

- **`cudaMallocManaged` for the ring buffer**: managed pages ping-pong
  migrate while the GPU spin-polls and the CPU writes; the UVM fault storm
  holds driver locks and can starve every CUDA client in the process. Use
  pinned mapped host memory — no faults, no migration.
- **One-at-a-time dequeue over PCIe**: every poll of host memory from the
  GPU is a non-posted PCIe read. Dequeuing per-command cost ~11 µs/op;
  claiming up to 64 commands per poll with coalesced 16-byte fetches into
  shared memory dropped it to **0.62 µs/op** — faster than replaying a
  CUDA graph (2.1 µs/op measured on the same machine).

## The takeaway

Persistent kernels are a fantastic tool — ~96 ns GPU-side dispatch, 0.5 µs
CPU-side enqueue — but they change the rules of the CUDA context they live
in. The failure mode is not a crash but a silent, permanent 100%-GPU hang,
and the trigger can be a library's private allocation you'll never see in
your own code. Design for it up front: allocate before launch, free after
shutdown, and keep other libraries out of the context or out of the
resident window.

The implementation, benchmark, and smoke test are in this repo.
