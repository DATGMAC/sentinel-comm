# Launch notes — how people find sentinel-comm

Everything below is copy-paste ready. The rule behind all of it: lead with a
measured number, never an adjective. The CUDA crowd rewards receipts and
punishes hype.

## Tagline (README + everywhere)

> **Stop launching kernels. Start sending commands.**

One-sentence expansion:

> A persistent-kernel CPU→GPU command bus — 0.5 µs enqueue, 96 ns dispatch,
> no `cudaLaunchKernel`.

## GitHub repo settings

**Description** (the ~120 chars shown in search results and link previews):

> Persistent-kernel CPU→GPU command bus. 64-byte packets, 0.5 µs enqueue,
> ~96 ns dispatch — no cudaLaunchKernel. C++/CUDA, 2 files.

**Topics** (this is real SEO — people search GitHub by these):

```
cuda  gpu  persistent-threads  low-latency  ring-buffer  gpgpu
kernel-launch  cuda-kernels  hpc  cpp  command-queue  megakernel
```

Set with: `gh repo edit --description "..." --add-topic cuda --add-topic gpu ...`

## Show HN post

Title (HN strips marketing; a plain factual claim with a number survives):

> Show HN: A persistent-kernel CPU→GPU command bus (3–4× CUDA Graphs on tiny ops)

First comment (post it yourself immediately — it frames the thread):

> Author here. This started as the dispatch layer of a homebrew LLM engine.
> The interesting parts: (1) dispatching GPU work by writing 64 bytes into a
> pinned ring buffer instead of calling cudaLaunchKernel — 0.62 µs/op
> sustained vs 2.1 µs for CUDA graph replay on my 5060 Ti; (2) the README
> documents, with an empirical bisection, exactly how a persistent kernel
> silently deadlocks any library that lazily allocates (cuBLAS workspaces,
> graph capture, cudaFree) — that part cost me a day and I've never seen it
> written down. Honest caveats included: isolated round-trips are ~10 µs,
> slower than launch+sync — this is for streams of small ops, not ping-pong.

## The shareable piece (post separately, later)

`docs/why-your-persistent-kernel-deadlocks.md` is the thing people will
actually share. Post it as a blog article / HN link with the title:

> Your persistent kernel will deadlock llama.cpp — and it's not the reason
> you think

This is a different audience hook than the library itself: people click
war stories about debugging. It links back to the repo. Space the two posts
a week or two apart; whichever lands gives the other a second life.

## Reddit

- r/CUDA and r/gpgpu: post the benchmark table + repo link, same framing as
  the HN first comment. These small subs convert well — the people there
  have this exact problem.
- r/cpp (only the war-story article; the sub dislikes pure-CUDA projects
  but loves debugging stories).

## The three numbers to repeat everywhere

| | |
|---|---|
| 0.62 µs/op | sustained tiny-op throughput (vs 2.1 µs CUDA graph replay) |
| 0.5 µs | CPU cost to enqueue (one 64-byte write, no driver call) |
| 96 ns | GPU-side dispatch, decode→complete |

And the one honest disclaimer that buys credibility: *isolated round-trips
are slower than a plain launch+sync (~10 µs vs ~8 µs) — use streams for
ping-pong.*

## Before pushing publicly — 10-minute checklist

- [ ] `git remote add origin git@github.com:<you>/sentinel-comm.git && git push -u origin main`
- [ ] Set description + topics (above)
- [ ] Run `./bench_dispatch` on the target machine and paste YOUR numbers
      into the README table (numbers must match what a cloner will see)
- [ ] Enable Issues; add a "good first issue": multi-ring / multi-GPU support
- [ ] Pin the war-story doc in the README top ("Read: why your persistent
      kernel deadlocks →")
