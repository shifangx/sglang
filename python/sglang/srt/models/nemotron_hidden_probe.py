"""Dump what goes into the Nemotron language tower and what comes out of it.

Two runs of the same prompts under two weight sets picked different first tokens
in 429 of 512 samples. `--rollout-top-logprobs-num` says the output distribution
moved; it cannot say *where*. This splits the model at the one seam that
separates the two halves under suspicion:

    input ids + pixels
        -> embedding + vision tower + projector + placeholder injection
    ==> INPUT hidden states        [num_tokens, hidden]
        -> the 88-layer hybrid Mamba-2 / attention / latent-MoE tower
    ==> OUTPUT hidden states       [num_tokens, hidden]
        -> lm_head -> logits -> the sampled token

Input hidden states equal and output hidden states different puts the difference
in the tower -- which is where the one parameter known to survive the Megatron
round trip wrong lives, the MoE router's `expert_bias`. Input hidden states
already different puts it in the vision half or the embedding, and the tower is
exonerated. Nothing else this tree can run separates those two.

-----------------------------------------------------------------------------
THE SEAM

`NemotronHForCausalLM.forward` (``nemotron_h.py:1051``) is::

    hidden_states = self.model.forward(input_ids, positions, forward_batch,
                                       pp_proxy_tensors, input_embeds)
    return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

so ``language_model.model`` takes exactly the tensor this file calls INPUT and
returns exactly the one it calls OUTPUT. Intercepting that one method gets both,
and gets them without touching `general_mm_embed_routine`, which every other VLM
in the tree shares.

**It wraps the bound method rather than registering a forward hook**, and that is
not a style choice. The call above is ``self.model.forward(...)`` -- the method,
directly -- and torch dispatches forward hooks from ``Module.__call__`` only. A
``register_forward_hook`` on that module attaches successfully and never fires.
This probe's first run proves it: ``hooked language_model.model`` in the log,
three prefill batches, and not one file written.

It deliberately does not intercept ``language_model``: that returns the logits
processor's output, which is a different question and a much larger tensor.

-----------------------------------------------------------------------------
WHAT IT COSTS

hidden_size is 4096, so a 380-token prefill is 3.1 MB per tensor, 6.2 MB for the
pair. The default is the first forward only, on tp rank 0 only, which is one
file of a few MB per engine. A decode step is one token wide and costs nothing.

**It is off unless NEMOTRON_HIDDEN_PROBE_DIR is set**, and it detaches itself
after the configured number of forwards, so a run that forgets to unset it pays
for one batch and nothing more.

-----------------------------------------------------------------------------
ALIGNING TWO RUNS

A prefill forward covers a whole batch, and two runs do not have to assemble the
same batch. So the dump records ``input_ids`` and the sequence boundaries as
well as the tensors: the comparison keys on the ids, and finds the request
inside the batch rather than assuming it sits at the same offset.

Environment:
  NEMOTRON_HIDDEN_PROBE_DIR       where to write. Unset or empty = off.
  NEMOTRON_HIDDEN_PROBE_FORWARDS  forwards to capture per engine (default 1).
  NEMOTRON_HIDDEN_PROBE_MODES     which ForwardMode values to capture, by name
                                  (default "EXTEND,MIXED" -- the prefill. "ALL"
                                  for decode steps too).
  NEMOTRON_HIDDEN_PROBE_RANKS     comma-separated tp ranks (default "0", "all"
                                  for every rank -- hidden states are replicated
                                  across tp after the reduce, so rank 0 is
                                  representative and the rest are a consistency
                                  check).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_ATTACHED: set[int] = set()


def _enabled_ranks() -> set[str]:
    raw = os.environ.get("NEMOTRON_HIDDEN_PROBE_RANKS", "0").strip()
    return {"all"} if raw == "all" else {part.strip() for part in raw.split(",") if part.strip()}


def _tp_rank() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001 -- a probe must never be the reason a server dies
        return 0


def _seq_boundaries(forward_batch) -> dict:
    """Whatever this batch will admit about how its tokens split into requests."""
    out = {}
    for field in ("extend_seq_lens", "seq_lens", "extend_start_loc", "out_cache_loc"):
        value = getattr(forward_batch, field, None)
        if isinstance(value, torch.Tensor):
            out[field] = value.detach().to("cpu")
    mode = getattr(forward_batch, "forward_mode", None)
    out["forward_mode"] = str(mode)
    out["batch_size"] = getattr(forward_batch, "batch_size", None)
    return out


def attach(model) -> bool:
    """Attach once per model instance. Returns whether hooks are now live."""
    directory = os.environ.get("NEMOTRON_HIDDEN_PROBE_DIR", "").strip()
    if not directory:
        return False
    if id(model) in _ATTACHED:
        return True

    ranks = _enabled_ranks()
    rank = _tp_rank()
    if "all" not in ranks and str(rank) not in ranks:
        _ATTACHED.add(id(model))
        return False

    try:
        tower = model.language_model.model
    except AttributeError:
        logger.warning("nemotron-probe: no language_model.model to hook; probe not attached")
        _ATTACHED.add(id(model))
        return False

    limit = int(os.environ.get("NEMOTRON_HIDDEN_PROBE_FORWARDS", "1") or 1)
    root = Path(directory)
    state = {"calls": 0, "original": tower.forward}

    def capturing() -> bool:
        """True while a CUDA graph is being captured.

        A device-to-host copy is illegal inside capture, and SGLang captures
        graphs for decode. The probe's target is the first PREFILL, which this
        recipe runs eagerly (`cuda_graph_backend_prefill='disabled'`, and the
        scheduler's own line reads `cuda graph: False`), so in practice this
        never fires -- but a probe that can crash a capture is a probe nobody
        will leave enabled.
        """
        try:
            return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        except Exception:  # noqa: BLE001
            return False

    want_modes = {m.strip().upper() for m in os.environ.get(
        "NEMOTRON_HIDDEN_PROBE_MODES", "EXTEND,MIXED").split(",") if m.strip()}

    def wanted(forward_batch) -> bool:
        """Only the prefill, unless asked otherwise.

        The first run of this captured a DECODE batch -- `forward_mode 2`,
        `batch_size 195`, which is 195 requests of ONE token each, not one
        195-token sequence. That is not the forward the question is about: the
        first generated token comes out of the last row of the PREFILL, and a
        decode step's input hidden states are the embeddings of tokens the two
        runs had already diverged on.
        """
        if "ALL" in want_modes:
            return True
        mode = getattr(forward_batch, "forward_mode", None)
        name = getattr(mode, "name", None)
        if name is None:
            return True  # unknown shape: capture rather than silently skip
        return name.upper() in want_modes

    def wrapped(*args, **kwargs):
        # The caller is `self.model.forward(...)` (nemotron_h.py:1059), a direct
        # call on the METHOD. torch dispatches forward hooks from Module.__call__
        # only, so register_forward_hook here attaches successfully and never
        # fires -- which is exactly how this probe spent its first run: "hooked
        # language_model.model" in the log, three prefill batches, and not one
        # file. Wrapping the bound method is what actually intercepts that call
        # site, and works whether the caller uses __call__ or .forward().
        if capturing():
            return state["original"](*args, **kwargs)

        def pick(index, name):
            # Positional at the call site, but read defensively: a signature
            # change upstream should cost a missing field in a debug dump, not a
            # dead engine.
            if len(args) > index:
                return args[index]
            return kwargs.get(name)

        forward_batch = pick(2, "forward_batch")
        if not wanted(forward_batch):
            return state["original"](*args, **kwargs)

        # `input_ids` is None on the multimodal path: general_mm_embed_routine
        # calls the language model with input_embeds instead. The ids are still
        # on the batch, and without them the comparison has nothing to align on
        # but sequence length -- which is how the first run silently "matched"
        # two batches by their token count.
        ids = pick(0, "input_ids")
        if ids is None:
            ids = getattr(forward_batch, "input_ids", None)

        record = {
            "input_ids": _cpu(ids),
            "forward_batch": _seq_boundaries(forward_batch),
            "input_hidden_states": _cpu(pick(4, "input_embeds")),
        }
        output = state["original"](*args, **kwargs)
        record["output_hidden_states"] = _cpu(output if isinstance(output, torch.Tensor) else None)

        index = state["calls"]
        state["calls"] += 1
        _write(root, rank, index, record)
        if state["calls"] >= limit:
            detach(tower, state)
            logger.info("nemotron-probe: captured %d forward(s) on tp rank %d; detached", limit, rank)
        return output

    tower.forward = wrapped
    _ATTACHED.add(id(model))
    logger.info(
        "nemotron-probe: wrapped language_model.model.forward on tp rank %d, %d forward(s) -> %s",
        rank,
        limit,
        root,
    )
    return True


def detach(tower, state) -> None:
    """Put the real method back. Deleting the instance attribute is enough --
    it was shadowing the class's, which is untouched."""
    try:
        del tower.forward
    except AttributeError:
        # Already gone, or nn.Module refused; fall back to rebinding.
        try:
            tower.forward = state["original"]
        except Exception:  # noqa: BLE001
            pass


def _cpu(tensor):
    """Detached fp32 numpy copy on the host, or None.

    fp32 rather than the model's bf16 so the file is not itself a lossy record of
    what it is measuring -- a bf16 dump cannot show a difference smaller than a
    bf16 ulp, which is exactly the size of difference this is looking for.

    numpy rather than torch because of who reads it: the probe runs inside the
    engine, where torch is a given, but the comparison runs wherever the operator
    is -- and on this tree's login nodes there is no torch at all. An .npz needs
    numpy and nothing else.
    """
    if not isinstance(tensor, torch.Tensor):
        return None
    tensor = tensor.detach().to(device="cpu")
    # Token ids are integers and must stay exact -- fp32 is for the activations.
    return (tensor if not tensor.is_floating_point() else tensor.to(torch.float32)).numpy()


def _write(root: Path, rank: int, index: int, record: dict) -> None:
    import json

    import numpy as np

    arrays, meta = {}, {}
    for key, value in record.items():
        if key == "forward_batch":
            for field, sub in (value or {}).items():
                if isinstance(sub, torch.Tensor):
                    arrays[f"fb_{field}"] = sub.numpy()
                else:
                    meta[field] = sub
        elif value is not None:
            arrays[key] = value
        else:
            meta[f"{key}_missing"] = True
    meta["tp_rank"] = rank
    meta["forward_index"] = index
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"tp{rank}_forward{index:03d}.npz"
        # `.tmp.npz`, not `.npz.tmp`: np.savez appends `.npz` to any name that
        # does not already end in it, so the latter is written as
        # `...npz.tmp.npz` and os.replace below then fails on a missing source.
        tmp = path.with_name(path.name.replace(".npz", ".tmp.npz"))
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 -- never take the engine down for a log
        logger.warning("nemotron-probe: could not write forward %d: %s: %s", index, type(exc).__name__, exc)
        return
    shapes = {key: tuple(value.shape) for key, value in arrays.items() if key != "meta_json"}
    logger.info("nemotron-probe: wrote %s %s %s", path.name, shapes, meta)
