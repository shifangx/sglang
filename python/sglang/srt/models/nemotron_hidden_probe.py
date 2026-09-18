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
returns exactly the one it calls OUTPUT. Hooking that one module gets both, and
gets them without touching `general_mm_embed_routine`, which every other VLM in
the tree shares.

It is deliberately NOT hooked on ``language_model``: that returns the logits
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
    state = {"calls": 0, "pending": None, "handles": []}

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

    def pre_hook(_module, args, kwargs):
        if capturing():
            state["pending"] = None
            return None
        # nemotron_h.py calls this positionally:
        #   (input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds)
        # but read it defensively -- a signature change upstream should cost a
        # missing field in a debug dump, not a dead engine.
        def pick(index, name):
            if len(args) > index:
                return args[index]
            return kwargs.get(name)

        state["pending"] = {
            "input_ids": _cpu(pick(0, "input_ids")),
            "forward_batch": _seq_boundaries(pick(2, "forward_batch")),
            "input_hidden_states": _cpu(pick(4, "input_embeds")),
        }
        return None

    def post_hook(_module, _args, output):
        if capturing():
            return output
        record = state["pending"]
        state["pending"] = None
        if record is None:
            return output
        record["output_hidden_states"] = _cpu(output if isinstance(output, torch.Tensor) else None)
        index = state["calls"]
        state["calls"] += 1
        _write(root, rank, index, record)
        if state["calls"] >= limit:
            for handle in state["handles"]:
                handle.remove()
            state["handles"].clear()
            logger.info("nemotron-probe: captured %d forward(s) on tp rank %d; detached", limit, rank)
        return output

    state["handles"] = [
        tower.register_forward_pre_hook(pre_hook, with_kwargs=True),
        tower.register_forward_hook(post_hook),
    ]
    _ATTACHED.add(id(model))
    logger.info(
        "nemotron-probe: hooked language_model.model on tp rank %d, %d forward(s) -> %s",
        rank,
        limit,
        root,
    )
    return True


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
    return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()


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
        tmp = path.with_suffix(".npz.tmp")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 -- never take the engine down for a log
        logger.warning("nemotron-probe: could not write forward %d: %s: %s", index, type(exc).__name__, exc)
        return
    shapes = {key: tuple(value.shape) for key, value in arrays.items() if key != "meta_json"}
    logger.info("nemotron-probe: wrote %s %s %s", path.name, shapes, meta)
