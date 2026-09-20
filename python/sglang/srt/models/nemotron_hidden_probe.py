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

Be aware that a *batched* prefill is not a 380-token one: 24 requests at 8192
tokens is 268 MB a file, and `NEMOTRON_HIDDEN_PROBE_FORWARDS=8` across four
engines is a couple of GB. Worth it once; not worth leaving on.

Files are `tp<rank>_forward<NNN>_<host>-<pid>.npz`. The writer suffix is not
decoration -- see `_writer_id`.

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
  NEMOTRON_HIDDEN_PROBE_MIN_TOKENS
                                  skip forwards narrower than this (default 32).
                                  The engine's first EXTEND is a one-token
                                  warmup; without this the probe captures that
                                  and reports two runs agreeing on <unk>.
  NEMOTRON_HIDDEN_PROBE_MODES     which ForwardMode values to capture, by name
                                  (default "EXTEND,MIXED" -- the prefill. "ALL"
                                  for decode steps too).
  NEMOTRON_HIDDEN_PROBE_RANKS     comma-separated tp ranks (default "0", "all"
                                  for every rank -- hidden states are replicated
                                  across tp after the reduce, so rank 0 is
                                  representative and the rest are a consistency
                                  check).

-----------------------------------------------------------------------------
THE SECOND SEAM: INSIDE THE VISION HALF

The seam above answered its question and left a finer one. Two runs' INPUT
hidden states are bitwise equal on every text position and cosine 0.19-0.25 on
the image block, so the difference is made somewhere between the pixels and the
projector's output -- and that stretch has three stages, of which only the last
one has ever been looked at::

    pixel_values      [n_images, 3, H, W]     <- the processor's output
        -> RadioModel (patch_generator + 32 encoder blocks)
    ==> tower output  [.., patches, 1280]     <- NEVER MEASURED on either side
        -> pixel_shuffle  (pure reshape, no parameters)
    ==> mlp1 input    [.., patches/4, 5120]
        -> mlp1  RMSNorm -> Linear -> ReLU^2 -> Linear
    ==> image features[.., patches/4, 4096]   <- this is the image block above

``NEMOTRON_HIDDEN_PROBE_VISION=1`` captures all four, and the three-way verdict
it produces is the point of it::

    pixels equal + tower equal + features differ  =>  the projector (mlp1)
    pixels equal + tower differs                  =>  the tower
    pixels differ                                 =>  the preprocessing

The tower's weights have been checked tensor by tensor and match; that is an
argument about four of 391 tensors, and this is a measurement of the function
they compute. `mlp1` has never been checked at all -- it is not loaded by
``RadioModel.load_weights`` but by ``adapter_dict`` + ``default_weight_loader``
one level up, so neither the drop warning nor the emit/arrive norm trace has
ever seen it, and unlike the tower it is not frozen.

**It hooks rather than wraps here**, because unlike ``language_model.model``
these are called as ``self.vision_model(chunk)`` and ``self.mlp1(feats)`` --
through ``Module.__call__``, which is exactly what dispatches forward hooks.
``get_image_feature`` is the exception and is wrapped: it is a bound method the
forward hands to ``general_mm_embed_routine`` by reference.

Aligning two runs here cannot key on token ids -- there are none at this depth.
It keys on **the pixels themselves**: a sha256 per image over the exact fp32
bytes handed to the tower. Which makes the alignment key and the first question
the same object, and that is deliberate -- if the digests do not match, that is
not a failure to align, it is the answer.

Environment:
  NEMOTRON_HIDDEN_PROBE_VISION    1 to capture the vision half (default off).
  NEMOTRON_HIDDEN_PROBE_VISION_CALLS
                                  get_image_feature calls to capture (default 2:
                                  the engine's first prefill is one request, the
                                  next is a full batch).
  NEMOTRON_HIDDEN_PROBE_VISION_MAX_ELEMS
                                  per-tensor element budget (default 4e6 = 16 MB
                                  at fp32). Bigger tensors are kept head-first
                                  and the full shape, fp64 norm and digest go to
                                  the metadata regardless, so a truncated tensor
                                  still answers "are these the same" -- it just
                                  cannot say where they differ past the cut.
  NEMOTRON_HIDDEN_PROBE_VISION_BLOCKS
                                  the tower's INTERNALS: the patch embedding and
                                  the per-block outputs of the 32-block ViT
                                  encoder. `all` (default), `none`, or a list of
                                  indices -- `0,15,31`, and `-1` is the last
                                  block. Negative and out-of-range entries are
                                  dropped rather than raising.

                                  `tower_out` says WHETHER the tower diverged;
                                  these say WHERE. The first block whose output
                                  moves is the first block that computes
                                  something else, and everything after it is a
                                  consequence -- so a run with these on turns
                                  "the tower is wrong" into one block index.

                                  Cost: one tensor per captured block per
                                  get_image_feature call, each capped by
                                  MAX_ELEMS. At one 480x576 image that is
                                  1066 x 1280 fp32 = 5.5 MB a block, so `all` is
                                  ~175 MB a call and ~1.4 GB for four engines at
                                  CALLS=2. Thin the list before raising
                                  ROLLOUT_BATCH_SIZE, not after.
  NEMOTRON_HIDDEN_PROBE_VISION_INNER
                                  one level further in: the stages INSIDE the
                                  listed blocks. `none` (default), `all`, or a
                                  list with the same spelling as _BLOCKS.

                                  Nine tensors a block: norm1 out, qkv out (per
                                  TP rank), the attention context (proj's
                                  input, per rank), proj out, the first
                                  residual (norm2's input), norm2 out, fc1 out
                                  (per rank), GELU out, fc2 out. Turn it on
                                  AFTER the block list has named a block --
                                  it answers "where inside block N", which is
                                  not a question until N is known.

                                  Two of the nine are per-rank slices rather
                                  than whole activations, which is the point:
                                  a fused qkv re-cut wrongly by head is visible
                                  in `qkv` and in nothing downstream of the
                                  all-reduce. Both sides must be read at the
                                  same rank, which RANKS already guarantees.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_ATTACHED: set[int] = set()
_WRITER_ID: str | None = None


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
    min_tokens = int(os.environ.get("NEMOTRON_HIDDEN_PROBE_MIN_TOKENS", "32") or 0)

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

        # The engine's first EXTEND is its own warmup: one token, id 0, before a
        # single request has arrived. Capturing it satisfies every check this
        # probe makes -- prefill, ids present, ids identical across runs -- and
        # says nothing, because one <unk> exercises no image, no sequence and
        # essentially no routing. Two runs agreeing on it is not a result, and
        # it read as one.
        embeds = pick(4, "input_embeds")
        n_tokens = int(embeds.shape[0]) if hasattr(embeds, "shape") else 0
        if n_tokens < min_tokens:
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


# ---------------------------------------------------------------------------
# The vision half: pixels -> RADIO tower -> pixel shuffle -> mlp1 -> features


_VISION_ATTACHED: set[int] = set()


def _vision_wanted() -> bool:
    if not os.environ.get("NEMOTRON_HIDDEN_PROBE_DIR", "").strip():
        return False
    raw = os.environ.get("NEMOTRON_HIDDEN_PROBE_VISION", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _vision_blocks(count: int) -> list[int]:
    """Which ViT block indices to hook, out of `count` of them.

    `all` (the default) or `none` or a list. An index is resolved the way
    Python resolves one -- `-1` is the last block -- and anything that does not
    land inside the encoder is dropped, because a probe that refuses a launch
    over a typo in a list of diagnostics is worse than one that captures 31 of
    the 32 blocks and says so in the log.
    """
    if count <= 0:
        return []
    raw = os.environ.get("NEMOTRON_HIDDEN_PROBE_VISION_BLOCKS", "all").strip().lower()
    if raw in {"", "none", "0b", "off", "false"}:
        return []
    if raw == "all":
        return list(range(count))
    return _parse_block_list(raw, count)


def _parse_block_list(raw: str, count: int) -> list[int]:
    """`0,15,-1` -> sorted, de-duplicated, resolved block indices."""
    chosen: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError:
            logger.warning("nemotron-probe: vision block '%s' is not an integer; ignored", part)
            continue
        if index < 0:
            index += count
        if 0 <= index < count:
            if index not in chosen:
                chosen.append(index)
        else:
            logger.warning(
                "nemotron-probe: vision block %s is outside the encoder's %d block(s); ignored",
                part, count,
            )
    return sorted(chosen)


def _vision_inner_blocks(count: int) -> list[int]:
    """Which blocks to open up stage by stage. Default none.

    Separate from NEMOTRON_HIDDEN_PROBE_VISION_BLOCKS because the two answer
    different questions and cost differently: the block list says WHICH block
    diverges and is cheap enough to leave on for all 32, this says WHERE INSIDE
    one block it happens and is only worth turning on once a block has been
    named. `all` is accepted and is 9 more tensors per block per call.
    """
    raw = os.environ.get("NEMOTRON_HIDDEN_PROBE_VISION_INNER", "").strip().lower()
    if count <= 0 or raw in {"", "none", "off", "false"}:
        return []
    return list(range(count)) if raw == "all" else _parse_block_list(raw, count)


def _capturing() -> bool:
    """True while a CUDA graph is being captured -- see `attach.capturing`."""
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


def _digest(array) -> str:
    """A content key for a tensor, over its exact bytes.

    This is the alignment key for the vision dumps, and it is deliberately the
    same object as the first question: two runs whose `pixel_values` digests
    match handed the tower the same tensor, and two runs whose digests do not
    match have already answered why their image features differ.
    """
    import hashlib

    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _norm64(array) -> float:
    """fp64, always. An fp32 norm over 21M elements reads 0.1% low -- it lost
    `pos_embed` to a false mismatch once already."""
    import numpy as np

    return float(np.linalg.norm(np.asarray(array, dtype=np.float64)))


def _head(array, budget: int):
    """The leading slice of `array` that fits in `budget` elements.

    Head-first rather than a stride, because the leading axis is images (or
    tokens, which are grouped by image), so a head keeps whole images and a
    stride keeps none. The metadata records the full shape and the full tensor's
    norm and digest either way, so a truncated array can still answer "are these
    the same" -- it just cannot say where past the cut.
    """
    import numpy as np

    if array.ndim == 0 or array.size <= budget:
        return array
    axis = 0 if array.shape[0] > 1 else (1 if array.ndim > 1 else 0)
    per_slice = max(array.size // max(array.shape[axis], 1), 1)
    keep = max(budget // per_slice, 1)
    index = [slice(None)] * array.ndim
    index[axis] = slice(0, keep)
    return np.ascontiguousarray(array[tuple(index)])


def _note(arrays: dict, meta: dict, key: str, value, budget: int) -> None:
    """Record one tensor: the head of it as an array, all of it as metadata."""
    array = _cpu(value)
    if array is None:
        meta[key] = {"missing": type(value).__name__}
        return
    info = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "norm": _norm64(array),
        "digest": _digest(array),
    }
    kept = _head(array, budget)
    if kept.shape != array.shape:
        info["kept_shape"] = list(kept.shape)
    arrays[key] = kept
    meta[key] = info


def _note_images(arrays: dict, meta: dict, key: str, value, budget: int) -> None:
    """Record `pixel_values`, per image.

    It arrives one of two ways -- a list of per-image tensors on the dynamic
    resolution path, or one `[n_images, 3, H, W]` stack on the static one -- and
    the comparison needs the images apart either way, because the two runs do
    not have to batch them the same.
    """
    items = None
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, torch.Tensor) and value.ndim == 4:
        items = [value[i] for i in range(value.shape[0])]
    if items is None:
        _note(arrays, meta, key, value, budget)
        return

    per_image, spent = [], 0
    for position, item in enumerate(items):
        array = _cpu(item)
        if array is None:
            per_image.append({"missing": type(item).__name__})
            continue
        entry = {
            "shape": list(array.shape),
            "norm": _norm64(array),
            "digest": _digest(array),
        }
        if spent + array.size <= budget:
            arrays[f"{key}_{position:03d}"] = array
            spent += array.size
        else:
            entry["dropped"] = True
        per_image.append(entry)
    meta[key] = {"images": len(items), "per_image": per_image}


def _tensor_of(output):
    """The tensor in a stage's return value, and whatever else it carried.

    `RadioModel.forward` returns a bare tensor on the static path and
    `(features, num_patches_list)` on the dynamic one -- and `num_patches_list`
    is not decoration, it is how the comparison finds image `i`'s rows inside a
    concatenated `[1, total_patches, 1280]`.
    """
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, (tuple, list)) and output:
        extra = [x for x in output[1:] if isinstance(x, (int, float, list, tuple))]
        return (output[0] if isinstance(output[0], torch.Tensor) else None), extra or None
    return None, None


def attach_vision(model) -> bool:
    """Capture the four tensors of the vision half, once per `get_image_feature`.

    Off unless NEMOTRON_HIDDEN_PROBE_VISION=1, detaches itself after
    NEMOTRON_HIDDEN_PROBE_VISION_CALLS, and never raises into the engine.
    """
    if not _vision_wanted():
        return False
    if id(model) in _VISION_ATTACHED:
        return True

    ranks = _enabled_ranks()
    rank = _tp_rank()
    if "all" not in ranks and str(rank) not in ranks:
        _VISION_ATTACHED.add(id(model))
        return False

    tower = getattr(model, "vision_model", None)
    projector = getattr(model, "mlp1", None)
    original = getattr(model, "get_image_feature", None)
    if tower is None or projector is None or original is None:
        logger.warning(
            "nemotron-probe: no vision_model / mlp1 / get_image_feature to hook;"
            " vision probe not attached"
        )
        _VISION_ATTACHED.add(id(model))
        return False

    root = Path(os.environ["NEMOTRON_HIDDEN_PROBE_DIR"].strip())
    limit = int(os.environ.get("NEMOTRON_HIDDEN_PROBE_VISION_CALLS", "2") or 2)
    budget = int(float(os.environ.get("NEMOTRON_HIDDEN_PROBE_VISION_MAX_ELEMS", "8e6") or 8e6))
    state: dict = {"calls": 0, "open": None, "handles": []}

    def stage(name: str, module, images: bool = False, inputs: bool = True):
        """A forward hook that files this stage's tensors into the open record.

        A hook and not a method wrap, unlike `attach`: these stages are reached
        through `Module.__call__` (`self.vision_model(chunk)`, `self.mlp1(x)`),
        which is the one path that dispatches hooks at all.
        """

        def hook(_module, hook_inputs, output):
            record = state["open"]
            if record is None or _capturing():
                return
            try:
                call = record["counts"].get(name, 0)
                record["counts"][name] = call + 1
                if inputs and hook_inputs:
                    note = _note_images if images else _note
                    note(record["arrays"], record["meta"],
                         f"{name}_in_{call:03d}", hook_inputs[0], budget)
                tensor, extra = _tensor_of(output)
                _note(record["arrays"], record["meta"], f"{name}_out_{call:03d}", tensor, budget)
                if extra is not None:
                    record["meta"][f"{name}_out_{call:03d}_extra"] = extra
            except Exception as exc:  # noqa: BLE001 -- never take the engine down for a log
                logger.warning("nemotron-probe: vision stage %s failed: %s: %s",
                               name, type(exc).__name__, exc)

        state["handles"].append(module.register_forward_hook(hook))

    stage("tower", tower, images=True)
    stage("mlp1", projector)
    # Inside the projector, outputs only -- each stage's input is the previous
    # stage's output, and storing both doubles the file to say the same thing.
    # mlp1[2] is ReLU^2, which has no parameters and cannot be loaded wrong.
    try:
        stage("mlp1_rmsnorm", projector[0], inputs=False)
        stage("mlp1_fc1", projector[1], inputs=False)
    except (TypeError, IndexError):
        logger.warning("nemotron-probe: mlp1 is not indexable; projector stages skipped")

    # Inside the tower. `tower_out` answers whether the tower diverged; these
    # answer where, which is the difference between "diff 391 weights" and
    # "read block N". Outputs only, for the reason the projector stages are:
    # block N's input is block N-1's output.
    #
    # The layout the comparison has to undo is the extract path's, not this
    # hook's. On the dynamic path `RadioModel._forward_dynamic` calls the patch
    # generator once per image and the encoder ONCE over every image's patches
    # concatenated, so `patch_embed`'s call index is the image index while each
    # block fires once with `[1, sum(len_i), 1280]`. `num_skip` below is what
    # lets the comparison cut that back apart: the encoder's rows for image i
    # are the cls/register prefix plus its patches, where `tower_out`'s
    # `num_patches_list` counts the patches alone.
    inner = getattr(tower, "model", None)
    patch_generator = getattr(inner, "patch_generator", None)
    encoder = getattr(inner, "encoder", None)
    layers = getattr(encoder, "layers", None)

    if patch_generator is not None:
        stage("patch_embed", patch_generator, inputs=False)
    else:
        logger.warning("nemotron-probe: no vision_model.model.patch_generator; patch embed skipped")

    inner = _vision_inner_blocks(len(layers) if layers is not None else 0)
    blocks = _vision_blocks(len(layers) if layers is not None else 0)
    if layers is None:
        logger.warning("nemotron-probe: no vision_model.model.encoder.layers; ViT blocks skipped")
    elif blocks:
        if getattr(encoder, "enable_cg", False):
            # Replay does not dispatch submodule hooks, so the blocks would
            # silently capture nothing while every other stage kept working.
            logger.warning(
                "nemotron-probe: SGLANG_VIT_ENABLE_CUDA_GRAPH is on -- per-block hooks"
                " only fire on the eager path, so block captures may be missing"
            )
        for index in blocks:
            stage(f"block{index:02d}", layers[index], inputs=False)

    # One level further in: the seams INSIDE a block, for the case where the
    # block-level capture has already named a block and the question becomes
    # which of its eight steps moved first. Every target below is reached
    # through `Module.__call__`, which is what makes a forward hook fire --
    # `qkv_backend.forward(...)` is called directly and cannot be hooked, so the
    # attention context is taken as `proj`'s INPUT instead, which is the same
    # tensor one rearrange later.
    #
    # Two of these are per-TP-rank and not the whole activation: `qkv` is
    # [.., 3 x heads/tp x 80] and `fc1` is [.., 5120/tp]. That is a feature --
    # a fused tensor re-cut wrongly by head shows up there and nowhere else --
    # but it means the two sides must be read at the same rank, which they are:
    # RANKS selects the same rank on both.
    if layers is not None and inner:
        for index in inner:
            layer = layers[index]
            tag = f"inner{index:02d}"
            attn = getattr(getattr(layer, "attn", None), "attn", None)
            mlp = getattr(layer, "mlp", None)
            try:
                # norm1's input is the block's input, which `block<NN>_out` of
                # the previous block already carries; its output is [a].
                stage(f"{tag}_norm1", layer.norm1, inputs=False)
                stage(f"{tag}_qkv", attn.qkv_proj, inputs=False)          # [b]
                stage(f"{tag}_proj", attn.proj, inputs=True)              # [c] in, [d] out
                stage(f"{tag}_norm2", layer.norm2, inputs=True)           # [e] in, [f] out
                stage(f"{tag}_fc1", mlp.fc1, inputs=False)                # pre-activation
                stage(f"{tag}_act", mlp.act, inputs=False)                # post-GELU
                stage(f"{tag}_fc2", mlp.fc2, inputs=False)                # [g]
            except AttributeError as exc:
                logger.warning(
                    "nemotron-probe: block %d has no %s; inner stages for it skipped",
                    index, exc,
                )
    state["blocks"] = blocks
    state["inner"] = inner
    state["num_skip"] = getattr(patch_generator, "num_skip", None)
    state["num_layers"] = len(layers) if layers is not None else None

    def wrapped(items):
        if state["calls"] >= limit or _capturing():
            return original(items)
        record = {
            "arrays": {},
            "meta": {
                "num_items": len(items) if hasattr(items, "__len__") else None,
                # Which of the two extract paths ran. They differ in more than
                # batching -- the dynamic one concatenates every image into one
                # `[1, total_patches, 1280]` and calls mlp1 per image, the static
                # one keeps images on dim 0 and calls mlp1 once per micro-batch.
                "is_dynamic": bool(
                    any(getattr(item, "is_dynamic", False) for item in items)
                ) if hasattr(items, "__iter__") else None,
                # How the block dumps cut back into images, and which blocks
                # are in this file at all -- a reader must not have to infer
                # either from the key names.
                "num_skip": state.get("num_skip"),
                "num_layers": state.get("num_layers"),
                "blocks": list(state.get("blocks") or ()),
                "inner": list(state.get("inner") or ()),
            },
            "counts": {},
        }
        state["open"] = record
        try:
            output = original(items)
        finally:
            state["open"] = None

        try:
            _note(record["arrays"], record["meta"], "image_features", output, budget)
            record["meta"]["stage_calls"] = record["counts"]
            index = state["calls"]
            state["calls"] += 1
            _write(root, rank, index, {"meta": record["meta"], **record["arrays"]}, kind="vision")
            if state["calls"] >= limit:
                detach_vision(model, state)
                logger.info(
                    "nemotron-probe: captured %d vision call(s) on tp rank %d; detached",
                    limit, rank,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("nemotron-probe: could not write vision call: %s: %s",
                           type(exc).__name__, exc)
        return output

    model.get_image_feature = wrapped
    _VISION_ATTACHED.add(id(model))
    logger.info(
        "nemotron-probe: vision probe on tp rank %d, %d call(s), %d elem budget,"
        " patch embed %s, %d/%s ViT block(s), inner stages for block(s) %s -> %s",
        rank, limit, budget,
        "on" if patch_generator is not None else "off",
        len(blocks), state.get("num_layers"),
        inner or "none", root,
    )
    return True


def detach_vision(model, state) -> None:
    for handle in state.get("handles", ()):
        try:
            handle.remove()
        except Exception:  # noqa: BLE001
            pass
    state["handles"] = []
    try:
        del model.get_image_feature
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# The weights the tower actually runs with


_WEIGHT_STATE: dict[int, dict] = {}
_LOAD_EPOCH: dict[int, int] = {}


def note_weight_load(model) -> None:
    """Record that `load_weights` ran, so the next forward re-fingerprints.

    This exists because the first version of `attach_weights` dumped once, at
    the first forward, and that is the WRONG MOMENT on the side that matters.
    The engine's first forward is its startup warmup and CUDA-graph capture --
    job 18934040 fingerprinted at 00:49:38, and slime's first
    `update_weights_from_tensor` had not run at all (`nemotron-vision-trace:
    emit` count 0). So the training run's dump was the startup HF weights, and
    comparing it against the rollout-only run would have shown them equal and
    proved nothing, in a way that reads exactly like a clean result.

    A sync calls `load_weights` once per bucket -- 461 of them -- so this
    increments 461 times and the next forward dumps once. That is the intended
    behaviour: the counter says "something changed since you last looked", not
    "how many times".
    """
    _LOAD_EPOCH[id(model)] = _LOAD_EPOCH.get(id(model), 0) + 1


def _weights_wanted() -> bool:
    if not os.environ.get("NEMOTRON_HIDDEN_PROBE_DIR", "").strip():
        return False
    raw = os.environ.get("NEMOTRON_HIDDEN_PROBE_WEIGHTS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def attach_weights(model) -> bool:
    """Fingerprint every live vision parameter, once per weight load.

    Dump 0 is the startup load; dump 1 on the training side is the state after
    `update_weights_from_tensor`. Comparing A's LAST dump against B's is the
    measurement -- see `note_weight_load` for the mistake that made this a
    per-load dump rather than a one-shot at the first forward.

    Same pixels in and a near-orthogonal tower output out (doc 08) leaves two
    shapes: some of the tower's 391 tensors arrive wrong, or they arrive intact
    and are filed under the wrong parameter. Both are statements about **the
    values the module holds when it runs**, and everything measured so far is a
    statement about something else -- what the checkpoint holds, what the
    exporter emits, what four traced tensors looked like on arrival.

    This reads the parameters themselves, off the live module, at the moment the
    forward is about to use them. Three properties make that the right place:

    * **it needs no reference.** The rollout-only run is the same engine at the
      same commit with no weight sync, so its dump *is* the reference. Comparing
      the two answers "do the weights differ" for all 391 at once;
    * **it is TP-exact.** Reading `named_parameters()` gets this rank's shard as
      the kernel will see it, so a tensor that arrived complete but was split or
      assigned wrongly reads as different here -- which a sum of shard norms
      cannot show, since that is invariant under any permutation;
    * **it is after everything.** Remap, weight_loader, TP split, dtype cast and
      any startup transform have all already happened.

    The fingerprint is a sha256 of the raw bytes plus an fp64 norm. The digest is
    what actually decides -- it is exact and permutation-sensitive - and the norm
    is there to say *how far* apart two tensors are once the digest says they
    are. fp64 because an fp32 norm over a 21M-element table reads 0.1% low and
    has already cost this investigation a false mismatch.

    Cheap enough to run on every rank: ~400 entries of a few numbers each, a few
    KB a file, no activation-sized copies. `NEMOTRON_HIDDEN_PROBE_RANKS=all` is
    the useful setting here, unlike for the hidden states.
    """
    if not _weights_wanted() or _capturing():
        # `_capturing()` is not belt and braces here: this copies every vision
        # parameter to the host, and a device-to-host copy inside a CUDA graph
        # capture is illegal. The engine captures graphs during the same startup
        # window this used to fire in.
        return False

    ranks = _enabled_ranks()
    rank = _tp_rank()
    if "all" not in ranks and str(rank) not in ranks:
        return False

    state = _WEIGHT_STATE.setdefault(id(model), {"dumps": 0, "epoch": None})
    limit = int(os.environ.get("NEMOTRON_HIDDEN_PROBE_WEIGHT_DUMPS", "4") or 4)
    epoch = _LOAD_EPOCH.get(id(model), 0)
    if state["dumps"] >= limit or epoch == state["epoch"]:
        return False

    root = Path(os.environ["NEMOTRON_HIDDEN_PROBE_DIR"].strip())
    try:
        record = _fingerprint_vision(model)
    except Exception as exc:  # noqa: BLE001 -- never take the engine down for a log
        logger.warning("nemotron-probe: could not fingerprint vision weights: %s: %s",
                       type(exc).__name__, exc)
        state["epoch"] = epoch
        return False
    record["meta"]["load_epoch"] = epoch
    index = state["dumps"]
    state["dumps"] += 1
    state["epoch"] = epoch
    _write(root, rank, index, record, kind="weights")
    return True


def _fingerprint_vision(model) -> dict:
    """One entry per vision parameter and buffer, from the live module."""
    import hashlib

    import numpy as np

    entries: dict[str, dict] = {}
    for prefix, module in (("vision_model", getattr(model, "vision_model", None)),
                           ("mlp1", getattr(model, "mlp1", None))):
        if module is None:
            continue
        named = list(module.named_parameters()) + list(module.named_buffers())
        for name, tensor in named:
            if not isinstance(tensor, torch.Tensor):
                continue
            # One parameter at a time, and let it go: the whole tower is ~1.3 GB
            # across the engine, and there is no reason to hold more than the
            # largest shard at once.
            array = tensor.detach().to(device="cpu")
            raw = np.ascontiguousarray(
                array.to(torch.float32).numpy() if array.is_floating_point() else array.numpy()
            )
            entries[f"{prefix}.{name}"] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "digest": hashlib.sha256(raw.tobytes()).hexdigest()[:16],
                "norm": float(np.linalg.norm(raw.astype(np.float64))) if raw.dtype.kind == "f" else None,
                "sum": float(np.asarray(raw, dtype=np.float64).sum()),
            }
            del array, raw

    meta = {"entries": entries, "count": len(entries)}
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        meta["tp_size"] = get_tensor_model_parallel_world_size()
    except Exception:  # noqa: BLE001
        pass
    logger.info("nemotron-probe: fingerprinted %d vision parameter(s)/buffer(s)", len(entries))
    return {"meta": meta}


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


def _writer_id() -> str:
    """What separates one engine's files from another's in a shared directory.

    The probe's first working capture (job 18919994) lost most of what it took,
    and the log said so once::

        nemotron-probe: could not write forward 0: FileNotFoundError:
          '…/hidden/tp0_forward000.tmp.npz' -> '…/hidden/tp0_forward000.npz'

    `NEMOTRON_HIDDEN_PROBE_RANKS=0` is tp rank 0 **of every engine**, and this
    recipe runs four of them into one `NEMOTRON_HIDDEN_PROBE_DIR`. All four
    computed the same `tp0_forward000.npz`, and therefore the same
    `tp0_forward000.tmp.npz`. Three ways that goes wrong, in increasing
    nastiness:

    * two `os.replace` calls race and the loser raises `FileNotFoundError` on a
      source another engine already renamed -- the loud case, and the only one
      that leaves a trace;
    * the file that survives is whichever engine won, and nothing in it says
      which;
    * engine X writes the tmp file, engine Y overwrites it, X renames it --
      so the *contents* can belong to an engine other than the one that put
      them there. Silent, and it corrupts a comparison rather than failing it.

    host + pid fixes all three: unique per writer, stable for the life of the
    engine, and legible in an `ls`. It is also recorded in `meta["writer"]`, so
    a file answers "which engine" without being parsed by name.

    Cached because `gethostname` is a syscall and this runs per forward.
    """
    global _WRITER_ID
    if _WRITER_ID is None:
        try:
            host = socket.gethostname().split(".")[0]
        except Exception:  # noqa: BLE001 -- a probe must never be the reason a server dies
            host = "unknown"
        _WRITER_ID = f"{host}-{os.getpid()}"
    return _WRITER_ID


def _write(root: Path, rank: int, index: int, record: dict, kind: str = "forward") -> None:
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
        elif key == "meta":
            meta.update(value or {})
        elif value is not None:
            arrays[key] = value
        else:
            meta[f"{key}_missing"] = True
    meta["tp_rank"] = rank
    meta["forward_index"] = index
    meta["kind"] = kind
    meta["writer"] = _writer_id()
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

    try:
        root.mkdir(parents=True, exist_ok=True)
        # `_writer_id()` and not just the tp rank -- see its docstring. Four
        # engines all have a tp rank 0 and all write here.
        path = root / f"tp{rank}_{kind}{index:03d}_{_writer_id()}.npz"
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
