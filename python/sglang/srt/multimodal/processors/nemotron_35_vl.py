"""Nemotron 3.5 Super VL image serving with the checkpoint's HF processor.

Use the same resize, normalization, template and expansion as the trainer.
The older Nano processor applies a server-context-dependent image budget and
cannot be substituted for the processor used to calculate training logprobs.
"""

import asyncio
from threading import Lock

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.nano_nemotron_vl import NemotronH_Omni_Reasoning_V3
from sglang.srt.multimodal.nemotron_vl import image_token_counts, image_token_spans
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
)


class Nemotron35VLImageProcessor(NanoNemotronVLImageProcessor):
    models = [NemotronH_Omni_Reasoning_V3]
    preserve_processor_input_ids = True
    supports_mm_processor_concurrency = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hf_image_lock = Lock()

    def _call_hf_processor(self, prompt, images):
        # The worker owns this lock until the CPU work finishes, even if its
        # awaiting request is cancelled. An asyncio lock would release early.
        with self._hf_image_lock:
            return self._processor(text=prompt, images=images, return_tensors="pt")

    async def process_mm_data_async(
        self, image_data, audio_data, input_text, request_obj, **kwargs
    ):
        # Retain the upstream audio/video serving path. Image rollout/refit does
        # not qualify those modalities or speculative decoding.
        if audio_data or request_obj.video_data:
            return await super().process_mm_data_async(
                image_data, audio_data, input_text, request_obj, **kwargs
            )
        original_ids = list(input_text) if isinstance(input_text, list) else None
        # Generated BPE tokens and incomplete UTF-8 prefixes need not round-trip
        # through decode/encode. Use text only for HF image preprocessing; retain
        # the caller's exact IDs for the language model below.
        processor_text = (
            self.tokenizer.decode(
                original_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if original_ids is not None
            else input_text
        )
        loaded = await self.load_mm_data(
            prompt=processor_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )
        prompt = loaded.input_text
        if prompt.count(self.IMG_CONTEXT_TOKEN) != len(loaded.images):
            raise ValueError("Nemotron expects one unexpanded <image> per input image")
        if original_ids is not None and (
            prompt != processor_text
            or original_ids.count(self.mm_tokens.image_token_id) != len(loaded.images)
        ):
            raise ValueError(
                "Nemotron image loading changed the caller's prompt or image count"
            )
        if not loaded.images:
            ids = (
                original_ids
                if original_ids is not None
                else self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            return MultimodalProcessorOutput(input_ids=ids, mm_items=[])
        if not hasattr(self._processor, "image_processor"):
            raise ValueError("Nemotron 3.5 VL requires the checkpoint's AutoProcessor")
        # Remote processors may mutate resize/token-budget state. Serialize
        # only the HF call; image loading and GPU serving remain concurrent.
        output = await asyncio.to_thread(self._call_hf_processor, prompt, loaded.images)
        ids = output["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if len(ids) != 1:
            raise ValueError("Expected one processed prompt per serving request")
        ids = list(ids[0])
        images, counts = image_token_counts(
            output["pixel_values"],
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
        )
        if len(images) != len(loaded.images):
            raise ValueError("Nemotron HF processor changed the number of input images")
        # HF may expand only image placeholders in its text input. Validate
        # this before replacing that text's canonical tokenization with the
        # exact original token sequence supplied by an input_ids caller.
        spans = image_token_spans(
            ids,
            image_id=self.mm_tokens.image_token_id,
            start_id=self.img_start_token_id,
            end_id=self.img_end_token_id,
            token_counts=counts,
            unexpanded_ids=self.tokenizer(prompt, add_special_tokens=False)[
                "input_ids"
            ],
        )
        if original_ids is not None:
            ids = []
            counts_iter = iter(counts)
            for token in original_ids:
                if token == self.mm_tokens.image_token_id:
                    ids.extend(
                        [self.img_start_token_id]
                        + [token] * next(counts_iter)
                        + [self.img_end_token_id]
                    )
                else:
                    ids.append(token)
            spans = image_token_spans(
                ids,
                image_id=self.mm_tokens.image_token_id,
                start_id=self.img_start_token_id,
                end_id=self.img_end_token_id,
                token_counts=counts,
                unexpanded_ids=original_ids,
            )
        # TokenizerManager knows the resolved serving limit even when the user
        # did not override --context-length (the Nano default is only 8192).
        max_input_len = kwargs.get("max_req_input_len") or self.max_model_len
        if len(ids) > max_input_len:
            raise ValueError(
                "Nemotron processed prompt exceeds the server context length"
            )
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=pixels,
                offsets=[span],
                model_specific_data={"num_tokens": count, "is_dynamic": True},
            )
            for pixels, span, count in zip(images, spans, counts, strict=True)
        ]
        return MultimodalProcessorOutput(
            input_ids=ids,
            mm_items=items,
            im_start_id=self.img_start_token_id,
            im_end_id=self.img_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
        )
