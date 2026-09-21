"""Regression tests for Nemotron 3.5 image inputs and refit cache invalidation."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.io_struct import FlushCacheReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.models.nano_nemotron_vl import (
    NemotronH_Nano_VL_V2,
    NemotronH_Omni_Reasoning_V3,
)
from sglang.srt.multimodal.nemotron_vl import image_token_counts, image_token_spans
from sglang.srt.multimodal.processors.nemotron_35_vl import Nemotron35VLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNemotron35Refit(CustomTestCase):
    def test_ragged_images_reach_radio_as_single_image_batches(self):
        images = [torch.zeros(3, 8, 8), torch.zeros(1, 3, 8, 12)]
        normalized, counts = image_token_counts(
            images, patch_size=2, downsample_ratio=0.5
        )
        self.assertEqual(counts, [4, 6])
        self.assertEqual([x.ndim for x in normalized], [3, 3])
        model = object.__new__(NemotronH_Omni_Reasoning_V3)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(patch_size=2)
        model.downsample_ratio = 0.5
        model.model_dtype = torch.bfloat16
        with patch.object(
            NemotronH_Nano_VL_V2,
            "extract_feature_dynamic",
            return_value=torch.zeros(10, 3),
        ) as extract:
            model.extract_feature_dynamic(images)
        passed = extract.call_args.args[0]
        self.assertEqual(
            [tuple(x.shape) for x in passed], [(1, 3, 8, 8), (1, 3, 8, 12)]
        )
        self.assertTrue(all(x.dtype == torch.bfloat16 for x in passed))
        with self.assertRaises(ValueError):
            image_token_counts(torch.zeros(3, 7, 8), patch_size=2, downsample_ratio=0.5)

    def make_processor(self):
        processor = object.__new__(Nemotron35VLImageProcessor)
        processor._hf_image_lock = threading.Lock()
        processor.IMG_CONTEXT_TOKEN = "<image>"
        processor.mm_tokens = SimpleNamespace(image_token_id=18)
        processor.img_start_token_id, processor.img_end_token_id = 17, 19
        processor.patch_size, processor.downsample_ratio = 2, 0.5
        processor.max_model_len = 64
        processor.tokenizer = MagicMock(return_value={"input_ids": [18, 7, 18, 8]})
        processor.tokenizer.decode.return_value = "<image> between <image> answer"
        processor.load_mm_data = AsyncMock(
            return_value=SimpleNamespace(
                input_text="<image> between <image> answer", images=[object(), object()]
            )
        )
        processor._processor = MagicMock(
            image_processor=object(),
            return_value={
                "input_ids": torch.tensor(
                    [[17] + [18] * 4 + [19, 7, 17] + [18] * 6 + [19, 8]]
                ),
                "pixel_values": [torch.zeros(3, 8, 8), torch.zeros(3, 8, 12)],
            },
        )
        return processor

    def test_processor_preserves_noncanonical_caller_ids(self):
        processor = self.make_processor()
        # Simulate a BPE split or incomplete UTF-8 suffix that decode/encode changes.
        original = [18, 7, 18, 101, 102]
        out = asyncio.run(
            processor.process_mm_data_async(
                [object(), object()], None, original, SimpleNamespace(video_data=None)
            )
        )
        self.assertEqual(
            out.input_ids, [17] + [18] * 4 + [19, 7, 17] + [18] * 6 + [19, 101, 102]
        )
        self.assertEqual([item.offsets for item in out.mm_items], [[(1, 4)], [(8, 13)]])
        processor._processor.assert_called_once()
        # The resolved server limit takes precedence over the processor fallback.
        processor.max_model_len = 1
        asyncio.run(
            processor.process_mm_data_async(
                [object(), object()],
                None,
                original,
                SimpleNamespace(video_data=None),
                max_req_input_len=64,
            )
        )
        with self.assertRaisesRegex(ValueError, "context length"):
            asyncio.run(
                processor.process_mm_data_async(
                    [object(), object()],
                    None,
                    original,
                    SimpleNamespace(video_data=None),
                )
            )

    def test_processor_rejects_moved_images_and_wrong_feature_counts(self):
        ids = [17] + [18] * 4 + [19, 7, 17] + [18] * 6 + [19, 8]
        for counts, original in (([10], [18, 7, 18, 8]), ([4, 6], [18, 18, 7, 8])):
            with (
                self.subTest(counts=counts, original=original),
                self.assertRaises(ValueError),
            ):
                image_token_spans(
                    ids,
                    image_id=18,
                    start_id=17,
                    end_id=19,
                    token_counts=counts,
                    unexpanded_ids=original,
                )
        processor = self.make_processor()
        processor._processor.return_value["input_ids"][0, -1] = 99
        with self.assertRaisesRegex(ValueError, "changed surrounding prompt tokens"):
            asyncio.run(
                processor.process_mm_data_async(
                    [object(), object()],
                    None,
                    [18, 7, 18, 8],
                    SimpleNamespace(video_data=None),
                )
            )

    def test_flush_clears_embeddings_only_when_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = MagicMock()
        scheduler.req_to_token_pool = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.grammar_manager = MagicMock()
        scheduler.metrics_reporter = MagicMock(is_stats_logging_rank=False)
        scheduler.draft_worker = None
        scheduler.waiting_queue = []
        scheduler.running_batch = SimpleNamespace(reqs=[])
        embeddings = [torch.ones(1)]
        with patch.object(mm_schedule, "embedding_cache", embeddings):
            scheduler.is_fully_idle = lambda: False
            self.assertFalse(scheduler.flush_cache(empty_cache=False))
            self.assertEqual(len(embeddings), 1)
            scheduler.is_fully_idle = lambda: True
            self.assertTrue(scheduler.flush_cache(empty_cache=False))
            self.assertEqual(embeddings, [])

    def test_flush_requires_success_from_every_worker(self):
        for replies in ([True, False], [False, True], [True, True]):
            with self.subTest(replies=replies):
                manager = SimpleNamespace(
                    auto_create_handle_loop=lambda: None,
                    mm_processor=SimpleNamespace(clear_preprocess_cache=MagicMock()),
                    flush_cache_communicator=AsyncMock(
                        return_value=[
                            FlushCacheReqOutput(success=x, message="" if x else "busy")
                            for x in replies
                        ]
                    ),
                )
                result = asyncio.run(TokenizerControlMixin.flush_cache(manager))
                self.assertEqual(result.success, all(replies))
                self.assertEqual(
                    manager.mm_processor.clear_preprocess_cache.call_count,
                    int(all(replies)),
                )


if __name__ == "__main__":
    unittest.main()
