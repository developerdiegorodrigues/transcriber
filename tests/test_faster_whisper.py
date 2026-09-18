from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from transcriber.backends.faster_whisper import (
    _batch_candidates,
    _language_code,
    _run_inference,
)
from transcriber.config import TranscriptionConfig


class FasterWhisperHelpersTests(TestCase):
    def test_batch_candidates_end_in_one_without_duplicates(self):
        self.assertEqual(_batch_candidates(8), [8, 4, 2, 1])
        self.assertEqual(_batch_candidates(1), [1])

    def test_language_names_are_converted_to_codes(self):
        self.assertEqual(_language_code("English"), "en")
        self.assertEqual(_language_code("Portuguese"), "pt")
        self.assertEqual(_language_code("pt"), "pt")
        self.assertIsNone(_language_code(None))

    @patch("faster_whisper.BatchedInferencePipeline")
    def test_out_of_memory_reduces_batch(self, pipeline_class):
        pipeline = MagicMock()
        pipeline.transcribe.side_effect = [
            RuntimeError("CUDA out of memory"),
            (iter(()), object()),
        ]
        pipeline_class.return_value = pipeline
        config = TranscriptionConfig(batch_size=4)

        segments, _info, batch_size = _run_inference(
            MagicMock(), Path("video.mp4"), config
        )

        self.assertEqual(segments, [])
        self.assertEqual(batch_size, 2)
        self.assertEqual(pipeline.transcribe.call_count, 2)
