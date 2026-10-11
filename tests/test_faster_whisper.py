from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from transcriber.backends.faster_whisper import (
    WINDOW_SECONDS,
    _batch_candidates,
    _batched_clips,
    _window_clips,
    _language_code,
    _load_model,
    _plan_batch,
    _retry_low_confidence_ranges,
    _run_inference,
    vad_coverage_notice,
)
from transcriber.config import TranscriptionConfig
from transcriber.errors import TranscriptionError
from transcriber.hardware import GpuMemory


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

    def test_retry_replaces_range_only_when_confidence_improves(self):
        original = SimpleNamespace(
            start=10.0,
            end=12.0,
            avg_logprob=-2.0,
            compression_ratio=1.0,
            no_speech_prob=0.1,
            temperature=1.0,
        )
        improved = SimpleNamespace(
            start=10.0,
            end=12.0,
            avg_logprob=-0.2,
            compression_ratio=1.0,
            no_speech_prob=0.1,
            temperature=0.0,
        )
        model = MagicMock()
        model.transcribe.return_value = (iter((improved,)), object())
        config = TranscriptionConfig(retry_low_confidence=True)

        segments, attempted, accepted = _retry_low_confidence_ranges(
            model, Path("song.wav"), [original], config
        )

        self.assertEqual(segments, [improved])
        self.assertEqual((attempted, accepted), (1, 1))
        self.assertEqual(
            model.transcribe.call_args.kwargs["clip_timestamps"], [10.0, 12.0]
        )


class ModelLoadingTests(TestCase):
    @patch("faster_whisper.WhisperModel")
    def test_allocation_failure_explains_the_occupancy(self, model_class):
        model_class.side_effect = RuntimeError("CUDA failed with error out of memory")

        with self.assertRaises(TranscriptionError) as raised:
            _load_model("turbo", "cuda", "float16", GpuMemory(free_mib=1967, total_mib=16311))

        message = str(raised.exception)
        self.assertIn("1967 MiB livres de 16311 MiB", message)
        self.assertIn("--compute-type int8_float16", message)

    @patch("faster_whisper.WhisperModel")
    def test_unrelated_failures_are_not_rewritten(self, model_class):
        model_class.side_effect = RuntimeError("modelo não encontrado")

        with self.assertRaisesRegex(RuntimeError, "modelo não encontrado"):
            _load_model("turbo", "cuda", "float16", None)

    @patch("faster_whisper.WhisperModel")
    def test_cpu_failures_are_not_rewritten(self, model_class):
        model_class.side_effect = RuntimeError("out of memory")

        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            _load_model("small", "cpu", "int8", None)


class BatchPlanningTests(TestCase):
    def test_profile_batch_is_kept_when_memory_is_plentiful(self):
        config = TranscriptionConfig(model="large-v3", batch_size=4)

        plan = _plan_batch(config, "float16", GpuMemory(free_mib=15000, total_mib=16311))

        self.assertEqual(plan.batch_size, 4)

    def test_explicit_batch_survives_a_crowded_card(self):
        config = TranscriptionConfig(model="large-v3", batch_size=4, batch_size_explicit=True)

        plan = _plan_batch(config, "float16", GpuMemory(free_mib=2000, total_mib=16311))

        self.assertEqual(plan.batch_size, 4)

    def test_cpu_runs_keep_the_profile_batch(self):
        config = TranscriptionConfig(model="small", batch_size=4)

        plan = _plan_batch(config, "int8", None)

        self.assertEqual(plan.batch_size, 4)


class ComputeTypeTests(TestCase):
    @patch("transcriber.backends.faster_whisper.supported_compute_types")
    @patch("faster_whisper.WhisperModel")
    def test_unsupported_precision_lists_the_alternatives(self, model_class, supported):
        model_class.side_effect = ValueError(
            "Requested int16 compute type, but the target device or backend do not "
            "support efficient int16 computation."
        )
        supported.return_value = ["bfloat16", "float16", "int8_float16"]

        with self.assertRaises(TranscriptionError) as raised:
            _load_model("tiny", "cuda", "int16", None)

        message = str(raised.exception)
        self.assertIn("'int16' não é suportada", message)
        self.assertIn("int8_float16", message)

    @patch("faster_whisper.WhisperModel")
    def test_unrelated_value_errors_are_not_rewritten(self, model_class):
        model_class.side_effect = ValueError("caminho de modelo inválido")

        with self.assertRaisesRegex(ValueError, "caminho de modelo inválido"):
            _load_model("tiny", "cuda", "float16", None)


class WindowClipsTests(TestCase):
    def test_windows_cover_the_whole_audio(self):
        clips = _window_clips(70.0)

        self.assertEqual(
            clips,
            [
                {"start": 0.0, "end": 30.0},
                {"start": 30.0, "end": 60.0},
                {"start": 60.0, "end": 70.0},
            ],
        )

    def test_no_window_exceeds_thirty_seconds(self):
        # A longer clip would be silently truncated by the batched pipeline.
        for duration in (5.0, 30.0, 31.0, 1200.0):
            for clip in _window_clips(duration):
                self.assertLessEqual(clip["end"] - clip["start"], WINDOW_SECONDS)

    def test_short_audio_is_a_single_window(self):
        self.assertEqual(_window_clips(12.0), [{"start": 0.0, "end": 12.0}])


class BatchedClipsTests(TestCase):
    def test_vad_runs_supply_their_own_boundaries(self):
        config = TranscriptionConfig(vad_filter=True, window_batching=True)

        self.assertIsNone(_batched_clips(Path("video.mp4"), config))

    def test_no_clips_without_the_opt_in(self):
        config = TranscriptionConfig(vad_filter=False, window_batching=False)

        self.assertIsNone(_batched_clips(Path("video.mp4"), config))

    @patch("transcriber.backends.faster_whisper.media_duration", return_value=65.0)
    def test_opt_in_without_vad_produces_windows(self, _duration):
        config = TranscriptionConfig(vad_filter=False, window_batching=True)

        clips = _batched_clips(Path("video.mp4"), config)

        self.assertEqual(len(clips), 3)


class VadlessInferenceTests(TestCase):
    @patch("faster_whisper.BatchedInferencePipeline")
    def test_batching_without_vad_falls_back_to_sequential(self, pipeline_class):
        # The batched pipeline derives chunks from VAD and refuses longer audio
        # without explicit boundaries, so the profile must still produce a result.
        model = MagicMock()
        model.transcribe.return_value = (iter(()), object())
        config = TranscriptionConfig(vad_filter=False, batch_size=8)

        segments, _info, batch_size = _run_inference(model, Path("video.mp4"), config)

        self.assertEqual(segments, [])
        self.assertEqual(batch_size, 1)
        pipeline_class.assert_not_called()
        model.transcribe.assert_called_once()

    @patch("transcriber.backends.faster_whisper.media_duration", return_value=90.0)
    @patch("faster_whisper.BatchedInferencePipeline")
    def test_opt_in_keeps_batching_and_passes_the_windows(self, pipeline_class, _duration):
        pipeline = MagicMock()
        pipeline.transcribe.return_value = (iter(()), object())
        pipeline_class.return_value = pipeline
        config = TranscriptionConfig(vad_filter=False, batch_size=8, window_batching=True)

        _segments, _info, batch_size = _run_inference(model := MagicMock(), Path("video.mp4"), config)

        self.assertEqual(batch_size, 8)
        model.transcribe.assert_not_called()
        clips = pipeline.transcribe.call_args.kwargs["clip_timestamps"]
        self.assertEqual(len(clips), 3)

    @patch("faster_whisper.BatchedInferencePipeline")
    def test_a_refusal_from_the_backend_is_recovered(self, pipeline_class):
        pipeline = MagicMock()
        pipeline.transcribe.side_effect = RuntimeError(
            "No clip timestamps found. Set 'vad_filter' to True or provide 'clip_timestamps'."
        )
        pipeline_class.return_value = pipeline
        model = MagicMock()
        model.transcribe.return_value = (iter(()), object())
        # VAD on, so the pre-emptive fallback does not apply and the backend refuses.
        config = TranscriptionConfig(vad_filter=True, batch_size=8)

        _segments, _info, batch_size = _run_inference(model, Path("video.mp4"), config)

        self.assertEqual(batch_size, 1)
        model.transcribe.assert_called_once()


class TimestampGranularityTests(TestCase):
    @patch("faster_whisper.BatchedInferencePipeline")
    def test_batched_runs_ask_for_timestamps(self, pipeline_class):
        # Without this, each chunk becomes a single subtitle block spanning the window.
        pipeline = MagicMock()
        pipeline.transcribe.return_value = (iter(()), object())
        pipeline_class.return_value = pipeline
        config = TranscriptionConfig(vad_filter=True, batch_size=8)

        _run_inference(MagicMock(), Path("video.mp4"), config)

        self.assertIs(pipeline.transcribe.call_args.kwargs["without_timestamps"], False)

    def test_sequential_runs_ask_for_timestamps(self):
        model = MagicMock()
        model.transcribe.return_value = (iter(()), object())
        config = TranscriptionConfig(vad_filter=True, batch_size=1)

        _run_inference(model, Path("video.mp4"), config)

        self.assertIs(model.transcribe.call_args.kwargs["without_timestamps"], False)


class VadCoverageTests(TestCase):
    def test_warns_when_most_of_the_audio_was_discarded(self):
        notice = vad_coverage_notice(280.0, 11.9)

        self.assertIsNotNone(notice)
        self.assertIn("12s de 280s", notice)
        self.assertIn("4%", notice)
        self.assertIn("--profile lyrics", notice)

    def test_quiet_when_coverage_is_healthy(self):
        self.assertIsNone(vad_coverage_notice(280.0, 200.0))

    def test_quiet_at_the_threshold(self):
        self.assertIsNone(vad_coverage_notice(100.0, 50.0))

    def test_warns_just_below_the_threshold(self):
        self.assertIsNotNone(vad_coverage_notice(100.0, 49.0))

    def test_unknown_values_are_not_reported(self):
        self.assertIsNone(vad_coverage_notice(None, 10.0))
        self.assertIsNone(vad_coverage_notice(280.0, None))
        self.assertIsNone(vad_coverage_notice(0.0, 0.0))
