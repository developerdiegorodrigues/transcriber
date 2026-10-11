import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from transcriber.benchmark import _relative_output, run_benchmark, word_error_rate
from transcriber.hardware import GpuMemory
from transcriber.result import TranscriptionOutcome


class WordErrorRateTests(TestCase):
    def test_identical_text_has_zero_error(self):
        self.assertEqual(word_error_rate("Hello, world!", "hello world"), 0.0)

    def test_counts_substitution(self):
        self.assertEqual(word_error_rate("one two three", "one four three"), 1 / 3)

    def test_empty_reference(self):
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "unexpected"), 1.0)


class RelativeOutputTests(TestCase):
    def test_paths_are_relative_to_the_report(self):
        self.assertEqual(
            _relative_output(
                Path("/data/benchmarks/artifacts/video/fast/video"),
                Path("/data/benchmarks"),
            ),
            "artifacts/video/fast/video",
        )

    def test_unrelated_path_is_reduced_to_its_name(self):
        self.assertEqual(
            _relative_output(Path("/home/alguem/saida/video"), Path("/data/benchmarks")),
            "video",
        )


class ReportTests(TestCase):
    @patch("transcriber.benchmark.hardware_snapshot")
    @patch("transcriber.benchmark.concurrent_compute_usage")
    @patch("transcriber.benchmark.gpu_memory")
    @patch("transcriber.benchmark.transcribe_file")
    @patch("transcriber.benchmark.resolve_device", return_value="cuda")
    @patch("transcriber.benchmark.probe_cuda")
    @patch("transcriber.benchmark._prepare_model")
    @patch("transcriber.benchmark.media_duration", return_value=60.0)
    def test_report_describes_the_hardware_and_the_contention(
        self,
        _duration,
        _prepare,
        _probe,
        _resolve,
        transcribe,
        memory,
        contention,
        snapshot,
    ):
        memory.return_value = GpuMemory(free_mib=2053, total_mib=16311)
        contention.return_value = {"processes": 1, "memory_mib": 12332}
        snapshot.return_value = {"gpu": "RTX 5060 Ti", "compute_capability": "sm_120"}
        transcribe.return_value = TranscriptionOutcome(
            output_dir=Path("/tmp/benchmarks/artifacts/video/fast/video"),
            text="texto",
            backend="faster-whisper",
            model="turbo",
            device="cuda",
            compute_type="float16",
            batch_size=8,
        )

        with TemporaryDirectory() as temporary:
            report = run_benchmark(
                Path("/home/alguem/videos/video.mp4"),
                ["fast"],
                Path(temporary) / "results" / "benchmark.json",
            )

        self.assertEqual(report["hardware"]["compute_capability"], "sm_120")
        self.assertEqual(report["media"], "video.mp4")
        result = report["results"][0]
        self.assertEqual(result["free_vram_mib_at_start"], 2053)
        self.assertEqual(result["concurrent_compute"], {"processes": 1, "memory_mib": 12332})
        self.assertNotIn("/home/", json.dumps(report))


class OverrideTests(TestCase):
    @patch("transcriber.benchmark.hardware_snapshot", return_value={})
    @patch("transcriber.benchmark.concurrent_compute_usage", return_value=None)
    @patch("transcriber.benchmark.gpu_memory", return_value=None)
    @patch("transcriber.benchmark.transcribe_file")
    @patch("transcriber.benchmark.resolve_device", return_value="cuda")
    @patch("transcriber.benchmark.probe_cuda")
    @patch("transcriber.benchmark._prepare_model")
    @patch("transcriber.benchmark.media_duration", return_value=60.0)
    def test_overrides_reach_the_configuration_and_the_report(
        self, _duration, _prepare, _probe, _resolve, transcribe, *_mocks
    ):
        transcribe.return_value = TranscriptionOutcome(
            output_dir=Path("/tmp/out"),
            text="texto",
            backend="faster-whisper",
            model="large-v3",
            device="cuda",
            compute_type="int8_float16",
            batch_size=16,
        )

        with TemporaryDirectory() as temporary:
            report = run_benchmark(
                Path("/tmp/video.mp4"),
                ["quality"],
                Path(temporary) / "benchmark.json",
                model="large-v3",
                compute_type="int8_float16",
                batch_size=16,
                window_batching=True,
            )

        config = transcribe.call_args.args[1]
        self.assertEqual(config.model, "large-v3")
        self.assertEqual(config.compute_type, "int8_float16")
        self.assertEqual(config.batch_size, 16)
        self.assertTrue(config.batch_size_explicit)
        self.assertTrue(config.window_batching)
        self.assertEqual(report["overrides"]["batch_size"], 16)
        self.assertTrue(report["overrides"]["window_batching"])

    @patch("transcriber.benchmark.hardware_snapshot", return_value={})
    @patch("transcriber.benchmark.concurrent_compute_usage", return_value=None)
    @patch("transcriber.benchmark.gpu_memory", return_value=None)
    @patch("transcriber.benchmark.transcribe_file")
    @patch("transcriber.benchmark.resolve_device", return_value="cuda")
    @patch("transcriber.benchmark.probe_cuda")
    @patch("transcriber.benchmark._prepare_model")
    @patch("transcriber.benchmark.media_duration", return_value=60.0)
    def test_without_overrides_the_profile_decides(
        self, _duration, _prepare, _probe, _resolve, transcribe, *_mocks
    ):
        transcribe.return_value = TranscriptionOutcome(
            output_dir=Path("/tmp/out"),
            text="texto",
            backend="faster-whisper",
            model="large-v3",
            device="cuda",
        )

        with TemporaryDirectory() as temporary:
            run_benchmark(
                Path("/tmp/video.mp4"), ["quality"], Path(temporary) / "benchmark.json"
            )

        config = transcribe.call_args.args[1]
        self.assertEqual(config.model, "large-v3")
        self.assertFalse(config.batch_size_explicit)
        self.assertFalse(config.window_batching)
