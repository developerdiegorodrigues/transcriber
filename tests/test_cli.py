import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import TestCase
from unittest.mock import patch

from transcriber.cli import _normalize_argv, build_parser, main


class CliTests(TestCase):
    def test_file_without_subcommand_keeps_short_syntax(self):
        normalized = _normalize_argv(["video.mp4", "--device", "cpu"])
        args = build_parser().parse_args(normalized)

        self.assertEqual(args.command, "transcribe")
        self.assertEqual(str(args.files[0]), "video.mp4")
        self.assertEqual(args.device, "cpu")

    def test_empty_arguments_start_interactive_transcription(self):
        self.assertEqual(_normalize_argv([]), ["transcribe"])

    def test_doctor_is_preserved(self):
        self.assertEqual(_normalize_argv(["doctor"]), ["doctor"])

    def test_fast_profile_is_default(self):
        args = build_parser().parse_args(["transcribe", "video.mp4"])
        self.assertEqual(args.profile, "fast")

    def test_benchmark_is_preserved(self):
        self.assertEqual(_normalize_argv(["benchmark", "video.mp4"])[0], "benchmark")

    @patch("transcriber.cli.diagnostic_report")
    def test_doctor_json_passes_required_cuda(self, diagnostic_report):
        diagnostic_report.return_value = {
            "backends": {
                "faster-whisper": {"cuda_available": True},
                "openai-whisper": {"cuda_available": False},
            }
        }
        stdout = StringIO()

        with redirect_stdout(stdout):
            result = main(
                ["doctor", "--json", "--require-cuda", "--backend", "faster-whisper"]
            )

        report = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(report["validation"]["passed"])

    @patch("transcriber.cli.diagnostic_lines", return_value=["diagnóstico"])
    @patch("transcriber.cli.diagnostic_report")
    def test_doctor_fails_when_backend_cannot_access_cuda(
        self, diagnostic_report, _diagnostic_lines
    ):
        diagnostic_report.return_value = {
            "backends": {
                "faster-whisper": {"cuda_available": False},
                "openai-whisper": {"cuda_available": True},
            }
        }
        stderr = StringIO()

        with redirect_stdout(StringIO()), redirect_stderr(stderr):
            result = main(["doctor", "--require-cuda", "--backend", "faster-whisper"])

        self.assertEqual(result, 1)
        self.assertIn("FALHOU", stderr.getvalue())


class DoctorSmokeTestTests(TestCase):
    def _report(self) -> dict:
        return {
            "backends": {
                "faster-whisper": {"cuda_available": True},
                "openai-whisper": {"cuda_available": True},
            }
        }

    @patch("transcriber.cli.cuda_smoke_test", return_value=(False, "libcublas.so.12 ausente"))
    @patch("transcriber.cli.diagnostic_lines", return_value=["diagnóstico"])
    @patch("transcriber.cli.diagnostic_report")
    def test_require_cuda_fails_when_the_gpu_cannot_actually_run(
        self, diagnostic_report, _lines, _smoke
    ):
        diagnostic_report.return_value = self._report()
        stderr = StringIO()

        with redirect_stdout(StringIO()), redirect_stderr(stderr):
            result = main(["doctor", "--require-cuda", "--smoke-test"])

        self.assertEqual(result, 1)
        self.assertIn("FALHOU", stderr.getvalue())
        self.assertIn("libcublas.so.12 ausente", stderr.getvalue())

    @patch("transcriber.cli.cuda_smoke_test", return_value=(True, None))
    @patch("transcriber.cli.diagnostic_lines", return_value=["diagnóstico"])
    @patch("transcriber.cli.diagnostic_report")
    def test_passing_smoke_test_is_reported(self, diagnostic_report, _lines, _smoke):
        diagnostic_report.return_value = self._report()
        stdout = StringIO()

        with redirect_stdout(stdout), redirect_stderr(StringIO()):
            result = main(["doctor", "--require-cuda", "--smoke-test"])

        self.assertEqual(result, 0)
        self.assertIn("Teste de fumaça na GPU: OK", stdout.getvalue())

    @patch("transcriber.cli.cuda_smoke_test")
    @patch("transcriber.cli.diagnostic_report")
    def test_smoke_test_is_rejected_for_the_legacy_backend(
        self, diagnostic_report, smoke
    ):
        diagnostic_report.return_value = self._report()
        stderr = StringIO()

        with redirect_stdout(StringIO()), redirect_stderr(stderr):
            result = main(["doctor", "--smoke-test", "--backend", "openai-whisper"])

        self.assertEqual(result, 1)
        self.assertIn("faster-whisper", stderr.getvalue())
        smoke.assert_not_called()

    @patch("transcriber.cli.cuda_smoke_test")
    @patch("transcriber.cli.diagnostic_lines", return_value=["diagnóstico"])
    @patch("transcriber.cli.diagnostic_report")
    def test_smoke_test_is_not_run_unless_requested(
        self, diagnostic_report, _lines, smoke
    ):
        diagnostic_report.return_value = self._report()

        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(["doctor"]), 0)

        smoke.assert_not_called()
