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
