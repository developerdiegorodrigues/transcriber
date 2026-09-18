from unittest import TestCase

from transcriber.cli import _normalize_argv, build_parser


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
