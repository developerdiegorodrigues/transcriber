from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from transcriber.backends.openai_whisper import build_command, find_whisper_executable
from transcriber.config import TranscriptionConfig


class OpenAIWhisperBackendTests(TestCase):
    def test_finds_whisper_beside_symlinked_venv_python(self):
        with TemporaryDirectory() as temporary_directory:
            bin_directory = Path(temporary_directory) / "bin"
            bin_directory.mkdir()
            python = bin_directory / "python"
            python.symlink_to("/usr/bin/python3")
            whisper = bin_directory / "whisper"
            whisper.touch()
            with patch("transcriber.backends.openai_whisper.sys.executable", str(python)):
                self.assertEqual(find_whisper_executable(), whisper)

    def test_build_command_uses_media_directly_and_cuda_fp16(self):
        media = Path("video.mp4")
        config = TranscriptionConfig(
            model="large-v3",
            language="Portuguese",
            output_dir=Path("output"),
        )

        command = build_command(media, Path("output/video"), config, "cuda", Path("whisper"))

        self.assertEqual(command[1], "video.mp4")
        self.assertNotIn("audio.wav", command)
        self.assertEqual(command[command.index("--fp16") + 1], "True")
        self.assertEqual(command[command.index("--language") + 1], "Portuguese")

    def test_auto_language_is_omitted(self):
        config = TranscriptionConfig(language=None)

        command = build_command(Path("video.mp4"), Path("out"), config, "cpu", Path("whisper"))

        self.assertNotIn("--language", command)
        self.assertEqual(command[command.index("--fp16") + 1], "False")
