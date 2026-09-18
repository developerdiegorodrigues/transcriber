from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from transcriber.media import list_video_files, validate_video_file


class MediaTests(TestCase):
    def test_lists_supported_videos_case_insensitively(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "B.MKV").touch()
            (directory / "a.mp4").touch()
            (directory / "notes.txt").touch()

            files = list_video_files(directory)

            self.assertEqual([path.name for path in files], ["a.mp4", "B.MKV"])

    def test_rejects_unsupported_file(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notes.txt"
            path.touch()
            with self.assertRaisesRegex(ValueError, "Formato não suportado"):
                validate_video_file(path)
