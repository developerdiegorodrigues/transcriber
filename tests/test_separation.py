import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from transcriber.separation import separated_vocals


class SeparationTests(TestCase):
    @patch("transcriber.separation.subprocess.run")
    def test_vocal_stem_uses_original_media_name_and_is_temporary(self, run):
        def fake_run(command, check):
            if command[0] == "ffmpeg":
                Path(command[-1]).touch()
            else:
                output = Path(command[command.index("--out") + 1])
                vocals = output / "htdemucs" / "source" / "vocals.wav"
                vocals.parent.mkdir(parents=True)
                vocals.touch()
            return subprocess.CompletedProcess(command, 0)

        run.side_effect = fake_run
        with TemporaryDirectory() as temporary_directory:
            media = Path(temporary_directory) / "song.mp4"
            media.touch()
            with separated_vocals(media, device="cpu") as vocal_path:
                self.assertEqual(vocal_path.name, "song.wav")
                self.assertTrue(vocal_path.exists())
            self.assertFalse(vocal_path.exists())

        self.assertEqual(run.call_count, 2)
