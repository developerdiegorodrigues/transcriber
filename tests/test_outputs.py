import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from transcriber.outputs import write_outputs


class OutputWriterTests(TestCase):
    def test_writes_all_whisper_formats_with_original_stem(self):
        result = {
            "text": " Hello",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": " Hello"}],
        }
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)

            write_outputs(result, Path("example.mp4"), output, "all")

            self.assertEqual(
                {path.suffix for path in output.iterdir()},
                {".txt", ".vtt", ".srt", ".tsv", ".json"},
            )
            saved = json.loads((output / "example.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["text"], " Hello")
