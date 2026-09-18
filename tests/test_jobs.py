import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from transcriber.config import TranscriptionConfig
from transcriber.errors import TranscriptionError
from transcriber.jobs import run_job
from transcriber.result import TranscriptionOutcome


def fake_transcription(media_path, config, device):
    output = config.output_dir / media_path.stem
    output.mkdir(parents=True)
    (output / f"{media_path.stem}.txt").write_text("transcript\n", encoding="utf-8")
    return TranscriptionOutcome(
        output_dir=output,
        text="transcript",
        backend=config.backend,
        model=config.model,
        device=device,
        compute_type="int8",
        batch_size=1,
    )


class JobTests(TestCase):
    @patch("transcriber.jobs.transcribe_file", side_effect=fake_transcription)
    def test_publishes_complete_job_with_metadata(self, _transcribe):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            media = root / "video.mp4"
            media.write_bytes(b"media")
            config = TranscriptionConfig(output_dir=root / "output")

            outcome = run_job(media, config, "cpu")

            self.assertEqual(outcome.output_dir, root / "output" / "video")
            metadata_path = outcome.output_dir / "video.metadata.json"
            report = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["media"]["name"], "video.mp4")
            self.assertNotIn(str(root), json.dumps(report["media"]))
            self.assertNotIn("output_dir", report["config"])
            self.assertFalse(list((root / "output").glob(".transcriber-*")))

    @patch("transcriber.jobs.transcribe_file", side_effect=fake_transcription)
    def test_skip_existing_does_not_transcribe(self, transcribe):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            media = root / "video.mp4"
            media.touch()
            final = root / "output" / "video"
            final.mkdir(parents=True)
            config = TranscriptionConfig(output_dir=root / "output")

            outcome = run_job(media, config, "cpu", skip_existing=True)

            self.assertIsNone(outcome)
            transcribe.assert_not_called()

    @patch("transcriber.jobs.transcribe_file", side_effect=fake_transcription)
    def test_force_replaces_existing_only_after_success(self, _transcribe):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            media = root / "video.mp4"
            media.touch()
            final = root / "output" / "video"
            final.mkdir(parents=True)
            (final / "old.txt").write_text("old", encoding="utf-8")
            config = TranscriptionConfig(output_dir=root / "output")

            run_job(media, config, "cpu", force=True)

            self.assertFalse((final / "old.txt").exists())
            self.assertTrue((final / "video.txt").exists())
            self.assertFalse(list((root / "output").glob(".*.backup-*")))

    def test_existing_output_requires_explicit_policy(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            media = root / "video.mp4"
            media.touch()
            (root / "output" / "video").mkdir(parents=True)
            config = TranscriptionConfig(output_dir=root / "output")

            with self.assertRaisesRegex(TranscriptionError, "--skip-existing"):
                run_job(media, config, "cpu")

    @patch("transcriber.jobs.transcribe_file", side_effect=KeyboardInterrupt)
    def test_interruption_removes_staging_directory(self, _transcribe):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            media = root / "video.mp4"
            media.touch()
            output = root / "output"
            config = TranscriptionConfig(output_dir=output)

            with self.assertRaises(KeyboardInterrupt):
                run_job(media, config, "cpu")

            self.assertFalse(list(output.glob(".transcriber-*")))
