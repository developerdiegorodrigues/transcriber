from pathlib import Path
from unittest import TestCase

from transcriber.config import config_from_profile


class ProfileTests(TestCase):
    def test_fast_profile_targets_fp16_on_cuda(self):
        config = config_from_profile("fast")
        self.assertEqual(config.backend, "faster-whisper")
        self.assertEqual(config.profile_name, "fast")
        self.assertEqual(config.model, "turbo")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.compute_type_for("cuda"), "float16")

    def test_cpu_profile_uses_small_int8(self):
        config = config_from_profile("cpu")
        self.assertEqual(config.requested_device, "cpu")
        self.assertEqual(config.model, "small")
        self.assertEqual(config.compute_type_for("cpu"), "int8")

    def test_overrides_take_precedence(self):
        config = config_from_profile(
            "fast",
            model="large-v3",
            batch_size=2,
            output_dir=Path("custom"),
        )
        self.assertEqual(config.model, "large-v3")
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.output_dir, Path("custom"))

    def test_lyrics_profile_enables_vocal_quality_pipeline(self):
        config = config_from_profile("lyrics")
        self.assertEqual(config.model, "large-v3")
        self.assertTrue(config.separate_vocals)
        self.assertTrue(config.quality_review)
        self.assertTrue(config.retry_low_confidence)
        self.assertFalse(config.condition_on_previous_text)
        self.assertNotIn(1.0, config.temperatures)


class DemucsModelTests(TestCase):
    def test_lyrics_defaults_to_the_fine_tuned_ensemble(self):
        # Measured: 42% fewer low-confidence segments for 2.4x the time, same VRAM.
        self.assertEqual(config_from_profile("lyrics").demucs_model, "htdemucs_ft")

    def test_other_profiles_keep_the_fast_separation(self):
        for name in ("fast", "quality", "cpu"):
            self.assertEqual(config_from_profile(name).demucs_model, "htdemucs")

    def test_explicit_model_overrides_the_profile(self):
        config = config_from_profile("lyrics", demucs_model="htdemucs")

        self.assertEqual(config.demucs_model, "htdemucs")
