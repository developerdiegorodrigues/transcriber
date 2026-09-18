from unittest import TestCase
from unittest.mock import patch

from transcriber.errors import HardwareError
from transcriber.hardware import CudaStatus, diagnostic_lines, resolve_device


class ResolveDeviceTests(TestCase):
    def test_auto_prefers_cuda(self):
        status = CudaStatus("2.0", "13.0", True, ("GPU",), (16384,))
        self.assertEqual(resolve_device("auto", status), "cuda")

    def test_auto_falls_back_to_cpu(self):
        status = CudaStatus("2.0", "13.0", False)
        self.assertEqual(resolve_device("auto", status), "cpu")

    def test_explicit_cuda_fails_clearly(self):
        status = CudaStatus("2.0", "13.0", False)
        with self.assertRaisesRegex(HardwareError, "transcriber doctor"):
            resolve_device("cuda", status)

    def test_explicit_cpu_does_not_require_cuda(self):
        status = CudaStatus(None, None, False, error="torch ausente")
        self.assertEqual(resolve_device("cpu", status), "cpu")

    @patch("transcriber.hardware._ctranslate2_cuda_available", return_value=(True, None))
    def test_faster_whisper_uses_ctranslate2_probe(self, _probe):
        torch_status = CudaStatus("2.0", "13.0", False)
        self.assertEqual(
            resolve_device("auto", torch_status, "faster-whisper"),
            "cuda",
        )


class DiagnosticLinesTests(TestCase):
    def test_formats_machine_readable_report(self):
        report = {
            "system": {"platform": "Linux", "python_version": "3.12.0"},
            "dependencies": {
                "openai-whisper": "1.0",
                "faster-whisper": "1.2",
                "ctranslate2": "4.8",
                "demucs": "4.1",
                "ffmpeg_available": True,
                "whisper_cli_available": True,
                "nvidia_smi": "RTX 5060 Ti, 600.00, 16384 MiB",
            },
            "cuda": {
                "pytorch": {
                    "available": True,
                    "torch_version": "2.12",
                    "runtime_version": "13.0",
                    "error": None,
                },
                "ctranslate2": {"available": True, "error": None},
                "devices": [
                    {"index": 0, "name": "RTX 5060 Ti", "total_memory_mib": 16384}
                ],
            },
            "backends": {
                "openai-whisper": {"automatic_device": "cuda"},
                "faster-whisper": {"automatic_device": "cuda"},
            },
        }

        lines = diagnostic_lines(report)

        self.assertIn("Python: 3.12.0", lines)
        self.assertIn("GPU 0: RTX 5060 Ti (16384 MiB)", lines)
        self.assertFalse(any("/home/" in line for line in lines))
