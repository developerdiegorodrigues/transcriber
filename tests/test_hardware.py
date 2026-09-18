from unittest import TestCase
from unittest.mock import patch

from transcriber.errors import HardwareError
from transcriber.hardware import CudaStatus, resolve_device


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
