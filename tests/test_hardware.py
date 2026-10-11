from subprocess import CompletedProcess
from unittest import TestCase
from unittest.mock import patch

from transcriber.cuda_runtime import CudaLibraryStatus
from transcriber.errors import HardwareError
from transcriber.hardware import (
    CudaStatus,
    GpuMemory,
    concurrent_compute_usage,
    gpu_memory,
    _ctranslate2_cuda_available,
    cuda_smoke_test,
    diagnostic_lines,
    resolve_device,
)


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


class CtranslateCudaProbeTests(TestCase):
    @patch("transcriber.hardware.preload_cuda_libraries")
    @patch("transcriber.hardware._ctranslate2_driver_visible", return_value=(True, None))
    def test_visible_driver_is_not_enough_without_the_libraries(self, _visible, preload):
        preload.return_value = CudaLibraryStatus(missing=("libcublas.so.12",))

        available, error = _ctranslate2_cuda_available()

        self.assertFalse(available)
        self.assertIn("libcublas.so.12", error)

    @patch("transcriber.hardware.preload_cuda_libraries")
    @patch("transcriber.hardware._ctranslate2_driver_visible", return_value=(True, None))
    def test_available_when_driver_and_libraries_agree(self, _visible, preload):
        preload.return_value = CudaLibraryStatus(loaded=("libcublas.so.12",))

        self.assertEqual(_ctranslate2_cuda_available(), (True, None))

    @patch("transcriber.hardware.preload_cuda_libraries")
    @patch("transcriber.hardware._ctranslate2_driver_visible", return_value=(True, None))
    def test_explicit_cuda_fails_when_libraries_are_missing(self, _visible, preload):
        preload.return_value = CudaLibraryStatus(missing=("libcublas.so.12",))

        with self.assertRaisesRegex(HardwareError, "transcriber doctor"):
            resolve_device("cuda", CudaStatus("2.0", "13.0", True), "faster-whisper")

    @patch("transcriber.hardware.preload_cuda_libraries")
    @patch("transcriber.hardware._ctranslate2_driver_visible", return_value=(True, None))
    def test_auto_falls_back_to_cpu_when_libraries_are_missing(self, _visible, preload):
        preload.return_value = CudaLibraryStatus(missing=("libcublas.so.12",))

        self.assertEqual(
            resolve_device("auto", CudaStatus("2.0", "13.0", True), "faster-whisper"),
            "cpu",
        )

    @patch("transcriber.hardware._ctranslate2_cuda_available")
    def test_smoke_test_reports_the_missing_library_without_loading_a_model(self, available):
        available.return_value = (False, "bibliotecas CUDA do CTranslate2 ausentes")

        passed, error = cuda_smoke_test()

        self.assertFalse(passed)
        self.assertIn("ausentes", error)


class DiagnosticDivergenceTests(TestCase):
    def _report(self, *, libraries_complete: bool) -> dict:
        return {
            "system": {"platform": "Linux", "python_version": "3.12.0"},
            "dependencies": {
                "openai-whisper": "1.0",
                "faster-whisper": "1.2",
                "ctranslate2": "4.8",
                "demucs": "4.1",
                "ffmpeg_available": True,
                "whisper_cli_available": True,
                "nvidia_smi": "NVIDIA GPU, 000.00, 16384 MiB",
            },
            "cuda": {
                "pytorch": {
                    "available": True,
                    "torch_version": "2.12",
                    "runtime_version": "13.0",
                    "error": None,
                },
                "ctranslate2": {
                    "available": libraries_complete,
                    "driver_visible": True,
                    "driver_error": None,
                    "error": None,
                    "libraries": {
                        "complete": libraries_complete,
                        "loaded": [],
                        "missing": [] if libraries_complete else ["libcublas.so.12"],
                    },
                },
                "devices": [],
            },
            "backends": {
                "openai-whisper": {"automatic_device": "cuda"},
                "faster-whisper": {
                    "automatic_device": "cuda" if libraries_complete else "cpu"
                },
            },
        }

    def test_warns_when_the_driver_is_visible_but_libraries_are_not(self):
        lines = diagnostic_lines(self._report(libraries_complete=False))

        self.assertIn("Bibliotecas ausentes: libcublas.so.12", lines)
        self.assertTrue(any("transcrição em gpu falharia" in line.casefold() for line in lines))

    def test_stays_quiet_when_the_stack_is_consistent(self):
        lines = diagnostic_lines(self._report(libraries_complete=True))

        self.assertIn("Bibliotecas CUDA do CTranslate2: completas", lines)
        self.assertFalse(any("falharia" in line for line in lines))


class GpuMemoryTests(TestCase):
    @patch("transcriber.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("transcriber.hardware.subprocess.run")
    def test_parses_free_and_total_memory(self, run, _which):
        run.return_value = CompletedProcess([], 0, stdout="1967, 16311\n", stderr="")

        memory = gpu_memory()

        self.assertEqual(memory, GpuMemory(free_mib=1967, total_mib=16311))

    @patch("transcriber.hardware.shutil.which", return_value=None)
    def test_absent_nvidia_smi_is_not_an_error(self, _which):
        self.assertIsNone(gpu_memory())

    @patch("transcriber.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("transcriber.hardware.subprocess.run")
    def test_unexpected_output_is_ignored(self, run, _which):
        run.return_value = CompletedProcess([], 0, stdout="[N/A], [N/A]\n", stderr="")

        self.assertIsNone(gpu_memory())

    @patch("transcriber.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("transcriber.hardware.subprocess.run")
    def test_failed_query_is_ignored(self, run, _which):
        run.return_value = CompletedProcess([], 9, stdout="", stderr="falhou")

        self.assertIsNone(gpu_memory())


class ConcurrentComputeTests(TestCase):
    @patch("transcriber.hardware.os.getpid", return_value=4242)
    @patch("transcriber.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("transcriber.hardware.subprocess.run")
    def test_own_process_is_excluded_from_the_contention(self, run, _which, _pid):
        run.return_value = CompletedProcess(
            [], 0, stdout="4242, 1500\n9999, 12332\n", stderr=""
        )

        self.assertEqual(
            concurrent_compute_usage(), {"processes": 1, "memory_mib": 12332}
        )

    @patch("transcriber.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("transcriber.hardware.subprocess.run")
    def test_idle_gpu_reports_nothing_running(self, run, _which):
        run.return_value = CompletedProcess([], 0, stdout="\n", stderr="")

        self.assertEqual(concurrent_compute_usage(), {"processes": 0, "memory_mib": 0})

    @patch("transcriber.hardware.shutil.which", return_value=None)
    def test_absent_nvidia_smi_is_not_an_error(self, _which):
        self.assertIsNone(concurrent_compute_usage())
