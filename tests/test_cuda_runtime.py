from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from transcriber import cuda_runtime
from transcriber.cuda_runtime import (
    CudaLibraryStatus,
    REQUIRED_LIBRARIES,
    library_directories,
    preload_cuda_libraries,
)


class CudaLibraryStatusTests(TestCase):
    def test_complete_when_nothing_is_missing(self):
        status = CudaLibraryStatus(loaded=("libcublas.so.12",))

        self.assertTrue(status.complete)
        self.assertIsNone(status.describe_missing())

    def test_missing_description_names_the_package_to_install(self):
        status = CudaLibraryStatus(missing=("libcublas.so.12",))

        description = status.describe_missing()

        self.assertFalse(status.complete)
        self.assertIn("libcublas.so.12", description)
        self.assertIn("nvidia-cublas-cu12", description)


class LibraryDirectoriesTests(TestCase):
    def test_collects_nvidia_library_directories(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nvidia" / "cublas" / "lib").mkdir(parents=True)
            (root / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
            (root / "nvidia" / "cublas" / "include").mkdir(parents=True)
            with patch.object(
                cuda_runtime, "_site_package_roots", return_value=[root]
            ):
                directories = library_directories()

        self.assertEqual(
            [path.parent.name for path in directories], ["cublas", "cudnn"]
        )

    def test_tolerates_environment_without_nvidia_packages(self):
        with TemporaryDirectory() as temporary:
            with patch.object(
                cuda_runtime, "_site_package_roots", return_value=[Path(temporary)]
            ):
                self.assertEqual(library_directories(), [])


class PreloadTests(TestCase):
    def setUp(self):
        cuda_runtime._cached_status = None
        self.addCleanup(setattr, cuda_runtime, "_cached_status", None)

    def _environment(self, root: Path) -> Path:
        directory = root / "nvidia" / "cublas" / "lib"
        directory.mkdir(parents=True)
        for name in REQUIRED_LIBRARIES:
            (directory / name).write_bytes(b"")
        return directory

    def test_reports_loaded_libraries(self):
        with TemporaryDirectory() as temporary:
            directory = self._environment(Path(temporary))
            with patch.object(cuda_runtime, "library_directories", return_value=[directory]), patch.object(
                cuda_runtime, "_load", return_value=None
            ):
                status = preload_cuda_libraries(refresh=True)

        self.assertTrue(status.complete)
        self.assertEqual(set(status.missing), set())
        for name in REQUIRED_LIBRARIES:
            self.assertIn(name, status.loaded)

    def test_reports_missing_library_when_loading_fails(self):
        with TemporaryDirectory() as temporary:
            directory = self._environment(Path(temporary))
            with patch.object(cuda_runtime, "library_directories", return_value=[directory]), patch.object(
                cuda_runtime, "_load", return_value="não é um ELF válido"
            ), patch.object(cuda_runtime, "_already_loaded", return_value=False):
                status = preload_cuda_libraries(refresh=True)

        self.assertFalse(status.complete)
        self.assertEqual(set(status.missing), set(REQUIRED_LIBRARIES))
        self.assertTrue(status.errors)

    def test_absent_library_is_missing_without_an_error(self):
        with patch.object(cuda_runtime, "library_directories", return_value=[]), patch.object(
            cuda_runtime, "_already_loaded", return_value=False
        ):
            status = preload_cuda_libraries(refresh=True)

        self.assertFalse(status.complete)
        self.assertEqual(status.loaded, ())
        self.assertEqual(status.errors, ())

    def test_library_already_present_in_the_process_is_not_missing(self):
        with patch.object(cuda_runtime, "library_directories", return_value=[]), patch.object(
            cuda_runtime, "_already_loaded", return_value=True
        ):
            status = preload_cuda_libraries(refresh=True)

        self.assertTrue(status.complete)

    def test_result_is_cached_because_loading_is_a_process_side_effect(self):
        with patch.object(cuda_runtime, "library_directories", return_value=[]) as directories, patch.object(
            cuda_runtime, "_already_loaded", return_value=True
        ):
            first = preload_cuda_libraries(refresh=True)
            second = preload_cuda_libraries()

        self.assertIs(first, second)
        self.assertEqual(directories.call_count, 1)
