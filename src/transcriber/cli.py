"""Command-line interface."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .backends.openai_whisper import transcribe_file
from .config import OUTPUT_FORMATS, TranscriptionConfig
from .errors import TranscriberError
from .hardware import diagnostic_lines, probe_cuda, resolve_device
from .media import list_video_files, validate_video_file
from .ui import select_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcreve vídeos localmente com OpenAI Whisper.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="diagnostica dependências, driver e CUDA")

    transcribe = subparsers.add_parser("transcribe", help="transcreve um ou mais vídeos")
    transcribe.add_argument("files", nargs="*", type=Path, help="arquivos de vídeo")
    transcribe.add_argument("--model", default="large", help="modelo Whisper (padrão: large)")
    transcribe.add_argument(
        "--language",
        default="English",
        help="idioma ou 'auto' para detecção automática (padrão: English)",
    )
    transcribe.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="dispositivo de inferência (padrão: auto)",
    )
    transcribe.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="diretório de saída"
    )
    transcribe.add_argument(
        "--output-format", choices=OUTPUT_FORMATS, default="all", help="formato de saída"
    )
    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["transcribe"]
    if argv[0] in {"doctor", "transcribe", "-h", "--help", "--version"}:
        return argv
    return ["transcribe", *argv]


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise TranscriberError("ffmpeg não encontrado. No Ubuntu, instale com: sudo apt install ffmpeg")


def _interactive_files() -> list[Path]:
    files = list_video_files(Path.cwd())
    if not files:
        raise TranscriberError(f"Nenhum vídeo encontrado em: {Path.cwd()}")
    selected = select_file(files)
    return [selected] if selected else []


def _run_transcription(args: argparse.Namespace) -> int:
    _require_ffmpeg()
    cuda = probe_cuda()
    device = resolve_device(args.device, cuda)
    if args.device == "auto" and device == "cpu":
        print("Aviso: CUDA indisponível; usando CPU. Execute 'transcriber doctor' para detalhes.")

    language = None if args.language.casefold() == "auto" else args.language
    config = TranscriptionConfig(
        model=args.model,
        language=language,
        requested_device=args.device,
        output_dir=args.output_dir,
        output_format=args.output_format,
    )
    files = [validate_video_file(path) for path in args.files] if args.files else _interactive_files()
    for media_path in files:
        transcribe_file(media_path, config, device)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(list(sys.argv[1:] if argv is None else argv)))
    try:
        if args.command == "doctor":
            print("\n".join(diagnostic_lines()))
            return 0
        return _run_transcription(args)
    except (TranscriberError, ValueError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
