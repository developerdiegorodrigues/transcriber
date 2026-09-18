"""Command-line interface."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .benchmark import run_benchmark
from .config import BACKENDS, OUTPUT_FORMATS, PROFILES, config_from_profile
from .errors import TranscriberError
from .hardware import diagnostic_lines, probe_cuda, resolve_device
from .media import list_video_files, validate_video_file
from .pipeline import transcribe_file
from .ui import select_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcreve vídeos localmente com OpenAI Whisper.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.4.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="diagnostica dependências, driver e CUDA")

    transcribe = subparsers.add_parser("transcribe", help="transcreve um ou mais vídeos")
    transcribe.add_argument("files", nargs="*", type=Path, help="arquivos de vídeo")
    transcribe.add_argument(
        "--profile", choices=tuple(PROFILES), default="fast", help="perfil de execução"
    )
    transcribe.add_argument("--backend", choices=BACKENDS, help="sobrescreve o backend do perfil")
    transcribe.add_argument("--model", help="sobrescreve o modelo do perfil")
    transcribe.add_argument(
        "--language",
        default="English",
        help="idioma ou 'auto' para detecção automática (padrão: English)",
    )
    transcribe.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="sobrescreve o dispositivo do perfil",
    )
    transcribe.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="diretório de saída"
    )
    transcribe.add_argument(
        "--output-format", choices=OUTPUT_FORMATS, default="all", help="formato de saída"
    )
    transcribe.add_argument("--compute-type", help="precisão do CTranslate2")
    transcribe.add_argument("--batch-size", type=int, help="batch inicial do faster-whisper")
    transcribe.add_argument("--beam-size", type=int, help="largura da busca")
    transcribe.add_argument(
        "--vad", action=argparse.BooleanOptionalAction, default=None, help="filtro de voz"
    )
    transcribe.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="timestamps por palavra",
    )
    transcribe.add_argument(
        "--separate-vocals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="isola os vocais com Demucs antes da transcrição",
    )
    transcribe.add_argument(
        "--separation-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="dispositivo usado pelo Demucs",
    )
    transcribe.add_argument("--demucs-model", default="htdemucs", help="modelo do Demucs")
    transcribe.add_argument(
        "--quality-review",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="gera relatório de trechos com baixa confiança",
    )
    transcribe.add_argument(
        "--retry-low-confidence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reavalia intervalos de baixa confiança",
    )

    benchmark = subparsers.add_parser("benchmark", help="compara perfis de transcrição")
    benchmark.add_argument("file", type=Path, help="vídeo usado na comparação")
    benchmark.add_argument(
        "--profiles", nargs="+", choices=tuple(PROFILES), default=["fast", "quality", "cpu"]
    )
    benchmark.add_argument("--language", default="English", help="idioma ou 'auto'")
    benchmark.add_argument("--reference", type=Path, help="texto de referência para calcular WER")
    benchmark.add_argument("--output", type=Path, help="arquivo JSON do relatório")
    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    if not argv:
        return ["transcribe"]
    if argv[0] in {"benchmark", "doctor", "transcribe", "-h", "--help", "--version"}:
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
    language = None if args.language.casefold() == "auto" else args.language
    config = config_from_profile(
        args.profile,
        backend=args.backend,
        model=args.model,
        language=language,
        requested_device=args.device,
        output_dir=args.output_dir,
        output_format=args.output_format,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        word_timestamps=args.word_timestamps,
        separate_vocals=args.separate_vocals,
        separation_device=args.separation_device,
        demucs_model=args.demucs_model,
        quality_review=args.quality_review,
        retry_low_confidence=args.retry_low_confidence,
    )
    cuda = probe_cuda()
    device = resolve_device(config.requested_device, cuda, config.backend)
    if config.requested_device == "auto" and device == "cpu":
        print("Aviso: CUDA indisponível; usando CPU. Execute 'transcriber doctor' para detalhes.")
    files = [validate_video_file(path) for path in args.files] if args.files else _interactive_files()
    for media_path in files:
        transcribe_file(media_path, config, device)
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    media_path = validate_video_file(args.file)
    language = None if args.language.casefold() == "auto" else args.language
    reference = args.reference.read_text(encoding="utf-8") if args.reference else None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or Path("benchmarks/results") / f"benchmark-{timestamp}.json"
    report = run_benchmark(
        media_path, args.profiles, output, language=language, reference=reference
    )
    print(f"Benchmark concluído: {output}")
    for result in report["results"]:
        print(
            f"{result['profile']}: {result['elapsed_seconds']}s, "
            f"{result['audio_x_realtime']}x tempo real, "
            f"VRAM pico: {result['peak_gpu_memory_mib'] or 'n/d'} MiB"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(list(sys.argv[1:] if argv is None else argv)))
    try:
        if args.command == "doctor":
            print("\n".join(diagnostic_lines()))
            return 0
        if args.command == "benchmark":
            return _run_benchmark(args)
        return _run_transcription(args)
    except (TranscriberError, ValueError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1
