#!/usr/bin/env python3
"""
Transcrição de vídeo usando OpenAI Whisper.

Exibe uma lista interativa dos arquivos de vídeo presentes no diretório,
converte o selecionado para WAV via ffmpeg e transcreve com Whisper.

Uso:
    python3 transcribe.py
"""

import os
import sys
import subprocess
import curses
import shutil
from pathlib import Path

# ─── Configurações ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
VENV_DIR   = SCRIPT_DIR / ".venv"
OUTPUT_DIR = SCRIPT_DIR / "output"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}

WHISPER_MODEL    = "large"
WHISPER_LANGUAGE = "English"
WHISPER_DEVICE   = "cpu"


# ─── Ambiente Virtual ─────────────────────────────────────────────────────────

def _is_in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def setup_venv() -> None:
    """Garante que o script está rodando dentro do .venv com todas as dependências."""
    venv_python = VENV_DIR / "bin" / "python"
    venv_pip    = VENV_DIR / "bin" / "pip"

    if not _is_in_venv():
        # Cria o venv se ainda não existe
        if not venv_python.exists():
            print("Criando ambiente virtual .venv ...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                check=True,
            )
            print("Ambiente virtual criado.\n")

        # Instala / atualiza dependências
        print("Instalando/atualizando openai-whisper ...")
        subprocess.run(
            [str(venv_pip), "install", "-U", "openai-whisper"],
            check=True,
        )

        # Re-executa o script dentro do venv (substitui o processo atual)
        print("\nReiniciando no ambiente virtual...\n")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
        sys.exit(0)  # nunca alcançado — segurança

    # Já está no venv: verifica se o binário whisper existe
    whisper_bin = VENV_DIR / "bin" / "whisper"
    if not whisper_bin.exists():
        print("Instalando openai-whisper no venv ativo ...")
        subprocess.run(
            [str(venv_pip), "install", "-U", "openai-whisper"],
            check=True,
        )


# ─── Verificações do Sistema ──────────────────────────────────────────────────

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERRO: ffmpeg não encontrado no sistema.")
        print("      Instale com: sudo apt install ffmpeg")
        sys.exit(1)


# ─── Listagem de Arquivos ─────────────────────────────────────────────────────

def list_video_files() -> list:
    return sorted(
        f for f in SCRIPT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )


# ─── Interface TUI (curses) ───────────────────────────────────────────────────

def select_file_curses(files: list):
    """Exibe lista interativa via curses. Retorna Path selecionado ou None."""

    def draw(stdscr, selected: int, scroll: int) -> None:
        stdscr.clear()
        h, w = stdscr.getmaxyx()

        # Cabeçalho
        title = " Transcrição com Whisper — Selecione o vídeo "
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(0, 0, title.center(w - 1)[:w - 1])
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(1, 0, "─" * (w - 1))

        # Lista de arquivos
        visible_rows = h - 5
        for i in range(min(visible_rows, len(files) - scroll)):
            real_i = i + scroll
            row    = i + 2
            name   = files[real_i].name
            if len(name) > w - 7:
                name = name[: w - 10] + "..."
            prefix = " ► " if real_i == selected else "   "
            line   = (prefix + name).ljust(w - 1)

            if real_i == selected:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(row, 0, line[:w - 1])
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr(row, 0, line[:w - 1])

        # Rodapé
        help_line = " ↑↓/jk: navegar   Enter: selecionar   q/Esc: sair "
        counter   = f" {selected + 1}/{len(files)} "
        stdscr.addstr(h - 2, 0, "─" * (w - 1))
        footer = (help_line + counter)[:w - 1]
        stdscr.addstr(h - 1, 0, footer)
        stdscr.refresh()

    def run(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)   # item selecionado
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # cabeçalho

        selected = 0
        scroll   = 0

        while True:
            h, _ = stdscr.getmaxyx()
            visible_rows = h - 5
            draw(stdscr, selected, scroll)
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord("k")):
                if selected > 0:
                    selected -= 1
                    if selected < scroll:
                        scroll = selected

            elif key in (curses.KEY_DOWN, ord("j")):
                if selected < len(files) - 1:
                    selected += 1
                    if selected >= scroll + visible_rows:
                        scroll = selected - visible_rows + 1

            elif key in (curses.KEY_ENTER, 10, 13):
                return files[selected]

            elif key in (ord("q"), ord("Q"), 27):  # 27 = Esc
                return None

    return curses.wrapper(run)


# ─── Conversão de Vídeo ───────────────────────────────────────────────────────

def convert_to_wav(video_path: Path) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    wav_path = OUTPUT_DIR / "audio.wav"

    print(f"\nConvertendo '{video_path.name}' para WAV ...")
    cmd = [
        "ffmpeg", "-y",
        "-i",      str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar",     "16000",
        "-ac",     "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"ERRO na conversão:\n{result.stderr}")
        sys.exit(1)

    size_mb = wav_path.stat().st_size / (1024 * 1024)
    print(f"Conversão concluída — {wav_path.name} ({size_mb:.1f} MB)")
    return wav_path


# ─── Transcrição Whisper ──────────────────────────────────────────────────────

def transcribe(wav_path: Path, video_name: str) -> Path:
    stem          = Path(video_name).stem
    output_subdir = OUTPUT_DIR / stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    whisper_bin = VENV_DIR / "bin" / "whisper"

    print(f"\nTranscrevendo com modelo '{WHISPER_MODEL}' (device: {WHISPER_DEVICE}) ...")
    print("Este processo pode levar vários minutos em CPU.\n")

    cmd = [
        str(whisper_bin),
        str(wav_path),
        "--language",    WHISPER_LANGUAGE,
        "--model",       WHISPER_MODEL,
        "--device",      WHISPER_DEVICE,
        "--output_dir",  str(output_subdir),
        "--output_format", "all",
    ]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("ERRO durante a transcrição.")
        sys.exit(1)

    print(f"\nTranscrição concluída! Arquivos salvos em:\n  {output_subdir}")
    return output_subdir


# ─── Limpeza ──────────────────────────────────────────────────────────────────

def cleanup(wav_path: Path) -> None:
    if wav_path.exists():
        wav_path.unlink()
        print(f"Arquivo temporário removido: {wav_path.name}")


# ─── Ponto de Entrada ─────────────────────────────────────────────────────────

def main() -> None:
    setup_venv()
    check_ffmpeg()

    while True:
        files = list_video_files()

        if not files:
            print(f"Nenhum arquivo de vídeo encontrado em:\n  {SCRIPT_DIR}")
            break

        selected = select_file_curses(files)

        if selected is None:
            print("Operação cancelada.")
            break

        print(f"\nArquivo selecionado: {selected.name}")

        wav_path = convert_to_wav(selected)
        transcribe(wav_path, selected.name)
        cleanup(wav_path)

        resp = input("\nDeseja transcrever outro arquivo? (s/N): ").strip().lower()
        if resp != "s":
            break

    print("\nFinalizado.")


if __name__ == "__main__":
    main()
