"""Curses-based file selector."""

from __future__ import annotations

import curses
from pathlib import Path


def select_files(files: list[Path]) -> list[Path]:
    def draw(stdscr, selected: int, scroll: int, checked: set[int]) -> None:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        usable_width = max(1, width - 1)
        title = " Transcrição com Whisper — Selecione o vídeo "
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        stdscr.addnstr(0, 0, title.center(usable_width), usable_width)
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
        stdscr.addnstr(1, 0, "─" * usable_width, usable_width)

        visible_rows = max(1, height - 5)
        for index in range(min(visible_rows, len(files) - scroll)):
            real_index = index + scroll
            row = index + 2
            mark = "[x]" if real_index in checked else "[ ]"
            prefix = f" ► {mark} " if real_index == selected else f"   {mark} "
            line = (prefix + files[real_index].name).ljust(usable_width)
            if real_index == selected:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addnstr(row, 0, line, usable_width)
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addnstr(row, 0, line, usable_width)

        if height >= 2:
            help_line = " ↑↓/jk: navegar   Espaço: marcar   a: todos   Enter: confirmar "
            counter = f" {len(checked)} marcados | {selected + 1}/{len(files)} "
            stdscr.addnstr(height - 2, 0, "─" * usable_width, usable_width)
            stdscr.addnstr(height - 1, 0, help_line + counter, usable_width)
        stdscr.refresh()

    def run(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
        selected = 0
        scroll = 0
        checked: set[int] = set()
        while True:
            height, _ = stdscr.getmaxyx()
            visible_rows = max(1, height - 5)
            draw(stdscr, selected, scroll, checked)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")) and selected > 0:
                selected -= 1
                scroll = min(scroll, selected)
            elif key in (curses.KEY_DOWN, ord("j")) and selected < len(files) - 1:
                selected += 1
                if selected >= scroll + visible_rows:
                    scroll = selected - visible_rows + 1
            elif key in (curses.KEY_ENTER, 10, 13):
                indices = sorted(checked) if checked else [selected]
                return [files[index] for index in indices]
            elif key == ord(" "):
                if selected in checked:
                    checked.remove(selected)
                else:
                    checked.add(selected)
            elif key in (ord("a"), ord("A")):
                checked = set() if len(checked) == len(files) else set(range(len(files)))
            elif key in (ord("q"), ord("Q"), 27):
                return []

    return curses.wrapper(run)


def select_file(files: list[Path]) -> Path | None:
    """Backward-compatible single-file selector."""
    selected = select_files(files)
    return selected[0] if selected else None
