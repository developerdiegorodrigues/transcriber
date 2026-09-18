"""Curses-based file selector."""

from __future__ import annotations

import curses
from pathlib import Path


def select_file(files: list[Path]) -> Path | None:
    def draw(stdscr, selected: int, scroll: int) -> None:
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
            prefix = " ► " if real_index == selected else "   "
            line = (prefix + files[real_index].name).ljust(usable_width)
            if real_index == selected:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addnstr(row, 0, line, usable_width)
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addnstr(row, 0, line, usable_width)

        if height >= 2:
            help_line = " ↑↓/jk: navegar   Enter: selecionar   q/Esc: sair "
            counter = f" {selected + 1}/{len(files)} "
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
        while True:
            height, _ = stdscr.getmaxyx()
            visible_rows = max(1, height - 5)
            draw(stdscr, selected, scroll)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")) and selected > 0:
                selected -= 1
                scroll = min(scroll, selected)
            elif key in (curses.KEY_DOWN, ord("j")) and selected < len(files) - 1:
                selected += 1
                if selected >= scroll + visible_rows:
                    scroll = selected - visible_rows + 1
            elif key in (curses.KEY_ENTER, 10, 13):
                return files[selected]
            elif key in (ord("q"), ord("Q"), 27):
                return None

    return curses.wrapper(run)
