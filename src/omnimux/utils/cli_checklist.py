import select
import sys
from collections.abc import Sequence


def run_checklist(
    options: Sequence[tuple[str, str]],
    prompt: str = "Select models to enable (Space to toggle, 'a' for all, Enter to confirm):",
) -> list[str]:
    """Interactive multi-select checklist in the terminal using standard termios."""
    if not options:
        return []

    # Fallback for non-interactive environments (CI, pipes)
    if not sys.stdin.isatty():
        return [opt[0] for opt in options]

    import termios
    import tty

    selected = {i: True for i in range(len(options))}
    cursor = 0
    total = len(options)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def draw(first_time: bool = False) -> None:
        lines_to_clear = total + 1 if not first_time else 0
        if lines_to_clear > 0:
            sys.stdout.write(f"\033[{lines_to_clear}A")

        sys.stdout.write(f"\033[1m{prompt}\033[0m\n")
        for i, (opt_id, label) in enumerate(options):
            check = "[*]" if selected.get(i) else "[ ]"
            pointer = " > " if i == cursor else "   "
            color = "\033[36m" if i == cursor else "\033[0m"
            sys.stdout.write(f"\033[2K{color}{pointer}{check} {label} ({opt_id})\033[0m\n")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        draw(first_time=True)

        while True:
            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":
                break
            elif ch == " ":
                selected[cursor] = not selected.get(cursor, False)
            elif ch == "a" or ch == "A":
                all_selected = all(selected.get(i, False) for i in range(total))
                new_state = not all_selected
                for i in range(total):
                    selected[i] = new_state
            elif ch == "k":
                cursor = (cursor - 1) % total
            elif ch == "j":
                cursor = (cursor + 1) % total
            elif ch == "\x1b":  # Escape sequence for arrow keys or standalone Escape
                try:
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                except Exception:
                    r = [sys.stdin]
                if r:
                    seq = sys.stdin.read(2)
                    if seq == "[A":  # Up
                        cursor = (cursor - 1) % total
                    elif seq == "[B":  # Down
                        cursor = (cursor + 1) % total
                else:
                    sys.stdout.write("\nCancelled.\n")
                    return []
            elif ch == "\x03" or ch == "q":  # Ctrl+C or q
                sys.stdout.write("\nCancelled.\n")
                return []

            draw()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\n")
        sys.stdout.flush()

    return [options[i][0] for i in range(total) if selected.get(i)]
