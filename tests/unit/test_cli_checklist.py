from unittest.mock import MagicMock, patch

from omnimux.utils.cli_checklist import run_checklist


def test_checklist_empty() -> None:
    assert run_checklist([]) == []


def test_checklist_non_interactive() -> None:
    with patch("sys.stdin.isatty", return_value=False):
        res = run_checklist([("m1", "Model 1"), ("m2", "Model 2")])
        assert res == ["m1", "m2"]


def test_checklist_interactive_confirm() -> None:
    # Simulates pressing Enter
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdin.fileno.return_value = 0
    mock_stdin.read.side_effect = ["\n"]

    with (
        patch("sys.stdin", mock_stdin),
        patch("termios.tcgetattr", return_value=[]),
        patch("termios.tcsetattr"),
        patch("tty.setraw"),
    ):
        res = run_checklist([("m1", "Model 1"), ("m2", "Model 2")])
        assert res == ["m1", "m2"]


def test_checklist_interactive_toggle_and_arrows() -> None:
    # Keys: space (toggle 0), down arrow, space (toggle 1), up arrow, Enter
    key_sequence = [
        " ",  # uncheck m1
        "\x1b",  # arrow prefix
        "[B",  # down arrow
        " ",  # uncheck m2
        "\x1b",  # arrow prefix
        "[A",  # up arrow
        " ",  # check m1
        "\r",  # confirm
    ]
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdin.fileno.return_value = 0
    mock_stdin.read.side_effect = key_sequence

    with (
        patch("sys.stdin", mock_stdin),
        patch("termios.tcgetattr", return_value=[]),
        patch("termios.tcsetattr"),
        patch("tty.setraw"),
    ):
        res = run_checklist([("m1", "Model 1"), ("m2", "Model 2")])
        assert res == ["m1"]


def test_checklist_interactive_toggle_all_and_jk() -> None:
    # Keys: 'a' (toggle all to off), 'k' (up), 'j' (down), 'a' (toggle all to on), Enter
    key_sequence = ["a", "k", "j", "a", "\n"]
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdin.fileno.return_value = 0
    mock_stdin.read.side_effect = key_sequence

    with (
        patch("sys.stdin", mock_stdin),
        patch("termios.tcgetattr", return_value=[]),
        patch("termios.tcsetattr"),
        patch("tty.setraw"),
    ):
        res = run_checklist([("m1", "Model 1"), ("m2", "Model 2")])
        assert res == ["m1", "m2"]


def test_checklist_interactive_cancel() -> None:
    # Keys: 'q' to cancel
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdin.fileno.return_value = 0
    mock_stdin.read.side_effect = ["q"]

    with (
        patch("sys.stdin", mock_stdin),
        patch("termios.tcgetattr", return_value=[]),
        patch("termios.tcsetattr"),
        patch("tty.setraw"),
    ):
        res = run_checklist([("m1", "Model 1")])
        assert res == []


def test_checklist_interactive_standalone_escape() -> None:
    # Keys: '\x1b' with empty select
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdin.fileno.return_value = 0
    mock_stdin.read.side_effect = ["\x1b"]

    with (
        patch("sys.stdin", mock_stdin),
        patch("select.select", return_value=([], [], [])),
        patch("termios.tcgetattr", return_value=[]),
        patch("termios.tcsetattr"),
        patch("tty.setraw"),
    ):
        res = run_checklist([("m1", "Model 1")])
        assert res == []


def test_checklist_interactive_escape_select_exception() -> None:
    # Keys: '\x1b' then '[A' (up arrow) when select raises exception
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = True
    mock_stdin.fileno.return_value = 0
    mock_stdin.read.side_effect = ["\x1b", "[A", "\n"]

    with (
        patch("sys.stdin", mock_stdin),
        patch("select.select", side_effect=OSError("Not supported")),
        patch("termios.tcgetattr", return_value=[]),
        patch("termios.tcsetattr"),
        patch("tty.setraw"),
    ):
        res = run_checklist([("m1", "Model 1")])
        assert res == ["m1"]
