"""Tests for ADB shell injection prevention and input validation."""

import pytest

from src.device.adb import _escape_shell_text, _KEYCODE_RE, _PACKAGE_RE


def test_escape_normal_text():
    assert _escape_shell_text("hello") == "'hello'"


def test_escape_single_quote():
    result = _escape_shell_text("it's")
    assert "'" in result
    assert "\\'" in result


def test_escape_shell_injection():
    dangerous = "'; rm -rf /; echo '"
    result = _escape_shell_text(dangerous)
    # Should be safely quoted, not executable
    assert "rm -rf" in result
    assert result.startswith("'")
    assert result.endswith("'")


def test_escape_dollar_sign():
    result = _escape_shell_text("price is $99")
    assert "$99" in result


def test_escape_backticks():
    result = _escape_shell_text("`whoami`")
    assert "`whoami`" in result


def test_escape_chinese():
    result = _escape_shell_text("你好世界")
    assert "你好世界" in result


def test_escape_empty():
    assert _escape_shell_text("") == "''"


def test_keycode_valid():
    assert _KEYCODE_RE.match("KEYCODE_BACK")
    assert _KEYCODE_RE.match("KEYCODE_HOME")
    assert _KEYCODE_RE.match("KEYCODE_ENTER")
    assert _KEYCODE_RE.match("4")  # numeric keycode


def test_keycode_invalid():
    assert not _KEYCODE_RE.match("KEYCODE_BACK; rm -rf /")
    assert not _KEYCODE_RE.match("key code")
    assert not _KEYCODE_RE.match("$(whoami)")


def test_package_valid():
    assert _PACKAGE_RE.match("com.tencent.mm")
    assert _PACKAGE_RE.match("com.android.settings")
    assert _PACKAGE_RE.match("org.mozilla.firefox")


def test_package_invalid():
    assert not _PACKAGE_RE.match("com.evil; rm -rf /")
    assert not _PACKAGE_RE.match("com evil app")
    assert not _PACKAGE_RE.match("$(whoami)")
