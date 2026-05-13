from __future__ import annotations

from pipeline.normalise import normalise_sorani


def test_sorani_core_mappings_and_trace() -> None:
    result = normalise_sorani("\u0643\u064a\u0649")
    assert result.normalised == "\u06a9\u06cc\u06cc"
    assert [change.reason for change in result.changes] == [
        "arabic_kaf_to_kurdish_kaf",
        "arabic_yeh_to_kurdish_yeh",
        "alef_maksura_to_kurdish_yeh",
    ]


def test_tashkeel_stripping_preserves_raw() -> None:
    result = normalise_sorani("\u0643\u064e")
    assert result.raw == "\u0643\u064e"
    assert result.normalised == "\u06a9"
    assert result.changes[-1].reason == "strip_tashkeel"
