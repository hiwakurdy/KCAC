from __future__ import annotations

import unicodedata
from typing import Literal, cast

from .models import NormalisationChange, NormalisedText

ARABIC_YEH = "\u064a"
FARSI_YEH = "\u06cc"
ALEF_MAKSURA = "\u0649"
ARABIC_KAF = "\u0643"
KURDISH_KAF = "\u06a9"

MAPPINGS: dict[str, tuple[str, str]] = {
    ARABIC_YEH: (FARSI_YEH, "arabic_yeh_to_kurdish_yeh"),
    ALEF_MAKSURA: (FARSI_YEH, "alef_maksura_to_kurdish_yeh"),
    ARABIC_KAF: (KURDISH_KAF, "arabic_kaf_to_kurdish_kaf"),
}

TASHKEEL_RANGES = (
    (0x0610, 0x061A),
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06ED),
)


def is_tashkeel(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in TASHKEEL_RANGES)


def normalise_sorani(text: str, *, strip_tashkeel: bool = True, unicode_form: str = "NFC") -> NormalisedText:
    if unicode_form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"Unsupported unicode normalisation form: {unicode_form}")
    form = cast(Literal["NFC", "NFD", "NFKC", "NFKD"], unicode_form)
    changes: list[NormalisationChange] = []
    output: list[str] = []
    for index, char in enumerate(text):
        if strip_tashkeel and is_tashkeel(char):
            changes.append(NormalisationChange(index=index, before=char, after="", reason="strip_tashkeel"))
            continue
        mapped = MAPPINGS.get(char)
        if mapped:
            after, reason = mapped
            output.append(after)
            changes.append(NormalisationChange(index=index, before=char, after=after, reason=reason))
        else:
            output.append(char)

    pre_unicode = "".join(output)
    normalised = unicodedata.normalize(form, pre_unicode)
    if normalised != pre_unicode:
        changes.append(
            NormalisationChange(
                index=-1,
                before=pre_unicode,
                after=normalised,
                reason=f"unicode_{unicode_form}",
            )
        )
    return NormalisedText(raw=text, normalised=normalised, changes=changes)


def strip_diacritics(text: str) -> str:
    return "".join(char for char in text if not is_tashkeel(char))
