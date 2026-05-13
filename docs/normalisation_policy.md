# Sorani Unicode Normalisation Policy

The raw OCR layer is never rewritten. Every line keeps raw text and receives a separate normalised text value plus a trace of each change.

Mappings in v0.1:

| From | To | Reason |
|---|---|---|
| U+064A Arabic Yeh | U+06CC Kurdish/Persian Yeh | Modern Sorani orthography |
| U+0649 Alef Maksura | U+06CC Kurdish/Persian Yeh | OCR often confuses final yeh forms |
| U+0643 Arabic Kaf | U+06A9 Kurdish/Persian Kaf | Modern Sorani orthography |
| Arabic tashkeel ranges | removed | Optional marks are stripped for the normalised layer |

After character-level mappings, text is normalised with NFC by default. The trace log records character index, before value, after value, and reason. Unicode recomposition is recorded as a whole-line change when it changes the string.
