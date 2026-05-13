# PAGE XML Schema Compliance

The exporter targets PAGE XML namespace `http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15`.

Each file contains:

- `PcGts` root with schema location.
- `Metadata/Creator`.
- One `Page` with image filename, dimensions, and right-to-left reading direction.
- One v0.1 `TextRegion` covering the page text body.
- One `TextLine` per consensus line.
- `Coords` and `Baseline` in page pixel coordinates.
- `TextEquiv index="1" dataType="raw"` for exact raw consensus text.
- `TextEquiv index="2" dataType="normalised"` for Sorani-normalised text.

PAGE custom data is stored in the standard `custom` attribute, for example `confidence_label:near_agreement;engine_consensus_count:3`, so strict validators do not reject unknown attributes.

Use:

```powershell
python -m pipeline --config config.yaml.example pagexml --limit 1
```

Programmatic validation is available through `pipeline.pagexml_export.validate_page_xml(path, xsd_path)`.
