# Classic Thesis Layout (matches previous-year structure)

This directory mirrors your earlier thesis template using `Thesis.cls` and `Chapters/` files.

- Class: `Thesis.cls`
- Entry: `main.tex`
- Bibliography: `Nomzui.bib`
- Chapters dir: `Chapters/` (CoverPage, TitlePage, Abstract/Abstrct, plan, Talarhal, acrolist, appendix)

## Build

```pwsh
cd D:\Tugsultiin_Ajil\thesis_classic
..\thesis\build.ps1
```

This reuses the robust build script from `thesis/` (auto-detects xelatex/latexmk and falls back when needed).

## Notes

- Uncomment the includes in `main.tex` for the pages you need (Cover, Title, Abstract, etc.).
- Add your content under `Chapters/` following the previous structure.
- If MiKTeX prompts to install missing packages, allow it.
