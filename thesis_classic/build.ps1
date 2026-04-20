#!/usr/bin/env pwsh
# Build thesis_classic locally with latexmk if available; otherwise fall back to xelatex+bibtex.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-Exe {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [string[]]$Hints
  )
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  foreach ($h in $Hints) { if (Test-Path $h) { return $h } }
  return $null
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][scriptblock]$Command
  )
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Name exited with code $LASTEXITCODE"
  }
}

function Invoke-XeLaTeXChecked {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][scriptblock]$Command,
    [Parameter(Mandatory=$true)][string]$OutDir
  )
  & $Command
  if ($LASTEXITCODE -eq 0) { return }

  $mainLog = Join-Path $OutDir 'main.log'
  if (Test-Path $mainLog) {
    $log = Get-Content $mainLog -Raw
    $hasPdfOutput = $log -match 'Output written on .*main\.pdf'
    $hasFatal = $log -match '(?m)^!|Emergency stop|Fatal error'
    if ($hasPdfOutput -and -not $hasFatal) {
      Write-Warning "$Name exited with code $LASTEXITCODE, but PDF was generated; continuing."
      return
    }
  }

  throw "$Name exited with code $LASTEXITCODE"
}

Push-Location $here
try {
  $outDir = $here
  if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

  $miKTeXBin = Join-Path $env:LOCALAPPDATA 'Programs/MiKTeX/miktex/bin/x64'
  $programFilesBin = Join-Path $env:ProgramFiles 'MiKTeX/miktex/bin/x64'
  $programFilesX86Bin = Join-Path ${env:ProgramFiles(x86)} 'MiKTeX/miktex/bin/x64'
  $xelatexExe = Find-Exe -Name 'xelatex' -Hints @(
    (Join-Path $miKTeXBin 'xelatex.exe'),
    (Join-Path $programFilesBin 'xelatex.exe'),
    (Join-Path $programFilesX86Bin 'xelatex.exe'),
    'C:\texlive\2023\bin\win32\xelatex.exe'
  )
  $biberExe = Find-Exe -Name 'biber' -Hints @(
    (Join-Path $miKTeXBin 'biber.exe'),
    (Join-Path $programFilesBin 'biber.exe'),
    (Join-Path $programFilesX86Bin 'biber.exe'),
    'C:\texlive\2023\bin\win32\biber.exe'
  )
  $bibtexExe = Find-Exe -Name 'bibtex' -Hints @(
    (Join-Path $miKTeXBin 'bibtex.exe'),
    (Join-Path $programFilesBin 'bibtex.exe'),
    (Join-Path $programFilesX86Bin 'bibtex.exe'),
    'C:\texlive\2023\bin\win32\bibtex.exe'
  )

  # Detect bibliography backend from main.tex
  $mainTexPath = Join-Path $PSScriptRoot 'main.tex'
  $backend = 'bibtex'
  if (Test-Path $mainTexPath) {
    $texContent = Get-Content $mainTexPath -Raw
    if ($texContent -match 'backend\s*=\s*bibtex') { $backend = 'bibtex' }
  }
  $latexmkExe = Find-Exe -Name 'latexmk' -Hints @(
    (Join-Path $miKTeXBin 'latexmk.exe')
  )
  $perlExe = Find-Exe -Name 'perl' -Hints @(
    'C:\Strawberry\perl\bin\perl.exe',
    'C:\Perl64\bin\perl.exe',
    'C:\Perl\bin\perl.exe'
  )
  $canUseLatexmk = $latexmkExe -and $perlExe

  if ($canUseLatexmk) {
    try {
      & $latexmkExe -xelatex -interaction=nonstopmode -file-line-error -outdir:"$outDir" main.tex
      if ($LASTEXITCODE -ne 0) {
        throw "latexmk exited with code $LASTEXITCODE"
      }
    } catch {
      Write-Warning "latexmk failed; falling back to xelatex/bibtex sequence."
      if (-not $xelatexExe) { throw "xelatex not found. Please ensure MiKTeX is installed and in PATH." }
      Invoke-XeLaTeXChecked -Name 'xelatex (pass 1)' -Command { & $xelatexExe -interaction=nonstopmode -file-line-error -output-directory:"$outDir" main.tex } -OutDir $outDir
      if ($backend -eq 'bibtex') {
        if ($bibtexExe) {
          Push-Location $outDir
          try {
            Invoke-Checked -Name 'bibtex' -Command { & $bibtexExe main }
          } finally {
            Pop-Location
          }
        }
        else { Write-Warning "BibTeX backend requested but bibtex.exe not found." }
      } else {
        if ($biberExe) { Invoke-Checked -Name 'biber' -Command { & $biberExe (Join-Path $outDir 'main') } }
        else { Write-Warning "Biber backend requested but biber.exe not found." }
      }
      Invoke-XeLaTeXChecked -Name 'xelatex (pass 2)' -Command { & $xelatexExe -interaction=nonstopmode -file-line-error -output-directory:"$outDir" main.tex } -OutDir $outDir
      Invoke-XeLaTeXChecked -Name 'xelatex (pass 3)' -Command { & $xelatexExe -interaction=nonstopmode -file-line-error -output-directory:"$outDir" main.tex } -OutDir $outDir
    }
  } else {
    if ($latexmkExe -and -not $perlExe) {
      Write-Warning "latexmk found, but Perl is not available; using xelatex/bibtex sequence."
    } else {
      Write-Warning "latexmk not found; using xelatex/bibtex sequence."
    }
    if (-not $xelatexExe) {
      Write-Error "xelatex not found. Tried PATH and: '$miKTeXBin', '$programFilesBin', '$programFilesX86Bin', 'C:\\texlive\\2023\\bin\\win32'. Please install MiKTeX or TeX Live, or provide the executable path."
      exit 1
    }
    Invoke-XeLaTeXChecked -Name 'xelatex (pass 1)' -Command { & $xelatexExe -interaction=nonstopmode -file-line-error -output-directory:"$outDir" main.tex } -OutDir $outDir
    if ($backend -eq 'bibtex') {
      if ($bibtexExe) {
        Push-Location $outDir
        try {
          Invoke-Checked -Name 'bibtex' -Command { & $bibtexExe main }
        } finally {
          Pop-Location
        }
      }
      else { Write-Warning "BibTeX backend requested but bibtex.exe not found." }
    } else {
      if ($biberExe) { Invoke-Checked -Name 'biber' -Command { & $biberExe (Join-Path $outDir 'main') } }
      else { Write-Warning "Biber backend requested but biber.exe not found." }
    }
    Invoke-XeLaTeXChecked -Name 'xelatex (pass 2)' -Command { & $xelatexExe -interaction=nonstopmode -file-line-error -output-directory:"$outDir" main.tex } -OutDir $outDir
    Invoke-XeLaTeXChecked -Name 'xelatex (pass 3)' -Command { & $xelatexExe -interaction=nonstopmode -file-line-error -output-directory:"$outDir" main.tex } -OutDir $outDir
  }
  Write-Host "Build finished. PDF: $(Join-Path $outDir 'main.pdf')"
} finally {
  Pop-Location
}
