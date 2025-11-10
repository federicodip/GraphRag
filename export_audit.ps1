# export_audit.ps1
# Writes a directory overview + selected file contents to audit_clean.txt

$OutFile   = "audit_clean.txt"
$MaxBytes  = 200KB          # change if you want a different cap
$WrapWidth = 160

# Folders to ignore anywhere in the tree
$excludeDirPattern = '\\(\.git|node_modules|__pycache__|\.venv|\.ipynb_checkpoints|logs|import|plugins)(\\|$)'

# Specific big-data subtrees you want ignored completely (add more if needed)
$excludeRoots = @(
  '\data\pleiades',      # your large Pleiades dump
  '\data\raw',
  '\data\cache'
)

# File types to include in the "contents" section
$includeExt = @(
  '.ps1','.psm1','.bat','.cmd',
  '.py','.php','.js','.ts',
  '.json','.md','.txt',
  '.html','.htm',
  '.yml','.yaml','.ini','.cfg'
)

# --- helpers ---------------------------------------------------------------

function Add-Line([string]$s='') {
  $s | Out-File -Encoding utf8 -FilePath $OutFile -Append
}

function Is-ExcludedPath([string]$fullPath) {
  if ($fullPath -match $excludeDirPattern) { return $true }
  foreach ($root in $excludeRoots) {
    if ($fullPath.ToLower().Contains($root.ToLower())) { return $true }
  }
  return $false
}

function RelPath([string]$fullPath) {
  $root = (Get-Location).Path
  if ($fullPath.StartsWith($root)) { return ".\" + $fullPath.Substring($root.Length + 1) }
  return $fullPath
}

function Wrap-Line([string]$line, [int]$width) {
  if ([string]::IsNullOrEmpty($line)) { return @('') }
  # soft wrap on whitespace up to $width
  return [regex]::Matches($line, "(.{1,$width})(\s+|$)") | ForEach-Object { $_.Groups[1].Value }
}

function Dump-File([IO.FileInfo]$f) {
  $rel = RelPath $f.FullName
  if ($f.Length -gt $MaxBytes) {
    Add-Line ""
    Add-Line "----- SKIP (too large) $rel ($($f.Length) bytes) -----"
    return
  }

  Add-Line ""
  Add-Line "----- BEGIN $rel -----"

  $text =
    if ($f.Extension -in @('.html','.htm')) {
      # strip giant data URIs; keep text readable
      (Get-Content -Raw -Encoding UTF8 $f.FullName) `
        -replace 'data:(?:image|font|application)/[a-zA-Z0-9.\+\-]+;base64,[A-Za-z0-9+/=\s]+' ,'[[data-uri-stripped]]'
    } else {
      Get-Content -Raw -Encoding UTF8 $f.FullName
    }

  $i = 1
  foreach ($ln in ($text -split "`r?`n")) {
    foreach ($w in (Wrap-Line $ln $WrapWidth)) {
      Add-Line ("{0,6}  {1}" -f $i, $w)
    }
    $i++
  }

  Add-Line "----- END $rel -----"
}

# --- start fresh -----------------------------------------------------------
Remove-Item -ErrorAction Ignore $OutFile

# --- Directory overview ----------------------------------------------------
Add-Line "=== Directory Structure Overview ==="
# build a simple tree (directories first, then files), excluding junk
$items = Get-ChildItem -Recurse -Force | Where-Object {
  -not (Is-ExcludedPath $_.FullName)
}

# group by directory depth for pretty printing
$rootPath   = (Get-Location).Path
$itemsDirs  = $items | Where-Object { $_.PSIsContainer } | Sort-Object FullName
$itemsFiles = $items | Where-Object { -not $_.PSIsContainer } | Sort-Object FullName

# print directories
foreach ($d in $itemsDirs) {
  $rel = RelPath $d.FullName
  $depth = ($rel -split '[\\/]').Count - 1
  $indent = ('|   ' * ($depth - 1)) + (if ($depth -gt 0) {'|-- '} else {''})
  Add-Line ($indent + $rel)
}
# print files
foreach ($f in $itemsFiles) {
  $rel = RelPath $f.FullName
  $depth = ($rel -split '[\\/]').Count - 1
  $indent = ('|   ' * ($depth - 1)) + (if ($depth -gt 0) {'|-- '} else {''})
  Add-Line ($indent + $rel)
}

# --- File contents ---------------------------------------------------------
Add-Line ""
Add-Line "=== File Contents (filtered, readable) ==="

$filesToDump = Get-ChildItem -Recurse -File -Force |
  Where-Object {
    -not (Is-ExcludedPath $_.FullName) -and
    ($includeExt -contains $_.Extension.ToLower())
  } |
  Sort-Object FullName

foreach ($f in $filesToDump) {
  Dump-File $f
}

Add-Line ""
Add-Line "Wrote: $OutFile"
Write-Host "Wrote: $OutFile"
