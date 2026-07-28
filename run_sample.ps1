$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$root\.venv\Scripts\goalkeeper-highlights.exe" `
  "C:\videorohdaten\158_0726\FCWittlinge-SFETeil1.MP4" `
  --frame-stride 2 `
  --max-candidates 10 `
  --overwrite
