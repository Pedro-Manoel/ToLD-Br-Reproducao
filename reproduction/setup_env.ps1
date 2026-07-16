# Cria o venv FORA do OneDrive (evita sincronizar milhares de arquivos de pacotes)
# e instala as dependencias. Pode ser rodado de qualquer diretorio.
#
# Uso:
#   .\setup_env.ps1
#   .\setup_env.ps1 -Python "py -3.12"   # fallback se houver problema de wheel no 3.13
param(
    [string]$Python = "python",
    [string]$VenvDir = "$env:LOCALAPPDATA\toldbr-repro-venv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Criando venv em $VenvDir ..."
    & $Python -m venv $VenvDir
}
$Py = Join-Path $VenvDir "Scripts\python.exe"

& $Py -m pip install --upgrade pip
# Torch com CUDA (cu124 cobre a RTX 4070 Ti SUPER)
& $Py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
& $Py -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "----------------------------------------------------"
Write-Host "VENV pronto em: $VenvDir"
Write-Host "Python do venv: $Py"
& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
