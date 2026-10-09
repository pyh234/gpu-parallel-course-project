param(
    [string]$MsvcToolset = "14.44",
    [string]$CudaVersion = "12.8"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$serialSource = Join-Path $projectRoot "src\serial"
$parallelSource = Join-Path $projectRoot "src\parallel"
$serialBuild = Join-Path $projectRoot "build\serial"
$parallelBuild = Join-Path $projectRoot "build\parallel"

New-Item -ItemType Directory -Path $serialBuild -Force | Out-Null
New-Item -ItemType Directory -Path $parallelBuild -Force | Out-Null

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio with the Desktop development with C++ workload."
}

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$visualStudio = & $vswhere -latest -products * -property installationPath -utf8
if (-not $visualStudio) {
    throw "No Visual Studio installation was found."
}

$vcvars = Join-Path $visualStudio "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path -LiteralPath $vcvars)) {
    throw "Visual Studio C++ environment script was not found: $vcvars"
}

$nvcc = Join-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CudaVersion" "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvcc)) {
    throw "CUDA compiler was not found: $nvcc"
}

$serialCommand = @(
    "call `"$vcvars`" -vcvars_ver=$MsvcToolset >nul",
    "&& cl /nologo /EHsc /std:c++17 /O2",
    "`"$serialSource\main.cpp`"",
    "`"$serialSource\data_loader.cpp`"",
    "`"$serialSource\dense_matrix.cpp`"",
    "`"$serialSource\hvg_select.cpp`"",
    "`"$serialSource\pca.cpp`"",
    "`"$serialSource\kmeans.cpp`"",
    "/Fe:serial_pipeline.exe"
) -join " "

Push-Location $serialBuild
try {
    & $env:ComSpec /d /c $serialCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Serial build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$parallelCommand = @(
    "call `"$vcvars`" -vcvars_ver=$MsvcToolset >nul",
    "&& `"$nvcc`" -std=c++17 -O2 -Xcompiler /openmp -Xcompiler /EHsc",
    "`"$parallelSource\parallel_optimize_version.cpp`"",
    "`"$parallelSource\data_loader_omp.cpp`"",
    "`"$parallelSource\dense_matrix_optim.cpp`"",
    "`"$parallelSource\hvg_select_optim.cpp`"",
    "`"$parallelSource\pca_cuda_optim.cu`"",
    "`"$parallelSource\kmeans_optim.cu`"",
    "-o parallel_pipeline.exe"
) -join " "

Push-Location $parallelBuild
try {
    & $env:ComSpec /d /c $parallelCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Parallel build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Build completed:"
Write-Host "  $serialBuild\serial_pipeline.exe"
Write-Host "  $parallelBuild\parallel_pipeline.exe"
