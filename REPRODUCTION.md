# Reproduction notes

## Imported source

The original ZIP was preserved at the repository root. Its source files were
copied without algorithm changes into:

- `src/serial`
- `src/parallel`

All 24 extracted source files were checked against their ZIP entries with
SHA-256; there were no mismatches.

## Verified local environment

- Windows-native Codex environment with PowerShell
- Visual Studio Community 2026
- MSVC toolset 14.44 for CUDA host compilation
- CUDA Toolkit 12.8
- OpenMP runtime supplied by Visual Studio
- NVIDIA GeForce RTX 5060 Laptop GPU

The programs read the original dataset from the hard-coded path:

`D:\llm_learn_and_lab_demo_1\bio_lab\cpp_data`

## Build

From the `project` directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

This produces:

- `build/serial/serial_pipeline.exe`
- `build/parallel/parallel_pipeline.exe`

## Stage validation

Validation is optional and does not change a normal run. Use `--validate` to
print deterministic fingerprints for each completed stage, and use
`--stop-after` to exit before later stages run.

For example, these commands compare both pipelines through sparse-to-dense
conversion without starting PCA:

```powershell
.\build\serial\serial_pipeline.exe --validate --stop-after dense
.\build\parallel\parallel_pipeline.exe --validate --stop-after dense
```

Accepted stop stages are `load`, `hvg`, `dense`, `pca-fit`, `pca-transform`,
and `kmeans`. Because floating-point CPU and GPU implementations may differ by
rounding, a different PCA fingerprint identifies a checkpoint that needs a
tolerance-based numerical comparison; it does not by itself prove the result
is incorrect.

The build succeeds with the original source, but currently reports these
warnings:

- three `printf("%.6f", ...)` calls receive a `std::chrono::duration` object
  instead of a `double` value;
- one local variable in `pca_cuda_optim.cu` is unused;
- CUDA 12.8 warns that support for older offline GPU architectures will be
  removed in a future release.

## Baseline run on 2026-10-07

Both original pipelines compiled and completed successfully on the current
machine.

| Stage | Serial | Parallel |
| --- | ---: | ---: |
| Load sparse matrix | 8.295901 s | 1.385822 s |
| Select HVGs | 0.357953 s | 0.046145 s |
| Sparse to dense | 2.068543 s | 1.959524 s |
| PCA fit | 951.689234 s | 34.569825 s |
| PCA transform | 29.500786 s | 1.409285 s |
| K-Means | 1.304173 s | 0.121284 s |
| Complete pipeline | 993.230097 s | 39.500131 s |

The measured end-to-end speedup was approximately 25.15x.

Both runs loaded a 164,978 x 18,000 sparse matrix with 176,609,566 nonzero
values, selected 5,000 genes, and reported gene 14194 as the highest-variance
gene.

The parallel run printed these first 20 cluster labels:

`7 5 5 3 8 8 1 9 2 3 8 9 8 4 7 8 9 2 7 0`

The original serial entry point does not print or save its labels, so numerical
output equivalence has not yet been established. That requires an explicit,
shared output/validation path for both executables.
