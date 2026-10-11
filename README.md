# 基于 OpenMP 和 CUDA 的大规模空间转录组数据分析流程并行优化

本项目为 GPU 并行计算课程大作业，使用 C++ 实现空间转录组数据的串行处理流程，并使用 OpenMP 与 CUDA 对数据加载、基因筛选、矩阵转换、PCA 降维和 K-Means 聚类等步骤进行并行优化。

## 1. 项目结构

```text
.
├── README.md
├── reports/                 课程实验报告
├── src/
│   ├── serial/              串行版本源代码
│   ├── parallel/            OpenMP 与 CUDA 并行版本源代码
│   └── pipeline_validation.h  分阶段验证工具
├── scripts/
│   └── build.ps1            Windows 编译脚本
└── build/                   编译输出目录，不纳入 Git 管理
```

## 2. 运行环境

- Windows 11
- Visual Studio，并安装“使用 C++ 的桌面开发”工作负载
- MSVC 14.44
- CUDA Toolkit 12.8
- 支持 CUDA 的 NVIDIA GPU

编译脚本默认使用 MSVC 14.44 和 CUDA 12.8。如本机版本不同，可通过脚本参数进行调整。

## 3. 数据准备

程序默认从以下目录读取数据：

```text
D:/llm_learn_and_lab_demo_1/bio_lab/cpp_data/
```

该目录需要包含 `metadata.txt`、`indptr.bin`、`indices.bin` 和 `values.bin`。如需使用其他目录，请修改串行与并行数据加载代码中的 `data_dir`，以及主程序中的 `output_dir`。

## 4. 编译方式

在项目目录打开 PowerShell，运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

编译完成后生成：

```text
build/serial/serial_pipeline.exe
build/parallel/parallel_pipeline.exe
```

## 5. 运行方式

```powershell
.\build\serial\serial_pipeline.exe
.\build\parallel\parallel_pipeline.exe
```

程序支持使用 `--validate` 输出各处理阶段的验证信息，并使用 `--stop-after` 在指定阶段结束运行。例如：

```powershell
.\build\serial\serial_pipeline.exe --validate --stop-after dense
.\build\parallel\parallel_pipeline.exe --validate --stop-after dense
```

可选阶段包括 `load`、`hvg`、`dense`、`pca-fit`、`pca-transform` 和 `kmeans`。
