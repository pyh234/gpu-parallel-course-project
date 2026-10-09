#include "data_loader.h"
#include "sparse_matrix.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <omp.h> 

SparseMatrix loadFromBinary() {
    SparseMatrix mat;

    std::ifstream meta("D:/llm_learn_and_lab_demo_1/bio_lab/cpp_data/metadata.txt");
    if (!meta.is_open()) {
        std::cerr << "Error: Cannot open metadata.txt" << std::endl;
        exit(1);
    }
    meta >> mat.n_rows >> mat.n_cols;
    meta.close();

    mat.indptr.resize(mat.n_rows + 1);

    std::ifstream indptr_file("D:/llm_learn_and_lab_demo_1/bio_lab/cpp_data/indptr.bin", std::ios::binary);
    indptr_file.read(reinterpret_cast<char*>(mat.indptr.data()), (mat.n_rows + 1) * sizeof(int));
    indptr_file.close();

    int nnz = mat.indptr[mat.n_rows]; // 获取非零元素总数

    // 分配内存
    mat.indices.resize(nnz);
    mat.values.resize(nnz);

    // 并行加载 Indices 和 Values (OpenMP Sections)
#pragma omp parallel sections
    {
        // 任务 A: 读取 indices
#pragma omp section
        {
            std::ifstream indices_file("D:/llm_learn_and_lab_demo_1/bio_lab/cpp_data/indices.bin", std::ios::binary);
            if (indices_file.is_open()) {
                indices_file.read(reinterpret_cast<char*>(mat.indices.data()), nnz * sizeof(int));
                indices_file.close();
            }
        }

        // 任务 B: 读取 values
#pragma omp section
        {
            std::ifstream data_file("D:/llm_learn_and_lab_demo_1/bio_lab/cpp_data/data.bin", std::ios::binary);
            if (data_file.is_open()) {
                data_file.read(reinterpret_cast<char*>(mat.values.data()), nnz * sizeof(float));
                data_file.close();
            }
        }
    }

    return mat; 
}