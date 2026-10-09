#include "data_loader.h"
#include "sparse_matrix.h"
#include "../binary_io.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>

SparseMatrix loadFromBinary() {
    SparseMatrix mat;

    // 读取meta数据
    const std::string data_dir = "D:/llm_learn_and_lab_demo_1/bio_lab/cpp_data/";
    std::ifstream meta(data_dir + "metadata.txt");
    if (!meta.is_open() || !(meta >> mat.n_rows >> mat.n_cols) ||
        mat.n_rows <= 0 || mat.n_cols <= 0) {
        throw std::runtime_error("Cannot read valid dimensions from metadata.txt");
    }
    meta.close();

    // indptr
    mat.indptr.resize(mat.n_rows + 1);

    readBinaryExact(data_dir + "indptr.bin", mat.indptr.data(),
                    mat.indptr.size() * sizeof(int));

    int nnz = mat.indptr[mat.n_rows]; // 获取非零元素总数
    if (mat.indptr.front() != 0 || nnz < 0) {
        throw std::runtime_error("Invalid CSR row pointer data");
    }

    // 分配内存
    mat.indices.resize(nnz);
    mat.values.resize(nnz);

    // 读取 Indices 
    readBinaryExact(data_dir + "indices.bin", mat.indices.data(),
                    mat.indices.size() * sizeof(int));

    // 读取 Values 
    readBinaryExact(data_dir + "data.bin", mat.values.data(),
                    mat.values.size() * sizeof(float));

    return mat;
}