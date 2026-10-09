#pragma once
#include <vector>
#include "sparse_matrix.h" // 需包含SparseMatrix定义

class DenseMatrix {
private:
    std::vector<float> data_;
    int n_rows_;
    int n_cols_;

public:
    DenseMatrix(int n_rows = 0, int n_cols = 0)
        : n_rows_(n_rows), n_cols_(n_cols) {
        data_.resize(n_rows * n_cols, 0.0f); // 初始化为0
    }

    // 从稀疏矩阵构建稠密子矩阵（关键函数）
    static DenseMatrix fromSparse(const SparseMatrix& sparse,
        const std::vector<int>& selected_genes);

    // 获取数据指针
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    int rows() const { return n_rows_; }
    int cols() const { return n_cols_; }
}; 