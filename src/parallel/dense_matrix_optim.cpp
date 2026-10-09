#include "dense_matrix.h"
#include <vector>
#include <algorithm>
#include <omp.h>

DenseMatrix DenseMatrix::fromSparse(
    const SparseMatrix& sparse,
    const std::vector<int>& selected_genes) {

    const int n_spots = sparse.n_rows;
    const int n_selected = selected_genes.size();
    const int n_total_genes = sparse.n_cols; // 总共gene num

    // 创建稠密矩阵
    DenseMatrix dense(n_spots, n_selected);

    // [优化点] 构建直接寻址表
    // 创建一个大小为 18085 的数组，全部填 -1
    std::vector<int> lookup_table(n_total_genes, -1);

    // 填充映射关系
    for (int i = 0; i < n_selected; i++) {
        int original_id = selected_genes[i];
        // check
        if (original_id >= 0 && original_id < n_total_genes) {
            lookup_table[original_id] = i;
        }
    }


#pragma omp parallel for schedule(dynamic, 128)
    for (int spot_idx = 0; spot_idx < n_spots; spot_idx++) {
        int start = sparse.indptr[spot_idx];
        int end = sparse.indptr[spot_idx + 1];

        // 提前算出这一spot的基地址
        float* current_row_ptr = &dense.data_[static_cast<long long>(spot_idx) * n_selected];

        for (int j = start; j < end; j++) {
            int gene_idx = sparse.indices[j];
            float value = sparse.values[j];

            // 用数组下标直接查找
            // 只有当 gene_idx 在范围内，且对应的值不为 -1 
            if (gene_idx < n_total_genes) {
                int new_col = lookup_table[gene_idx];

                if (new_col != -1) {
                    // 填入数据
                    current_row_ptr[new_col] = value;
                }
            }
        }
    }

    return dense;
}


