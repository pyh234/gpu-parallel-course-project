#include "dense_matrix.h"
#include <unordered_map>
#include <algorithm>

DenseMatrix DenseMatrix::fromSparse(
    const SparseMatrix& sparse,
    const std::vector<int>& selected_genes) {

    const int n_spots = sparse.n_rows;
    const int n_selected = selected_genes.size();

    // 创建稠密矩阵，初始值都为0
    DenseMatrix dense(n_spots, n_selected);

    // 构建基因索引映射
    std::unordered_map<int, int> gene_to_new_index;
    for (int i = 0; i < n_selected; i++) {
        gene_to_new_index[selected_genes[i]] = i;
    }

    // 填充非零值
    for (int spot_idx = 0; spot_idx < n_spots; spot_idx++) {
        int start = sparse.indptr[spot_idx];
        int end = sparse.indptr[spot_idx + 1];

        for (int j = start; j < end; j++) {
            int gene_idx = sparse.indices[j];
            float value = sparse.values[j];

            // 检查基因是否被选中
            auto it = gene_to_new_index.find(gene_idx);
            if (it != gene_to_new_index.end()) {
                int new_col = it->second;
                dense.data_[spot_idx * n_selected + new_col] = value;
            }
            // 未选中的基因保持为0
        }
    }

    return dense;
}