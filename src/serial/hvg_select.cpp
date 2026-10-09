#include "gene_select.h"
#include "sparse_matrix.h"  
#include <vector>
#include <algorithm>
#include <numeric>  
#include <cmath>    

/**
 * 选择高方差基因
 * @param sparse 归一化后的稀疏矩阵（Log转换后）
 * @param n_genes 要选择的基因数（默认5000）
 * @return 高方差基因的索引列表（按方差降序）
 */
std::vector<int> selectGenesByVariance(const SparseMatrix& sparse, int n_genes ) {
    const int n_spots = sparse.n_rows;    // 166,186 spots
    const int n_genes_total = sparse.n_cols; // 18,085 genes

    // 初始化容器
    std::vector<double> sum(n_genes_total, 0.0);    // 非零值之和
    std::vector<double> sum_sq(n_genes_total, 0.0); // 非零值平方和
    std::vector<int> nnz(n_genes_total, 0);         // 非零值数量

    // 遍历
    for (int spot_idx = 0; spot_idx < n_spots; spot_idx++) {
        const size_t start = sparse.indptr[spot_idx];
        const size_t end = sparse.indptr[spot_idx + 1];

        for (size_t j = start; j < end; j++) {
            const int gene_idx = sparse.indices[j];
            const float value = sparse.values[j];

            sum[gene_idx] += static_cast<double>(value);
            sum_sq[gene_idx] += static_cast<double>(value * value);
            nnz[gene_idx]++;
        }
    }

    // 计算真实方差
    std::vector<double> variances(n_genes_total, 0.0);
    for (int gene_idx = 0; gene_idx < n_genes_total; gene_idx++) {
        double total_sum = sum[gene_idx];
        double total_sq = sum_sq[gene_idx];
        double mean = total_sum / n_spots;
        double mean_sq = total_sq / n_spots;

        // 真实方差 
        double variance = mean_sq - (mean * mean);

        if (variance < 0 && fabs(variance) < 1e-10) {
            variance = 0.0;
        }
        else if (variance < 0) {
            // check
            printf("WARNING: Negative variance for gene %d! (value=%.5f)\n",
                gene_idx, variance);
            variance = 0.0;
        }
        variances[gene_idx] = variance;
    }

    // 创建基因索引并按方差降序排序
    std::vector<int> indices(n_genes_total);
    std::iota(indices.begin(), indices.end(), 0); // 0,1,2,...,n_genes_total-1

    std::sort(indices.begin(), indices.end(),
        [&variances](int a, int b) {
            return variances[a] > variances[b];
        }
    );

    // 选择前n_genes个
    n_genes = std::min(n_genes, n_genes_total);
    indices.resize(n_genes);

    return indices;
}
