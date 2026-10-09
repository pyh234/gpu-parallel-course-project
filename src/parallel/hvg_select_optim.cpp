#include "gene_select.h"
#include "sparse_matrix.h"  // 假设包含CSR结构: n_rows, n_cols, indptr, indices, values
#include <vector>
#include <algorithm>
#include <numeric>  // for std::iota
#include <cmath>    // for fabs
#include <omp.h>

/**
 * 选择高方差基因
 * @param sparse 归一化后的稀疏矩阵（Log转换后）
 * @param n_genes 要选择的基因数（默认5000）
 * @return 高方差基因的索引列表（按方差降序）
 */

std::vector<int> selectGenesByVariance(const SparseMatrix& sparse, int n_genes) {
    const int n_spots = sparse.n_rows;    
    const int n_genes_total = sparse.n_cols; 
    const int total_nnz = sparse.indptr[sparse.n_rows]; 

    // 全局统计量
    std::vector<double> global_sum(n_genes_total, 0.0);
    std::vector<double> global_sum_sq(n_genes_total, 0.0);

    //并行区域
#pragma omp parallel
    {
        // 每个线程创建私有的统计数组
        std::vector<double> local_sum(n_genes_total, 0.0);
        std::vector<double> local_sum_sq(n_genes_total, 0.0);

        // 并行遍历所有非零元素 (忽略spot行结构)
#pragma omp for schedule(dynamic, 4096)
        for (int i = 0; i < total_nnz; ++i) {
            int gene_idx = sparse.indices[i];
            double val = static_cast<double>(sparse.values[i]);

            local_sum[gene_idx] += val;
            local_sum_sq[gene_idx] += val * val;
        }

        // 归约：将局部结果累加到全局结果 (加锁保护)
#pragma omp critical
        {
            for (int j = 0; j < n_genes_total; ++j) {
                global_sum[j] += local_sum[j];
                global_sum_sq[j] += local_sum_sq[j];
            }
        }
    }

    // 计算方差
    std::vector<double> variances(n_genes_total, 0.0);
    for (int gene_idx = 0; gene_idx < n_genes_total; gene_idx++) {
        double total_sum = global_sum[gene_idx];
        double total_sq = global_sum_sq[gene_idx];
        double mean = total_sum / n_spots;
        double mean_sq = total_sq / n_spots;

        // 方差 
        double variance = mean_sq - (mean * mean);

        // check
        if (variance < 0 && fabs(variance) < 1e-10) {
            variance = 0.0;
        }
        else if (variance < 0) {
            printf("WARNING: Negative variance for gene %d! (value=%.5f)\n",
                gene_idx, variance);
            variance = 0.0;
        }

        variances[gene_idx] = variance;
    }


    // 排序优化：使用 partial_sort 只排前 n 个
    std::vector<int> indices(n_genes_total);
    std::iota(indices.begin(), indices.end(), 0);

    // 只要前 n_genes 个是有序的即可，后面不用管
    if (n_genes < n_genes_total) {
        std::partial_sort(indices.begin(), indices.begin() + n_genes, indices.end(),
            [&variances](int a, int b) {
                return variances[a] > variances[b];
            });
        indices.resize(n_genes);
    }
    else {
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {return variances[a] > variances[b]; });
    }
    return indices;
}





