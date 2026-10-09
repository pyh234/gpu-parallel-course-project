#pragma once
#include <vector>


struct SparseMatrix;

/**
 * 选择高方差基因（修正版）
 * @param sparse 归一化后的稀疏矩阵（Log转换后）
 * @param n_genes 要选择的基因数（默认5000）
 * @return 高方差基因的索引列表（按方差降序）
 */
std::vector<int> selectGenesByVariance(const SparseMatrix& sparse, int n_genes = 5000);