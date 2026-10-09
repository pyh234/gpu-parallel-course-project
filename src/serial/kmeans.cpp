#include "kmeans.h"

void KMeans::initializeCentroids(const std::vector<float>& pca_data) {
    int dim = n_components; // 维度就是 PCA 的维度
    centroids.assign(k * dim, 0.0f);
    assignments.resize(n_spots);
    distances.resize(n_spots);

    // 随机选择 k 个点作为初始中心
    std::vector<int> init_indices(k);
    const unsigned int FIXED_SEED = 7; // 使用固定的种子
    std::mt19937 gen(FIXED_SEED);
    std::uniform_int_distribution<> dis(0, n_spots - 1);

    for (int i = 0; i < k; i++) {
        init_indices[i] = dis(gen);
    }

    // 复制初始中心
    for (int i = 0; i < k; i++) {
        int idx = init_indices[i];
        int src_offset = idx * n_components;
        int dst_offset = i * dim;

        // 内存拷贝 
        std::copy(pca_data.begin() + src_offset,
            pca_data.begin() + src_offset + n_components,
            centroids.begin() + dst_offset);
    }
}

// 计算分配 (E-step)
void KMeans::computeAssignments(const std::vector<float>& pca_data) {
    int dim = n_components;

    for (int idx = 0; idx < n_spots; idx++) {
        float best_dist = std::numeric_limits<float>::max();
        int best_cluster = 0;

        // 当前点的 PCA 数据起始位置
        int data_offset = idx * n_components;

        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            int center_offset = c * dim;

            // 计算纯 PCA 特征的欧氏距离平方
            for (int j = 0; j < n_components; j++) {
                float val = pca_data[data_offset + j];
                float center_val = centroids[center_offset + j];
                float diff = val - center_val;
                dist += diff * diff;
            }

            // 更新最近簇
            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = c;
            }
        }

        assignments[idx] = best_cluster;
        distances[idx] = best_dist; // 虽然这里存的是平方距离，但不影响比较
    }
}

// 更新质心并计算移动距离方便（提前退出
float KMeans::updateCentroids(const std::vector<float>& pca_data) {
    int dim = n_components;

    // 备份旧质心，并清空当前向量
    std::vector<float> old_centroids = centroids;
    std::fill(centroids.begin(), centroids.end(), 0.0f);
    std::vector<int> counts(k, 0);

    // 累加所有点的数据
    for (int idx = 0; idx < n_spots; idx++) {
        int cluster = assignments[idx];
        counts[cluster]++;

        int center_offset = cluster * dim;
        int data_offset = idx * n_components;

        for (int j = 0; j < n_components; j++) {
            centroids[center_offset + j] += pca_data[data_offset + j];
        }
    }

    // 计算平均值 (得到新质心)
    for (int c = 0; c < k; c++) {
        int count = counts[c];
        if (count > 0) {
            int center_offset = c * dim;
            float inv_count = 1.0f / count;
            for (int j = 0; j < dim; j++) {
                centroids[center_offset + j] *= inv_count;
            }
        }
    }

    // 计算质心总位移 (shift)
    float total_shift = 0.0f;
    for (int i = 0; i < k * dim; i++) {
        float diff = centroids[i] - old_centroids[i];
        total_shift += diff * diff; // 使用平方和，避免开根号更高效
    }

    return total_shift; // 返回总平方位移
}

std::vector<int> KMeans::fit(const std::vector<float>& pca_data, int n_comps) {
    n_components = n_comps;

    if (n_components == 0) return {};
    n_spots = pca_data.size() / n_components;

    std::cout << "Running KMeans (k=" << k << ") with convergence check..." << std::endl;

    initializeCentroids(pca_data);

    for (int iter = 0; iter < max_iter; iter++) {
        // E-Step
        computeAssignments(pca_data);

        // M-Step (获取移动距离)
        float shift = updateCentroids(pca_data);
        // 如果质心移动非常小 则收敛提前结束
        if (shift < tol) {
            std::cout << "  Converged at iteration " << iter + 1
                << " (Shift: " << shift << " < " << tol << ")" << std::endl;
            break;
        }
    }

    std::cout << "Clustering done." << std::endl;
    return assignments;
}

// 获取质心
std::vector<std::vector<float>> KMeans::getCentroids() const {
    std::vector<std::vector<float>> result(k, std::vector<float>(n_components));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n_components; ++j) {
            result[i][j] = centroids[i * n_components + j];
        }
    }
    return result;
}