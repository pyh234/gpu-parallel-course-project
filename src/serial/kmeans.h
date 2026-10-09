#pragma once
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <iostream>
#include <cmath> 

class KMeans {
private:
    int k;              // 聚类数
    int max_iter;       // 最大迭代次数
    float tol;          // 收敛阈值 (tolerance)
    int n_components;   // PCA维度
    int n_spots;        // 样本数量

    // 存储容器
    std::vector<float> centroids;
    std::vector<int> assignments;
    std::vector<float> distances;

    void initializeCentroids(const std::vector<float>& pca_data);
    void computeAssignments(const std::vector<float>& pca_data);

    // 表示质心移动的总距离
    float updateCentroids(const std::vector<float>& pca_data);

public:
    // 构造函数增加 tol 参数，默认 1e-4，方便提前结束
    KMeans(int k = 10, int max_iter = 100, float tol = 1e-4)
        : k(k), max_iter(max_iter), tol(tol), n_components(0), n_spots(0) {
    }

    ~KMeans() = default;

    std::vector<int> fit(const std::vector<float>& pca_data, int n_comps);

    std::vector<std::vector<float>> getCentroids() const;
};