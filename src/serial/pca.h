#pragma once
#include <vector>

class SerialPCA {
private:
    int n_components_;          // 主成分数 (默认50)
    std::vector<float> mean_;   // 每个基因的均值 [5000]
    std::vector<float> scale_;  // 每个基因的标准差 [5000]
    std::vector<float> components_; // 特征向量 [50 * 5000]

public:
    SerialPCA(int n_components = 50) : n_components_(n_components) {}

    // 核心函数
    void fit(const float* data, int n_samples, int n_features);

    // 降维
    void transform(const float* data, float* result,
        int n_samples, int n_features) const;

    // 获取主成分
    const std::vector<float>& getComponents() const { return components_; }

private:
    // 步骤1: 计算均值和标准差
    void compute_mean_variance(const float* data, int n_samples, int n_features);

    // 步骤2: 标准化数据
    void standardize(float* data, int n_samples, int n_features) const;

    // 步骤3: 计算协方差矩阵
    void compute_covariance(const float* data, float* cov,
        int n_samples, int n_features) const;

    // 步骤4: 雅可比特征分解（求前k个特征向量）
    void jacobi_eigensolver(float* cov, int n_features);
};  





