#include "pca.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// 步骤1: 计算均值和标准差
void SerialPCA::compute_mean_variance(const float* data,
    int n_samples, int n_features) {
    mean_.resize(n_features, 0.0f);
    scale_.resize(n_features, 0.0f);

    // 1.1 计算均值
    for (int j = 0; j < n_features; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            sum += data[i * n_features + j];
        }
        mean_[j] = sum / n_samples;
    }

    // 1.2 计算标准差
    for (int j = 0; j < n_features; j++) {
        float sum_sq = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float diff = data[i * n_features + j] - mean_[j];
            sum_sq += diff * diff;
        }
        scale_[j] = std::sqrt(sum_sq / (n_samples - 1));

        // check
        if (scale_[j] < 1e-8) scale_[j] = 1.0f;
    }
}

// 步骤2: 标准化数据
void SerialPCA::standardize(float* data, int n_samples, int n_features) const {
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            data[i * n_features + j] =
                (data[i * n_features + j] - mean_[j]) / scale_[j];
        }
    }
}

// 步骤3: 计算协方差矩阵
void SerialPCA::compute_covariance(const float* data, float* cov,
    int n_samples, int n_features) const {
    // 初始化协方差矩阵
    for (int i = 0; i < n_features * n_features; i++) {
        cov[i] = 0.0f;
    }
    // 计算协方差矩阵
    for (int i = 0; i < n_samples; i++) {
        const float* row = &data[i * n_features];
        for (int j = 0; j < n_features; j++) {
            for (int k = j; k < n_features; k++) { // 利用对称性
                cov[j * n_features + k] += row[j] * row[k];
            }
        }
    }
    // 除n-1、处理对称性
    float inv_n = 1.0f / (n_samples - 1);
    for (int j = 0; j < n_features; j++) {
        for (int k = j; k < n_features; k++) {
            float val = cov[j * n_features + k] * inv_n;
            cov[j * n_features + k] = val;
            cov[k * n_features + j] = val; // 对称位置
        }
    }
}

// 步骤4: 雅可比法求特征向量（只求前k个）
void SerialPCA::jacobi_eigensolver(float* cov, int n_features) {
    const int MAX_ITER = 50;
    const float EPS = 1e-6f;

    // 1. 初始化特征向量矩阵（单位矩阵）
    components_.resize(n_components_ * n_features, 0.0f);
    std::vector<float> V(n_features * n_features, 0.0f);
    for (int i = 0; i < n_features; i++) {
        V[i * n_features + i] = 1.0f;
    }

    // 2. 雅可比旋转迭代
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 寻找最大非对角元素
        float max_val = 0.0f;
        int p = 0, q = 1;

        for (int i = 0; i < n_features; i++) {
            for (int j = i + 1; j < n_features; j++) {
                float val = std::abs(cov[i * n_features + j]);
                if (val > max_val) {
                    max_val = val;
                    p = i; q = j;
                }
            }
        }

        if (max_val < EPS) break; // 收敛

        // 计算旋转角度
        float app = cov[p * n_features + p];
        float aqq = cov[q * n_features + q];
        float apq = cov[p * n_features + q];

        float theta = 0.5f * std::atan2(2.0f * apq, aqq - app);
        float c = std::cos(theta);
        float s = std::sin(theta);

        // 更新协方差矩阵
        float new_app = c * c * app - 2 * s * c * apq + s * s * aqq;
        float new_aqq = s * s * app + 2 * s * c * apq + c * c * aqq;
        cov[p * n_features + p] = new_app;
        cov[q * n_features + q] = new_aqq;
        cov[p * n_features + q] = cov[q * n_features + p] = 0.0f;

        for (int k = 0; k < n_features; k++) {
            if (k != p && k != q) {
                float apk = cov[p * n_features + k];
                float aqk = cov[q * n_features + k];
                cov[p * n_features + k] = cov[k * n_features + p] = c * apk - s * aqk;
                cov[q * n_features + k] = cov[k * n_features + q] = s * apk + c * aqk;
            }
        }

        // 更新特征向量
        for (int k = 0; k < n_features; k++) {
            float vpk = V[k * n_features + p];
            float vqk = V[k * n_features + q];
            V[k * n_features + p] = c * vpk - s * vqk;
            V[k * n_features + q] = s * vpk + c * vqk;
        }
    }

    // 3. 提取前k个特征向量（按特征值降序）
    std::vector<std::pair<float, int>> eigen_pairs;
    for (int i = 0; i < n_features; i++) {
        eigen_pairs.push_back({ cov[i * n_features + i], i });
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first; // 降序
        });

    // 复制前n_components_个特征向量
    for (int comp = 0; comp < n_components_; comp++) {
        int orig_idx = eigen_pairs[comp].second;
        for (int j = 0; j < n_features; j++) {
            components_[comp * n_features + j] =
                V[j * n_features + orig_idx];
        }
    }
}

// 核心函数：拟合PCA模型
void SerialPCA::fit(const float* data, int n_samples, int n_features) {
    // 步骤1: 复制数据（避免修改原始数据）
    std::vector<float> data_copy(n_samples * n_features);
    std::copy(data, data + n_samples * n_features, data_copy.data());

    // 步骤2: 计算均值/标准差
    compute_mean_variance(data_copy.data(), n_samples, n_features);

    // 步骤3: 标准化
    standardize(data_copy.data(), n_samples, n_features);

    // 步骤4: 计算协方差矩阵
    std::vector<float> cov(n_features * n_features);
    compute_covariance(data_copy.data(), cov.data(), n_samples, n_features);

    // 步骤5: 特征分解
    jacobi_eigensolver(cov.data(), n_features);
}

// 降维：投影到主成分
void SerialPCA::transform(const float* data, float* result,
    int n_samples, int n_features) const {
    // 标准化新数据
    std::vector<float> data_std(n_samples * n_features);
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            int idx = i * n_features + j;
            data_std[idx] = (data[idx] - mean_[j]) / scale_[j];
        }
    }

    // 矩阵乘法: result = data_std * components_^T
    for (int i = 0; i < n_samples; i++) {
        for (int k = 0; k < n_components_; k++) {
            float sum = 0.0f;
            for (int j = 0; j < n_features; j++) {
                sum += data_std[i * n_features + j] *
                    components_[k * n_features + j];
            }
            result[i * n_components_ + k] = sum;
        }
    }
}