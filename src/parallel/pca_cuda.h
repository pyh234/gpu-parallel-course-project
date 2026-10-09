#pragma once
#include <vector>
#include <cuda_runtime.h>

class CudaPCAManual {
private:
    int n_components_;
    int n_features_;
    int n_samples_;

    // Host 端结果副本
    std::vector<float> h_components_;
    std::vector<float> h_mean_;
    std::vector<float> h_scale_;

public:
    CudaPCAManual(int n_components = 50) : n_components_(n_components) {}
    ~CudaPCAManual() {} 

    // 核心：全GPU流程拟合
    void fit(const float* host_data, int n_samples, int n_features);

    // 变换
    void transform(const float* host_data, float* host_result, int n_samples, int n_features);

    const std::vector<float>& getComponents() const { return h_components_; }
};