#include "pca_cuda.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iomanip>

#define BLOCK_SIZE 256


// KERNEL 1 统计 每一列数据
// 策略：每个 Block 处理一个列，利用 Shared Memory 归约

__global__ void computeStatsKernel(const float* data, float* mean, float* scale, int n_samples, int n_features) {
    int feature_idx = blockIdx.x; // 当前处理哪一列
    int tid = threadIdx.x;

    if (feature_idx >= n_features) return;

    double sum = 0.0;
    double sum_sq = 0.0;

    // 1. 线程并行累加，跨步循环
    for (int i = tid; i < n_samples; i += blockDim.x) {
        float val = data[i * n_features + feature_idx];
        sum += val;
        sum_sq += val * val;
    }

    // 2. 共享内存归约
    __shared__ double s_sum[BLOCK_SIZE];
    __shared__ double s_sq[BLOCK_SIZE];
    s_sum[tid] = sum;
    s_sq[tid] = sum_sq;
    __syncthreads();

    // 典型的树状归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }

    // 3. 线程0写回并计算最终统计量
    if (tid == 0) {
        float m = (float)(s_sum[0] / n_samples);
        float s = (float)(s_sq[0] - (s_sum[0] * s_sum[0]) / n_samples); 
        double var = (s_sq[0] - (s_sum[0] * s_sum[0]) / n_samples) / (n_samples - 1);

        mean[feature_idx] = m;
        scale[feature_idx] = (var > 1e-8) ? sqrt(var) : 1.0f;
    }
}

// KERNEL 2: 标准化 (Standardize)
// 策略：每个线程处理一个元素
__global__ void standardizeKernel(float* data, const float* mean, const float* scale, int n_elements, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int col = idx % n_features;
        data[idx] = (data[idx] - mean[col]) / scale[col];
    }
}


// KERNEL 3: 协方差矩阵计算
// 策略：每个线程计算 Cov[row][col] 的一个元素。

__global__ void computeCovarianceKernel(const float* data, float* cov, int n_samples, int n_features) {
    // 输出矩阵的坐标 (row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_features && col < n_features) {
        // 利用对称性，只计算上三角 (row <= col)
        if (row <= col) {
            double sum = 0.0;
            // 计算两列的点积
            for (int k = 0; k < n_samples; k++) {
                float val1 = data[k * n_features + row];
                float val2 = data[k * n_features + col];
                sum += val1 * val2;
            }
            float res = (float)(sum / (n_samples - 1));
            cov[row * n_features + col] = res;
            cov[col * n_features + row] = res; // 对称赋值
        }
    }
}

// =============================================================
// KERNEL 4: 雅可比迭代 - 寻找最大非对角元素 (Find Max Off-Diagonal)
// =============================================================
// 这是一个并行 Reduction 来找最大值的 Index (p, q)
// 把数据规约到一个小数组，然后在 CPU 上找最终的 p, q
// d_max_vals: 存储每个 Block 的最大值, d_max_indices: 存储对应的 index

__global__ void findMaxOffDiagKernel(const float* cov, float* d_max_vals, int* d_p, int* d_q, int n_features) {
    // 动态共享内存
    extern __shared__ float s_max_val[];
    int* s_p = (int*)&s_max_val[blockDim.x];
    int* s_q = (int*)&s_p[blockDim.x];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 总元素个数
    float local_max = -1.0f;
    int local_p = 0;
    int local_q = 1;

    // 跨步检索
    int total_elements = n_features * n_features;
    for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
        int r = i / n_features;
        int c = i % n_features;

        if (c > r) { // 只看上三角且非对角
            float val = fabsf(cov[i]);
            if (val > local_max) {
                local_max = val;
                local_p = r;
                local_q = c;
            }
        }
    }

    // 存入共享内存
    s_max_val[tid] = local_max;
    s_p[tid] = local_p;
    s_q[tid] = local_q;
    __syncthreads();

    // Block 内归约找最大
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max_val[tid + stride] > s_max_val[tid]) {
                s_max_val[tid] = s_max_val[tid + stride];
                s_p[tid] = s_p[tid + stride];
                s_q[tid] = s_q[tid + stride];
            }
        }
        __syncthreads();
    }

    // 写回 Block 结果
    if (tid == 0) {
        d_max_vals[blockIdx.x] = s_max_val[0];
        d_p[blockIdx.x] = s_p[0];
        d_q[blockIdx.x] = s_q[0];
    }
}


// KERNEL 5 旋转更新kernel
// 并行更新 Cov 矩阵和特征向量矩阵 V
__global__ void jacobiRotateKernel(float* cov, float* V, int p, int q, float c, float s, int n_features) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_features) return;

    // 更新 Cov (只更新第 p 行/列 和 第 q 行/列)
    if (k != p && k != q) {
        float cov_pk = cov[p * n_features + k];
        float cov_qk = cov[q * n_features + k];

        // 公式参考串行代码
        float new_cov_pk = c * cov_pk - s * cov_qk;
        float new_cov_qk = s * cov_pk + c * cov_qk;

        cov[p * n_features + k] = new_cov_pk;
        cov[k * n_features + p] = new_cov_pk; // 对称
        cov[q * n_features + k] = new_cov_qk;
        cov[k * n_features + q] = new_cov_qk; // 对称
    }

    // 更新特征向量 V (所有行都需要更新)
    float v_kp = V[k * n_features + p]; // V[k][p]
    float v_kq = V[k * n_features + q]; // V[k][q]

    V[k * n_features + p] = c * v_kp - s * v_kq;
    V[k * n_features + q] = s * v_kp + c * v_kq;
}


// KERNEL 6: 矩阵乘法 (Transform)
// Result = Data * Components^T

__global__ void matrixMulKernel(const float* data, const float* components, float* result,
    int n_samples, int n_features, int n_components) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Sample index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Component index

    if (row < n_samples && col < n_components) {
        double sum = 0.0;
        // 点积：Data的第row行 dot Components的第col行
        for (int k = 0; k < n_features; k++) {
            float d = data[row * n_features + k];
            float c = components[col * n_features + k]; // components 存储是 [n_comp][n_feat]
            sum += d * c;
        }
        result[row * n_components + col] = (float)sum;
    }
}



// 主机端逻辑实现 (Host Implementation)
void CudaPCAManual::fit(const float* host_data, int n_samples, int n_features) {
    n_samples_ = n_samples;
    n_features_ = n_features;

    size_t data_size = (size_t)n_samples * n_features * sizeof(float);

    // 分配显存
    float* d_data, * d_mean, * d_scale, * d_cov, * d_V;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_mean, n_features * sizeof(float));
    cudaMalloc(&d_scale, n_features * sizeof(float));
    cudaMalloc(&d_cov, n_features * n_features * sizeof(float)); // 协方差矩阵
    cudaMalloc(&d_V, n_features * n_features * sizeof(float)); // 特征向量矩阵

    // 拷贝原始数据
    cudaMemcpy(d_data, host_data, data_size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // kernel 1 计时和优化
    printf("  [kernel 1] :\n");
    //打点开始
    cudaEventRecord(start);
    // 启动 Kernel
    computeStatsKernel << <5000, 256 >> > (d_data, d_mean, d_scale, n_samples, n_features);
    // 打点结束
    cudaEventRecord(stop);
    //必须等待这一次跑完才能算时间
    cudaEventSynchronize(stop);
    // 计算本次耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // 输出单次结果 (us)
    printf("    Run %2d: %.3f us\n", 1, milliseconds * 1000.0f);


    // kernel 2
    printf("  [kernel 2] :\n");
    int numBlocks = (n_samples * n_features + 256 - 1) / 256;
    cudaEventRecord(start);
    standardizeKernel << <numBlocks, 256 >> > (d_data, d_mean, d_scale, n_samples * n_features, n_features);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("    Run %2d: %.3f us\n", 1, milliseconds * 1000.0f);

    // 计算协方差矩阵 kernel 3
    // 这是耗时操作(还能优化？？？）
    printf("  [kernel 3] :\n");
    float total_time_ms = 0.0f;
    const int iterations = 1; // 设定计时测试次数
    //int threads_per_block = 256;

    dim3 dimBlock(16, 16);
    dim3 dimGrid((n_features + 15) / 16, (n_features + 15) / 16);
    for (int i = 0; i < iterations ; i++) {
        cudaEventRecord(start);

        // 启动 Kernel
        computeCovarianceKernel << <dimGrid, dimBlock >> > (d_data, d_cov, n_samples, n_features);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (i > -1) {
            total_time_ms += milliseconds;
        }

        printf("    Run %2d: %.3f us\n", i + 1, milliseconds * 1000.0f);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("!!!! CUDA ERROR in Run %d: %s !!!!\n", i + 1, cudaGetErrorString(err));
            return; // 遇到错误直接停止测试
        }
    }

    float avg_us = (total_time_ms * 1000.0f) / iterations;
    printf("  [kernel 3] Average Time: %.3f us\n\n", avg_us);


    //对这一整块进行计时
    auto jacobi_start = std::chrono::high_resolution_clock::now();
    // 初始化特征向量矩阵 V 为单位矩阵
    std::vector<float> h_I(n_features * n_features, 0.0f);
    for (int i = 0; i < n_features; ++i) h_I[i * n_features + i] = 1.0f;
    cudaMemcpy(d_V, h_I.data(), n_features * n_features * sizeof(float), cudaMemcpyHostToDevice);

    // 雅可比迭代
    const int MAX_ITER = 50; // 设定的迭代次数
    float* d_max_vals;
    int* d_p, * d_q;
    // 分配归约用的临时空间
    int reduce_blocks = 256;
    cudaMalloc(&d_max_vals, reduce_blocks * sizeof(float));
    cudaMalloc(&d_p, reduce_blocks * sizeof(int));
    cudaMalloc(&d_q, reduce_blocks * sizeof(int));

    // Host端临时变量
    std::vector<float> h_max_vals(reduce_blocks);
    std::vector<int> h_p(reduce_blocks);
    std::vector<int> h_q(reduce_blocks);

    std::cout << "Starting Jacobi Iteration (Manual CUDA)..." << std::endl;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // 寻找最大非对角元素 (Kernel 4)
        // 使用动态共享内存: size = 3 * BLOCK_SIZE * sizeof(type) (float + int + int)
        size_t shared_mem_size = BLOCK_SIZE * sizeof(float) + 2 * BLOCK_SIZE * sizeof(int);
        findMaxOffDiagKernel << <reduce_blocks, BLOCK_SIZE, shared_mem_size >> > (d_cov, d_max_vals, d_p, d_q, n_features);

        // 拷回 CPU 做最后一步归约 (因为 block 数很少，CPU 做这一步极快)
        cudaMemcpy(h_max_vals.data(), d_max_vals, reduce_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p.data(), d_p, reduce_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_q.data(), d_q, reduce_blocks * sizeof(int), cudaMemcpyDeviceToHost);

        float max_val = 0.0f;
        int p = 0, q = 1;
        for (int i = 0; i < reduce_blocks; ++i) {
            if (h_max_vals[i] > max_val) {
                max_val = h_max_vals[i];
                p = h_p[i];
                q = h_q[i];
            }
        }

        if (max_val < 1e-6f) break; // 收敛

        // 计算旋转参数 (Host 计算)
        // 需要读取 Cov[p][p], Cov[q][q], Cov[p][q]来计算旋转角度
        //将对应的三个数从cuda上传递到主机上用cpu处理
        float app, aqq, apq;
        cudaMemcpy(&app, &d_cov[p * n_features + p], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&aqq, &d_cov[q * n_features + q], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&apq, &d_cov[p * n_features + q], sizeof(float), cudaMemcpyDeviceToHost);

        float theta = 0.5f * std::atan2(2.0f * apq, app - aqq);
        float c = std::cos(theta);
        float s = std::sin(theta);

        // 手动更新对角线元素 (为了减少开销，这几个数直接算好填进去）
        float new_app = c * c * app - 2 * s * c * apq + s * s * aqq;
        float new_aqq = s * s * app + 2 * s * c * apq + c * c * aqq;
        float zero = 0.0f;
        cudaMemcpy(&d_cov[p * n_features + p], &new_app, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_cov[q * n_features + q], &new_aqq, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_cov[p * n_features + q], &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_cov[q * n_features + p], &zero, sizeof(float), cudaMemcpyHostToDevice);

        // 执行旋转更新其他元素 (Kernel 5)
        int rot_blocks = (n_features + BLOCK_SIZE - 1) / BLOCK_SIZE;
        jacobiRotateKernel << <rot_blocks, BLOCK_SIZE >> > (d_cov, d_V, p, q, c, s, n_features);
    }
    auto jacobi_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_jacobi = jacobi_end - jacobi_start;
    printf("   [jacobi_entire_time %d] %.6fs\n", 1, elapsed_jacobi);

    // 排序特征值并提取 Top K (Host 辅助)
    // 因为 K 很小，我们可以把 对角线元素(特征值) 和 整个 V 拷回来排序
    // 全在 GPU 排序比较复杂，需要Thrust，这里用 CPU 
    std::vector<float> h_diag(n_features);
    std::vector<float> h_V_full(n_features * n_features);

    // 提取对角线 
    std::vector<float> h_cov_full(n_features * n_features);
    cudaMemcpy(h_cov_full.data(), d_cov, n_features * n_features * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_features; ++i) h_diag[i] = h_cov_full[i * n_features + i];

    cudaMemcpy(h_V_full.data(), d_V, n_features * n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // 排序索引
    std::vector<int> indices(n_features);
    for (int i = 0; i < n_features; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return h_diag[a] > h_diag[b]; // 降序
        });

    // 提取 Components
    h_components_.resize(n_components_ * n_features);
    for (int k = 0; k < n_components_; ++k) {
        int idx = indices[k];
        for (int j = 0; j < n_features; ++j) {
            h_components_[k * n_features + j] = h_V_full[j * n_features + idx];
        }
    }

    // 清理
    cudaFree(d_data); cudaFree(d_mean); cudaFree(d_scale);
    cudaFree(d_cov); cudaFree(d_V);
    cudaFree(d_max_vals); cudaFree(d_p); cudaFree(d_q);
}

void CudaPCAManual::transform(const float* host_data, float* host_result, int n_samples, int n_features) {
    // 简单实现：分配 -> 拷贝 -> 矩阵乘法 Kernel -> 拷贝回
    float* d_data, * d_comps, * d_res;
    size_t data_size = (size_t)n_samples * n_features * sizeof(float);
    size_t comp_size = (size_t)n_components_ * n_features * sizeof(float);
    size_t res_size = (size_t)n_samples * n_components_ * sizeof(float);

    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_comps, comp_size);
    cudaMalloc(&d_res, res_size);

    cudaMemcpy(d_data, host_data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_comps, h_components_.data(), comp_size, cudaMemcpyHostToDevice);

    // 矩阵乘法
    dim3 dimBlock(16, 16);
    dim3 dimGrid((n_components_ + 15) / 16, (n_samples + 15) / 16);
    // 注意 matrixMulKernel 参数顺序
    matrixMulKernel << <dimGrid, dimBlock >> > (d_data, d_comps, d_res, n_samples, n_features, n_components_);

    cudaMemcpy(host_result, d_res, res_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data); cudaFree(d_comps); cudaFree(d_res);
}