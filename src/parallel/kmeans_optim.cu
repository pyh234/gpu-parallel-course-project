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
    //需要检查是否存在重复的index，即选出了重复的点
    for (int i = 0; i < k; i++) {
        int new_idx;
        do {
            new_idx = dis(gen);
        } while (std::find(init_indices.begin(), init_indices.begin() + i, new_idx) != init_indices.begin() + i);

        init_indices[i] = new_idx;
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


// E kernel
__global__ void computeAssignmentsKernel(
    const float* __restrict__ data,      // PCA 数据
    const float* __restrict__ centroids, // 质心数据
    int* __restrict__ assignments,       //label结果 
    float* __restrict__ distances,       // 距离平方 
    int n_spots,
    int n_components,
    int k
) {
    // 计算当前线程负责的样本索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_spots) return;

    float best_dist = 3.402823466e+38F; // float max
    int best_cluster = 0;

    // 数据在全局内存中的起始偏移量
    int data_offset = idx * n_components;

    // 遍历所有质心，每个线程对每个sopt计算对全部代表点的距离平方
    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        int center_offset = c * n_components;


        for (int j = 0; j < n_components; j++) {
            float val = data[data_offset + j];
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
    distances[idx] = best_dist;
}

// E-step cuda版本
void KMeans::computeAssignments(float* d_data, float* d_centroids, int* d_assignments, float* d_distances) {

    //配置 Kernel 参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_spots + threadsPerBlock - 1) / threadsPerBlock;
    computeAssignmentsKernel << <blocksPerGrid, threadsPerBlock >> > (
        d_data,
        d_centroids,
        d_assignments,
        d_distances,
        n_spots,
        n_components,
        k
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}


// M-step Kernel
__global__ void updateCentroidsAccumulateKernel(
    const float* __restrict__ data,
    const int* __restrict__ assignments,
    float* __restrict__ new_centroids_sum,
    int* __restrict__ cluster_counts,
    int n_spots,
    int n_components,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spots) return;

    // 获取当前点所属的簇
    int cluster_id = assignments[idx];

    // 原子操作防止冲突
    atomicAdd(&cluster_counts[cluster_id], 1);

    // 累加坐标
    int data_offset = idx * n_components;
    int center_offset = cluster_id * n_components;

    for (int j = 0; j < n_components; j++) {
        float val = data[data_offset + j];
        //同样用原子操作
        atomicAdd(&new_centroids_sum[center_offset + j], val);
    }
}

// 更新质心并计算移动距离方便（提前推出
float KMeans::updateCentroids(float* d_data, int* d_assignments, float* d_centroids_sum, int* d_counts) {
    int dim = n_components;

    // 每次迭代前，把用于累加的新质心内存和计数器清零
    cudaMemset(d_centroids_sum, 0, k * dim * sizeof(float));
    cudaMemset(d_counts, 0, k * sizeof(int));

    // 启动累加 Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_spots + threadsPerBlock - 1) / threadsPerBlock;

    updateCentroidsAccumulateKernel << <blocksPerGrid, threadsPerBlock >> > (
        d_data,
        d_assignments,
        d_centroids_sum,
        d_counts,
        n_spots,
        dim,
        k
        );
    cudaDeviceSynchronize();

    // 拷贝 k*dim 个数据
    std::vector<float> h_new_centroids_sum(k * dim);
    std::vector<int> h_counts(k);

    cudaMemcpy(h_new_centroids_sum.data(), d_centroids_sum, k * dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts.data(), d_counts, k * sizeof(int), cudaMemcpyDeviceToHost);

    // 计算这次的质心和上次的距离
    float total_shift = 0.0f;

    // 备份旧质心
    std::vector<float> old_centroids = centroids;

    for (int c = 0; c < k; c++) {
        int count = h_counts[c];
        int offset = c * dim;

        if (count > 0) {
            float inv_count = 1.0f / count;
            for (int j = 0; j < dim; j++) {
                // 计算平均值：累加和 / 计数
                centroids[offset + j] = h_new_centroids_sum[offset + j] * inv_count;
            }
        }
    }

    for (int i = 0; i < k * dim; i++) {
        float diff = centroids[i] - old_centroids[i];
        total_shift += diff * diff;
    }

    return total_shift;
}


//主体部分
std::vector<int> KMeans::fit(const std::vector<float>& pca_data, int n_comps) {
    n_components = n_comps;
    n_spots = pca_data.size() / n_components;

    std::cout << "Running KMeans (k=" << k << ") ..." << std::endl;
    initializeCentroids(pca_data);
    //在循环开始前完成数据传输
    float* d_data, * d_centroids, * d_distances, * d_centroids_sum;
    int* d_assignments, * d_counts;

    size_t data_size = n_spots * n_components * sizeof(float);
    size_t centroids_size = k * n_components * sizeof(float);
    size_t assign_size = n_spots * sizeof(int);
    size_t dist_size = n_spots * sizeof(float);
    size_t counts_size = k * sizeof(int);

    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_centroids, centroids_size);
    cudaMalloc(&d_assignments, assign_size);
    cudaMalloc(&d_distances, dist_size);
    cudaMalloc(&d_centroids_sum, centroids_size); // 用于存累加和
    cudaMalloc(&d_counts, counts_size);


    // pca_data 只拷贝一次，一直在显存中
    cudaMemcpy(d_data, pca_data.data(), data_size, cudaMemcpyHostToDevice);

    for (int iter = 0; iter < max_iter; iter++) {
        // E
        cudaMemcpy(d_centroids, centroids.data(), centroids_size, cudaMemcpyHostToDevice);//小数据
        computeAssignments(d_data, d_centroids, d_assignments, d_distances);
        float shift = updateCentroids(d_data, d_assignments, d_centroids_sum, d_counts);

        // 收敛判定
        // 如果质心移动非常小则提前结束
        if (shift < tol) {
            std::cout << "  Converged at iteration " << iter + 1
                << " (Shift: " << shift << " < " << tol << ")" << std::endl;
            break;
        }
    }
    //标签回传
    cudaMemcpy(assignments.data(), d_assignments, assign_size, cudaMemcpyDeviceToHost);

    // --- 5. 清理 ---
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_distances);
    cudaFree(d_centroids_sum);
    cudaFree(d_counts);

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