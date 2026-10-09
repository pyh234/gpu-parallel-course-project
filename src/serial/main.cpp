#include <iostream>
#include <vector>
#include <chrono> 
#include <cstdio> 
#include <fstream>      
#include <functional>


// 头文件
#include "data_loader.h"
#include "sparse_matrix.h"
#include "gene_select.h"
#include "dense_matrix.h"
#include "pca.h"
#include "kmeans.h"

//一个简单的多轮次计时函数
void benchmark(const std::string& name, int iterations, std::function<void()> func) {
    printf(">> [Benchmark] Starting: %s (Running %d loops)...\n", name.c_str(), iterations);

    double total_execution_time = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        double current_run_time = elapsed.count();
        if (i > -1) {
            total_execution_time += current_run_time;
        }
        printf("   [Run %d] %.6fs\n", i + 1, current_run_time);
    }

    double avg_time = total_execution_time / iterations;

    printf("-- [Result] %s\n", name.c_str());
    printf("   Total (without first): %.6fs | Avg: %.6fs\n", total_execution_time, avg_time);
    printf("--------------------------------------------------\n");
}

// 保存 PCA 降维结果 (CSV格式)
void saveEmbeddings(const std::string& filepath, const std::vector<float>& data, int rows, int cols) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << " for writing." << std::endl;
        return;
    }

    std::cout << "Saving embeddings to: " << filepath << " ..." << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << data[i * cols + j];
            if (j < cols - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Save complete." << std::endl;
}

// 保存 KMeans 标签结果 (CSV格式)
void saveLabels(const std::string& filepath, const std::vector<int>& labels) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << " for writing." << std::endl;
        return;
    }

    std::cout << "Saving labels to: " << filepath << " ..." << std::endl;
    file << "Cluster_Label\n"; // Header
    for (const auto& label : labels) {
        file << label << "\n";
    }
    file.close();
    std::cout << "Save complete." << std::endl;
}

int main() {
    //数据存储位置
    auto pipeline_start = std::chrono::high_resolution_clock::now();//整个程序计时
    const std::string output_dir = "D:/llm_learn_and_lab_demo_1/bio_lab/cpp_data/";
    
    // --------------------------
    // 1. 加载数据 (Sparse Matrix)
    // --------------------------

    std::cout << "Loading sparse matrix from binary...\n" << std::endl;
    
    SparseMatrix raw_data;
    benchmark("Load Sparse Matrix", 1, [&]() {
        raw_data = loadFromBinary();
        });

    printf("\nLoaded: %d spots x %d genes (NNZ: %d)\n",
        raw_data.n_rows, raw_data.n_cols, raw_data.indptr[raw_data.n_rows]);

    printf("Data Loaded\n");


    // ---------------------------------------------------------
    // 2. 特征选择 (Select HVGs)
    // ---------------------------------------------------------
    int target_genes = 5000;
    std::vector<int> selected_genes;

    std::cout << "Selecting top " << target_genes << " high variance genes...\n" << std::endl;

    benchmark("[2]Select HVGs", 1, [&]() {
        selected_genes = selectGenesByVariance(raw_data, target_genes);
        });

    printf("\nSelected %zu genes. Top gene ID: %d\n",
        selected_genes.size(), selected_genes[0]);

    printf("Feature Selection Done\n");


    // ---------------------------------------------------------
    // 3. 稀疏转稠密 (Sparse -> Dense)
    // ---------------------------------------------------------

    std::cout << "Converting to Dense Matrix (this may take memory)...\n" << std::endl;

    DenseMatrix dense_data;
    benchmark("[3]Sparse to Dense Conversion", 1, [&]() {
        dense_data = DenseMatrix::fromSparse(raw_data, selected_genes);
        });

    printf("\nDense Matrix Created: %d rows x %d cols\n",
        dense_data.rows(), dense_data.cols());

    printf("Conversion Done\n");

    //const float* raw_ptr = dense_data.data();
    //int width = dense_data.cols(); // 应该是 5000

    //// 打印前 3 行
    //for (int i = 0; i < 3; i++) {
    //    printf("Row %d: ", i);
    //    // 打印每行的前 15 个数据
    //    for (int j = 0; j < 15; j++) {
    //        // 计算一维数组中的索引：行号 * 宽度 + 列号
    //        float val = raw_ptr[i * width + j];
    //        printf("%.4f ", val);
    //    }
    //    printf("\n");
    //}

    // ---------------------------------------------------------
    // 4. PCA 分析 (Dimensionality Reduction)
    // ---------------------------------------------------------
    int n_pcs = 50;
    SerialPCA pca(n_pcs);

    std::cout << "Running PCA fit (calculating covariance & eigenvectors)..." << std::endl;
    std::cout << "WARNING: Since we use Jacobi method, this step might be slow for 5000 genes." << std::endl;

    // 训练模型
    benchmark("[4]PCA Fit (Covariance + Jacobi)", 1, [&]() {
        pca.fit(dense_data.data(), dense_data.rows(), dense_data.cols());
        });

    printf("PCA Fit Done");

    // ---------------------------------------------------------
    // 5. 数据投影 (Transform)
    // ---------------------------------------------------------

    std::cout << "Projecting data to " << n_pcs << " PCs..." << std::endl;

    // 准备结果容器 (166186 * 50)
    std::vector<float> embedding(dense_data.rows() * n_pcs);

    // 执行降维
    benchmark("[5]PCA Transform", 1, [&]() {
        pca.transform(dense_data.data(), embedding.data(), dense_data.rows(), dense_data.cols());
        });

    // 保存 PCA 结果
    //saveEmbeddings(output_dir + "pca_embeddings.csv", embedding, dense_data.rows(), n_pcs);

    printf("PCA Transform Done");

    // ---------------------------------------------------------
    // 6. 输出验证 (Validation)
    // ---------------------------------------------------------
    //std::cout << "\nResult Verification (Top 3 Spots):" << std::endl;
    //for (int i = 0; i < 3; i++) {
    //    printf("Spot %d: [PC1: %.4f, PC2: %.4f, ... PC50: %.4f]\n",
    //        i,
    //        embedding[i * n_pcs + 0], // PC1
    //        embedding[i * n_pcs + 1], // PC2
    //        embedding[i * n_pcs + 49] // PC50
    //    );
    //}


    // ---------------------------------------------------------
    // 7. 聚类 (KMeans Clustering) - 【新增部分】
    // ---------------------------------------------------------
    std::cout << "\nRunning KMeans Clustering..." << std::endl;

    int k = 10; // 设定聚类簇数
    int max_iter = 100;

    // 初始化 KMeans 对象
    KMeans kmeans(k, max_iter);

    // 运行聚类 (传入 embedding 向量 和 维度 n_pcs)
    std::vector<int> labels;

    benchmark("[6]KMeans Clustering", 1, [&]() {
        labels = kmeans.fit(embedding, n_pcs);
        });

    //saveLabels(output_dir + "kmeans_labels.csv", labels);
    printf("KMeans Clustering Done");

    // ---------------------------------------------------------
    // 8. 聚类结果验证
    // ---------------------------------------------------------
    //printf("First 20 labels: ");
    //for (int i = 0; i < 20; i++) {
    //    printf("%d ", labels[i]);
    //}
    //printf("\n");
    auto pipeline_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pipeline_elapsed = pipeline_end - pipeline_start;
    printf("   [ total time ] %.6fs\n", pipeline_elapsed);
    printf("Pipeline Finished");
    return 0;
}
