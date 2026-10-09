#pragma once
#include <vector>

struct SparseMatrix {
    int n_rows = 0;
    int n_cols = 0;

    // 【修改点】从 int* / float* 改为 std::vector
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<float> values;

    // 这一行是为了兼容性，让外部如果需要直接获取指针时也能拿到
    // 虽然通常直接用 .data() 就行，但为了你不用改其他代码，可以不做额外操作
};