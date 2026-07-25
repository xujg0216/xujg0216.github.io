
# faiss-c 安装

FAISS (Facebook AI Similarity Search) 是一个高效的相似性搜索库，广泛用于向量搜索和聚类任务。FAISS 提供了 C++ 和 Python 的接口。以下是如何在 C++ 项目中使用 FAISS 作为第三方库，并编译它的方法。

首先，你需要在你的系统中安装 FAISS。FAISS 可以通过源码编译安装，也可以通过包管理器（如 conda）安装。下面是从源码编译的步骤：

### 1. 克隆FAISS仓库

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss

```

也可以从[https://github.com/facebookresearch/faiss/releases](https://github.com/facebookresearch/faiss/releases)下载release版本

### 2. 编译和安装

```bash
cmake -B build -DBUILD_SHARED_LIBS=ON.
cmake --build build -j4
```
> 升级cmake
> [https://github.com/Kitware/CMake/releases](https://github.com/Kitware/CMake/releases)
> 下载sh版本，直接安装
> 临时用新的cmake
> `export PATH=/home/xujg/cmake-3.26.6-linux-aarch64/bin:$PATH`


### 3. 动态库
编译生成的动态库在
`./build/faiss/libfaiss.so`


示例代码，调用生成的动态库
```c
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>

// 64-bit int
using idx_t = faiss::idx_t;

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    faiss::IndexFlatL2 index(d); // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb); // add vectors to the index
    printf("ntotal = %zd\n", index.ntotal);

    int k = 4;

    { // sanity check: search 5 first vectors of xb
        idx_t* I = new idx_t[k * 5];
        float* D = new float[k * 5];

        index.search(5, xb, k, D, I);

        // print results
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
```

编译生成
```bash
g++ postprocess.cc -L/home/xujg/code/RKNN_C_API/3rdparty/faiss-1.8.0/build/faiss -lfaiss -o test
```
!!!note
    如果直接运行可执行文件，无法找到 libfaiss.so 共享库。尽管在编译时已经指定了库的路径，但在执行时，系统还需要知道该库的路径。最简单的方法是将 libfaiss.so复制到/usr/lib/文件下

