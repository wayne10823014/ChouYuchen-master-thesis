#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>        // 為了 std::min
#include <cuda.h>
#include <cuda_runtime.h>

const int STATES    = 3;    // 隱藏狀態數：Match (M), Insert (I), Delete (D)
const int ALPHABETS = 5;    // 可見字母數：A, C, G, T, -
#define LOG_ZERO -1e30f     // 表示 log(0) 的非常小值

// 三數最小值
__host__ int min3(int a, int b, int c) {
    return std::min(std::min(a, b), c);
}

// 安全版 log-sum
__device__ __host__ float logSum(float logA, float logB) {
    if (logA == LOG_ZERO) return logB;
    if (logB == LOG_ZERO) return logA;
    if (logA > logB) {
        return logA + logf(1 + expf(logB - logA));
    } else {
        return logB + logf(1 + expf(logA - logB));
    }
}

// 計算 weighted emission probability (log)
__device__ float weightedEmissionProbabilityLog(
    const float *readProb, int hapIndex,
    const float *emissionMatrix, int state
) {
    float logProb = LOG_ZERO;
    for (int i = 0; i < 4; ++i) {
        if (readProb[i] > 0) {
            logProb = logSum(
                logProb,
                logf(readProb[i]) +
                    emissionMatrix[state * ALPHABETS * ALPHABETS
                                 + i * ALPHABETS + hapIndex]
            );
        }
    }
    return logProb;
}

// 核心 Kernel：對角線計算
__global__ void pairHMMForwardKernel(
    const float *readProbMatrix,
    const char  *haplotype,
    const float *emissionMatrix,
    const float *transitionMatrix,
    float *prevM, float *prevI, float *prevD,
    float *currM, float *currI, float *currD,
    float *newM,  float *newI,  float *newD,
    int len, int diag, float *dAns
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx <= diag) {
        int i = diag - idx;
        int j = idx;
        if (i > 0 && i <= len && j > 0 && j <= len) {
            int hapIndex =
                (haplotype[j - 1] == 'A') ? 0 :
                (haplotype[j - 1] == 'C') ? 1 :
                (haplotype[j - 1] == 'G') ? 2 :
                (haplotype[j - 1] == 'T') ? 3 : 4;

            float logEmissM = weightedEmissionProbabilityLog(
                &readProbMatrix[(i - 1) * 4], hapIndex, emissionMatrix, 0);
            float logEmissI = weightedEmissionProbabilityLog(
                &readProbMatrix[(i - 1) * 4], 4, emissionMatrix, 1);

            __syncthreads();

            float newm = logEmissM + logSum(
                logSum(prevM[i - 1] + transitionMatrix[0 * STATES + 0],
                       prevI[i - 1] + transitionMatrix[1 * STATES + 0]),
                prevD[i - 1] + transitionMatrix[2 * STATES + 0]
            );

            float newi = logEmissI + logSum(
                currM[i - 1] + transitionMatrix[0 * STATES + 1],
                currI[i - 1] + transitionMatrix[1 * STATES + 1]
            );

            float newd = logSum(
                currM[i] + transitionMatrix[0 * STATES + 2],
                currD[i] + transitionMatrix[2 * STATES + 2]
            );

            __syncthreads();

            newM[i] = newm;
            newI[i] = newi;
            newD[i] = newd;

            if (i == len) {
                *dAns = logSum(*dAns, logSum(newm, newi));
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}

// 初始化 DP buffers
__global__ void initializeDPKernel(
    float* dpM1, float* dpI1, float* dpD1,
    float* dpM2, float* dpI2, float* dpD2,
    float* dpM3, float* dpI3, float* dpD3,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > len) return;
    if (idx == 0) {
        dpM1[0] = LOG_ZERO; dpI1[0] = LOG_ZERO; dpD1[0] = logf(1.0f / len);
        dpM2[0] = LOG_ZERO; dpI2[0] = LOG_ZERO; dpD2[0] = logf(1.0f / len);
        dpM3[0] = LOG_ZERO; dpI3[0] = LOG_ZERO; dpD3[0] = logf(1.0f / len);
    } else {
        dpM1[idx] = dpI1[idx] = dpD1[idx] =
        dpM2[idx] = dpI2[idx] = dpD2[idx] =
        dpM3[idx] = dpI3[idx] = dpD3[idx] = LOG_ZERO;
    }
}

// 初始化答案
__global__ void initializeAnsKernel(float* dAns) {
    *dAns = LOG_ZERO;
}

// 產生 transition matrix
void initializeTransitionMatrix(std::vector<float> &transitionMatrix) {
    transitionMatrix.resize(STATES * STATES);
    transitionMatrix[0*STATES+0] = logf(0.9f);
    transitionMatrix[0*STATES+1] = logf(0.1f);
    transitionMatrix[0*STATES+2] = LOG_ZERO;
    transitionMatrix[1*STATES+0] = logf(0.1f);
    transitionMatrix[1*STATES+1] = logf(0.8f);
    transitionMatrix[1*STATES+2] = logf(0.1f);
    transitionMatrix[2*STATES+0] = logf(0.1f);
    transitionMatrix[2*STATES+1] = LOG_ZERO;
    transitionMatrix[2*STATES+2] = logf(0.9f);
}

// 產生 emission matrix
void initializeEmissionMatrix(std::vector<float> &emissionMatrix) {
    emissionMatrix.resize(STATES * ALPHABETS * ALPHABETS);
    // M
    emissionMatrix[0*ALPHABETS*ALPHABETS + 0*ALPHABETS + 0] = logf(0.9f);
    emissionMatrix[0*ALPHABETS*ALPHABETS + 1*ALPHABETS + 1] = logf(0.8f);
    emissionMatrix[0*ALPHABETS*ALPHABETS + 2*ALPHABETS + 2] = logf(0.9f);
    emissionMatrix[0*ALPHABETS*ALPHABETS + 3*ALPHABETS + 3] = logf(0.7f);
    emissionMatrix[0*ALPHABETS*ALPHABETS + 4*ALPHABETS + 4] = logf(0.1f);
    // I
    emissionMatrix[1*ALPHABETS*ALPHABETS + 0*ALPHABETS + 4] = logf(0.1f);
    emissionMatrix[1*ALPHABETS*ALPHABETS + 1*ALPHABETS + 4] = logf(0.1f);
    emissionMatrix[1*ALPHABETS*ALPHABETS + 2*ALPHABETS + 4] = logf(0.1f);
    emissionMatrix[1*ALPHABETS*ALPHABETS + 3*ALPHABETS + 4] = logf(0.1f);
    emissionMatrix[1*ALPHABETS*ALPHABETS + 4*ALPHABETS + 4] = logf(0.6f);
    // D
    emissionMatrix[2*ALPHABETS*ALPHABETS + 4*ALPHABETS + 0] = logf(0.2f);
    emissionMatrix[2*ALPHABETS*ALPHABETS + 4*ALPHABETS + 1] = logf(0.2f);
    emissionMatrix[2*ALPHABETS*ALPHABETS + 4*ALPHABETS + 2] = logf(0.2f);
    emissionMatrix[2*ALPHABETS*ALPHABETS + 4*ALPHABETS + 3] = logf(0.2f);
    emissionMatrix[2*ALPHABETS*ALPHABETS + 4*ALPHABETS + 4] = logf(0.2f);
}

int main() {
    std::vector<int> lengths = {100, 1000, 10000, 100000, 1000000};
    std::vector<float> transitionMatrix, emissionMatrix;
    initializeTransitionMatrix(transitionMatrix);
    initializeEmissionMatrix(emissionMatrix);

    for (int len : lengths) {
        // 準備 host 資料
        std::vector<std::vector<float>> readProb(len, std::vector<float>(4, 0.25f));
        std::string hap(len, 'A');

        // GPU 記憶體配置
        float *d_read, *d_emit, *d_trans;
        float *d_prevM, *d_prevI, *d_prevD;
        float *d_currM, *d_currI, *d_currD;
        float *d_newM,  *d_newI,  *d_newD;
        float *d_ans;
        char  *d_hap;
        size_t szF = len * 4 * sizeof(float);
        size_t szV = (len + 1) * sizeof(float);

        cudaMalloc(&d_read, szF);
        cudaMalloc(&d_emit, STATES * ALPHABETS * ALPHABETS * sizeof(float));
        cudaMalloc(&d_trans, STATES * STATES * sizeof(float));

        cudaMalloc(&d_prevM, szV); cudaMalloc(&d_prevI, szV); cudaMalloc(&d_prevD, szV);
        cudaMalloc(&d_currM, szV); cudaMalloc(&d_currI, szV); cudaMalloc(&d_currD, szV);
        cudaMalloc(&d_newM,  szV); cudaMalloc(&d_newI,  szV); cudaMalloc(&d_newD,  szV);

        cudaMalloc(&d_hap, len * sizeof(char));
        cudaMalloc(&d_ans, sizeof(float));

        // 複製 host→device
        std::vector<float> flat(len * 4);
        for (int i = 0; i < len; ++i)
            for (int j = 0; j < 4; ++j)
                flat[i * 4 + j] = readProb[i][j];

        cudaMemcpy(d_read, flat.data(), szF, cudaMemcpyHostToDevice);
        cudaMemcpy(d_emit, emissionMatrix.data(),
                   STATES * ALPHABETS * ALPHABETS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_trans, transitionMatrix.data(),
                   STATES * STATES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hap, hap.c_str(), len * sizeof(char), cudaMemcpyHostToDevice);

        // 初始化 d_ans、DP arrays
        initializeAnsKernel<<<1,1>>>(d_ans);
        int threads = 128, blocks = (len + 1 + threads - 1) / threads;
        initializeDPKernel<<<blocks, threads>>>(
            d_prevM, d_prevI, d_prevD,
            d_currM, d_currI, d_currD,
            d_newM,  d_newI,  d_newD,
            len
        );
        cudaDeviceSynchronize();

        // 逐對角線計算
        auto t0 = std::chrono::high_resolution_clock::now();
        int diagThreads = 64;
        for (int diag = 1; diag <= 2 * len; ++diag) {
            int elems      = diag;
            int diagBlocks = (elems + diagThreads - 1) / diagThreads;
            pairHMMForwardKernel<<<diagBlocks, diagThreads>>>(
                d_read, d_hap, d_emit, d_trans,
                d_prevM, d_prevI, d_prevD,
                d_currM, d_currI, d_currD,
                d_newM,  d_newI,  d_newD,
                len, diag, d_ans
            );
            cudaDeviceSynchronize();

            // ==== 四行指標輪替 ====
            float* tmpM = d_prevM;
            d_prevM = d_currM;
            d_currM = d_newM;
            d_newM  = tmpM;

            float* tmpI = d_prevI;
            d_prevI = d_currI;
            d_currI = d_newI;
            d_newI  = tmpI;

            float* tmpD = d_prevD;
            d_prevD = d_currD;
            d_currD = d_newD;
            d_newD  = tmpD;
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        // 拷貝結果回主機並輸出
        float ans;
        cudaMemcpy(&ans, d_ans, sizeof(float), cudaMemcpyDeviceToHost);
        std::chrono::duration<float> dt = t1 - t0;
        std::cout << "Length: " << len
                  << ", Time: " << dt.count()
                  << " seconds, Log-Likelihood: " << ans << std::endl;

        // 釋放資源
        cudaFree(d_read);   cudaFree(d_emit);  cudaFree(d_trans);
        cudaFree(d_prevM);  cudaFree(d_prevI); cudaFree(d_prevD);
        cudaFree(d_currM);  cudaFree(d_currI); cudaFree(d_currD);
        cudaFree(d_newM);   cudaFree(d_newI);  cudaFree(d_newD);
        cudaFree(d_hap);    cudaFree(d_ans);
    }

    return 0;
}
