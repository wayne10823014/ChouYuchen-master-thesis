#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

const int STATES = 3; // 隱藏狀態數量：比對 (M)、插入 (I)、刪除 (D)
const int ALPHABETS = 5; // 可見字母數量：A, C, G, T, -
#define LOG_ZERO -1e30f // 用來代表 log(0) 的極小值

__host__ int min(int a, int b, int c) {
    return std::min(std::min(a, b), c);
}

__device__ __host__ float logSum(float logA, float logB) {
    if (logA == LOG_ZERO) return logB;
    if (logB == LOG_ZERO) return logA;
    if (logA > logB) {
        return logA + logf(1 + expf(logB - logA));
    } else {
        return logB + logf(1 + expf(logA - logB));
    }
}

__device__ float weightedEmissionProbabilityLog(const float *readProb, int hapIndex, const float *emissionMatrix, int state) {
    float logProbability = LOG_ZERO;
    for (int i = 0; i < 4; ++i) {
        if (readProb[i] > 0) {
            logProbability = logSum(logProbability, logf(readProb[i]) + emissionMatrix[state * ALPHABETS * ALPHABETS + i * ALPHABETS + hapIndex]);
        }
    }
    return logProbability;
}

__global__ void pairHMMForwardKernel(const float *readProbMatrix, const char *haplotype, const float *emissionMatrix,
                                     const float *transitionMatrix, float *prevM, float *prevI, float *prevD, 
                                     float *currM, float *currI, float *currD, 
                                     float *newM, float *newI, float *newD, int len, int diag, float *dAns ){                                    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx <= diag) {
        int i = diag - idx;
        int j = idx;
        if (i > 0 && i <= len && j > 0 && j <= len) {
            int hapIndex = (haplotype[j - 1] == 'A') ? 0 :
                           (haplotype[j - 1] == 'C') ? 1 :
                           (haplotype[j - 1] == 'G') ? 2 :
                           (haplotype[j - 1] == 'T') ? 3 : 4;
            float logEmissM = weightedEmissionProbabilityLog(&readProbMatrix[(i - 1) * 4], hapIndex, emissionMatrix, 0);
            float logEmissI = weightedEmissionProbabilityLog(&readProbMatrix[(i - 1) * 4], 4, emissionMatrix, 1);
            __syncthreads();
            float newm = logEmissM + logSum(
                              logSum(prevM[(i - 1)] + transitionMatrix[0 * STATES + 0],
                                     prevI[(i - 1)] + transitionMatrix[1 * STATES + 0]),
                              prevD[(i - 1)] + transitionMatrix[2 * STATES + 0]);
            float newi = logEmissI + logSum(
                              currM[(i - 1)] + transitionMatrix[0 * STATES + 1],
                              currI[(i - 1)] + transitionMatrix[1 * STATES + 1]);
            float newd = logSum(
                              currM[i] + transitionMatrix[0 * STATES + 2],
                              currD[i] + transitionMatrix[2 * STATES + 2]);
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

void initializeEmissionMatrix(std::vector<float> &emissionMatrix) {
    emissionMatrix.resize(STATES * ALPHABETS * ALPHABETS);
    emissionMatrix[0 * ALPHABETS * ALPHABETS + 0 * ALPHABETS + 0] = logf(0.9f);  // M 發射 'A' -> 'A'
    emissionMatrix[0 * ALPHABETS * ALPHABETS + 1 * ALPHABETS + 1] = logf(0.8f);  // M 發射 'C' -> 'C'
    emissionMatrix[0 * ALPHABETS * ALPHABETS + 2 * ALPHABETS + 2] = logf(0.9f);  // M 發射 'G' -> 'G'
    emissionMatrix[0 * ALPHABETS * ALPHABETS + 3 * ALPHABETS + 3] = logf(0.7f);  // M 發射 'T' -> 'T'
    emissionMatrix[0 * ALPHABETS * ALPHABETS + 4 * ALPHABETS + 4] = logf(0.1f);  // M 發射 '-' -> '-'
    emissionMatrix[1 * ALPHABETS * ALPHABETS + 0 * ALPHABETS + 4] = logf(0.1f);  // I 發射 'A' -> '-'
    emissionMatrix[1 * ALPHABETS * ALPHABETS + 1 * ALPHABETS + 4] = logf(0.1f);  // I 發射 'C' -> '-'
    emissionMatrix[1 * ALPHABETS * ALPHABETS + 2 * ALPHABETS + 4] = logf(0.1f);  // I 發射 'G' -> '-'
    emissionMatrix[1 * ALPHABETS * ALPHABETS + 3 * ALPHABETS + 4] = logf(0.1f);  // I 發射 'T' -> '-'
    emissionMatrix[1 * ALPHABETS * ALPHABETS + 4 * ALPHABETS + 4] = logf(0.6f);  // I 發射 '-' -> '-'
    emissionMatrix[2 * ALPHABETS * ALPHABETS + 4 * ALPHABETS + 0] = logf(0.2f);  // D 發射 '-' -> 'A'
    emissionMatrix[2 * ALPHABETS * ALPHABETS + 4 * ALPHABETS + 1] = logf(0.2f);  // D 發射 '-' -> 'C'
    emissionMatrix[2 * ALPHABETS * ALPHABETS + 4 * ALPHABETS + 2] = logf(0.2f);  // D 發射 '-' -> 'G'
    emissionMatrix[2 * ALPHABETS * ALPHABETS + 4 * ALPHABETS + 3] = logf(0.2f);  // D 發射 '-' -> 'T'
    emissionMatrix[2 * ALPHABETS * ALPHABETS + 4 * ALPHABETS + 4] = logf(0.2f);  // D 發射 '-' -> '-'
}

void initializeTransitionMatrix(std::vector<float> &transitionMatrix) {
    transitionMatrix.resize(STATES * STATES);
    transitionMatrix[0 * STATES + 0] = logf(0.9f);  // M -> M
    transitionMatrix[0 * STATES + 1] = logf(0.1f);  // M -> I
    transitionMatrix[0 * STATES + 2] = LOG_ZERO;      // M -> D
    transitionMatrix[1 * STATES + 0] = logf(0.1f);  // I -> M
    transitionMatrix[1 * STATES + 1] = logf(0.8f);  // I -> I
    transitionMatrix[1 * STATES + 2] = logf(0.1f);  // I -> D
    transitionMatrix[2 * STATES + 0] = logf(0.1f);  // D -> M
    transitionMatrix[2 * STATES + 1] = LOG_ZERO;      // D -> I
    transitionMatrix[2 * STATES + 2] = logf(0.9f);  // D -> D
}

__global__ void initializeDPKernel(float* dpM1, float* dpI1, float* dpD1, 
                                   float* dpM2, float* dpI2, float* dpD2,
                                   float* dpM3, float* dpI3, float* dpD3, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = (len + 1);
    if (idx < totalSize) {
        if (idx == 0) {
            dpM1[idx] = LOG_ZERO;
            dpI1[idx] = LOG_ZERO;
            dpD1[idx] = logf(1.0f / len);
            dpM2[idx] = LOG_ZERO;
            dpI2[idx] = LOG_ZERO;
            dpD2[idx] = logf(1.0f / len);
            dpM3[idx] = LOG_ZERO;
            dpI3[idx] = LOG_ZERO;
            dpD3[idx] = logf(1.0f / len);
        } else {
            dpM1[idx] = LOG_ZERO;
            dpI1[idx] = LOG_ZERO;
            dpD1[idx] = LOG_ZERO;
            dpM2[idx] = LOG_ZERO;
            dpI2[idx] = LOG_ZERO;
            dpD2[idx] = LOG_ZERO;
            dpM3[idx] = LOG_ZERO;
            dpI3[idx] = LOG_ZERO;
            dpD3[idx] = LOG_ZERO;
        }
    }
}

__global__ void initializeAnsKernel(float* dAns) {
    *dAns = LOG_ZERO;
}

int main() {
    std::vector<int> lengths = {100, 1000, 10000, 100000};  // 測試不同序列長度
    std::vector<float> transitionMatrix;
    initializeTransitionMatrix(transitionMatrix);
    std::vector<float> emissionMatrix;
    initializeEmissionMatrix(emissionMatrix);
    for (int len : lengths) {
        // 建立讀取機率矩陣，為簡化將每個位置設為均等機率 0.25
        std::vector<std::vector<float>> readProbMatrix(len, std::vector<float>(4, 0.25f));
        std::string haplotype(len, 'A');  // 簡化 haplotype，全部為 'A'
        
        // 裝置端記憶體配置
        float *d_readProbMatrix, *d_emissionMatrix, *d_transitionMatrix;
        // 改用 double buffering 指標：三組 DP 陣列（prev, curr, new）
        float *d_prevM, *d_prevI, *d_prevD;
        float *d_currM, *d_currI, *d_currD;
        float *d_newM,  *d_newI,  *d_newD;
        char   *d_haplotype;
        float *d_ans;
        
        cudaMalloc((void**)&d_readProbMatrix, len * 4 * sizeof(float));
        cudaMalloc((void**)&d_emissionMatrix, STATES * ALPHABETS * ALPHABETS * sizeof(float));
        cudaMalloc((void**)&d_transitionMatrix, STATES * STATES * sizeof(float));
        cudaMalloc((void**)&d_prevM, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_prevI, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_prevD, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_currM, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_currI, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_currD, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_newM, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_newI, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_newD, (len + 1) * sizeof(float));
        cudaMalloc((void**)&d_haplotype, len * sizeof(char));
        cudaMalloc((void**)&d_ans, sizeof(float));

        // 將 readProbMatrix 攤平成一維陣列後複製到裝置端
        std::vector<float> flatReadProbMatrix(len * 4);
        for (int i = 0; i < len; ++i) {
            for (int j = 0; j < 4; ++j) {
                flatReadProbMatrix[i * 4 + j] = readProbMatrix[i][j];
            }
        }
        cudaMemcpy(d_readProbMatrix, flatReadProbMatrix.data(), len * 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_emissionMatrix, emissionMatrix.data(), STATES * ALPHABETS * ALPHABETS * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_transitionMatrix, transitionMatrix.data(), STATES * STATES * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_haplotype, haplotype.c_str(), len * sizeof(char), cudaMemcpyHostToDevice);
        
        // 初始化答案
        initializeAnsKernel<<<1, 1>>>(d_ans);
  
        // 初始化 DP 陣列 (三組)
        int totalSize = (len + 1);
        int threadsPerBlock = 128;
        int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
        initializeDPKernel<<<blocksPerGrid, threadsPerBlock>>>(d_prevM, d_prevI, d_prevD, 
                                                               d_currM, d_currI, d_currD, 
                                                               d_newM, d_newI, d_newD, len);
        cudaDeviceSynchronize();

        // 設定對角線 kernel 的執行參數
        int diagThreadsPerBlock = 64;
        auto start = std::chrono::high_resolution_clock::now();
        for (int diag = 1; diag <= 2 * len; ++diag) {
            int elementsInDiag = diag;
            int diagBlocksPerGrid = (elementsInDiag + diagThreadsPerBlock - 1) / diagThreadsPerBlock;
            pairHMMForwardKernel<<<diagBlocksPerGrid, diagThreadsPerBlock>>>(d_readProbMatrix, d_haplotype, d_emissionMatrix,
                                                                             d_transitionMatrix, 
                                                                             d_prevM, d_prevI, d_prevD, 
                                                                             d_currM, d_currI, d_currD, 
                                                                             d_newM, d_newI, d_newD, 
                                                                             len, diag, d_ans);
            cudaDeviceSynchronize();
            // 利用指標交換取代原本的 cudaMemcpyDeviceToDevice 複製
            float *temp;
            temp = d_prevM; d_prevM = d_currM; d_currM = d_newM; d_newM = temp;
            temp = d_prevI; d_prevI = d_currI; d_currI = d_newI; d_newI = temp;
            temp = d_prevD; d_prevD = d_currD; d_currD = d_newD; d_newD = temp;
        }
        float ans;
        cudaMemcpy(&ans, d_ans, sizeof(float), cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        std::cout << "Length: " << len << ", Time: " << duration.count() << " seconds, Log-Likelihood: " << ans << std::endl;
        // 釋放裝置端記憶體
        cudaFree(d_readProbMatrix);
        cudaFree(d_emissionMatrix);
        cudaFree(d_transitionMatrix);
        cudaFree(d_prevM);
        cudaFree(d_prevI);
        cudaFree(d_prevD);
        cudaFree(d_currM);
        cudaFree(d_currI);
        cudaFree(d_currD);
        cudaFree(d_newM);
        cudaFree(d_newI);
        cudaFree(d_newD);
        cudaFree(d_haplotype);
        cudaFree(d_ans);
    }
    return 0;
}
