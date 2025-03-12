#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

const int STATES = 3;    // 隱藏狀態數：Match (M), Insert (I), Delete (D)
const int ALPHABETS = 5; // 可見字母數：A, C, G, T, -

const float LOG_ZERO = -1e30f; // 表示 log(0) 的一個非常小的值

// 初始化發射機率矩陣
void initializeEmissionMatrix(std::vector<std::vector<std::vector<float>>> &emissionMatrix) {
    emissionMatrix[0][0][0] = logf(0.9f);
    emissionMatrix[0][1][1] = logf(0.8f);
    emissionMatrix[0][2][2] = logf(0.9f);
    emissionMatrix[0][3][3] = logf(0.7f);
    emissionMatrix[0][4][4] = logf(0.1f);

    emissionMatrix[1][0][4] = logf(0.1f);
    emissionMatrix[1][1][4] = logf(0.1f);
    emissionMatrix[1][2][4] = logf(0.1f);
    emissionMatrix[1][3][4] = logf(0.1f);
    emissionMatrix[1][4][4] = logf(0.6f);

    emissionMatrix[2][4][0] = logf(0.2f);
    emissionMatrix[2][4][1] = logf(0.2f);
    emissionMatrix[2][4][2] = logf(0.2f);
    emissionMatrix[2][4][3] = logf(0.2f);
    emissionMatrix[2][4][4] = logf(0.2f);
}

// 初始化轉換矩陣
void initializeTransitionMatrix(std::vector<std::vector<float>> &transitionMatrix) {
    transitionMatrix[0][0] = logf(0.9f);
    transitionMatrix[0][1] = logf(0.1f);
    transitionMatrix[0][2] = LOG_ZERO; // log(0) 為 -inf

    transitionMatrix[1][0] = logf(0.1f);
    transitionMatrix[1][1] = logf(0.8f);
    transitionMatrix[1][2] = logf(0.1f);

    transitionMatrix[2][0] = logf(0.1f);
    transitionMatrix[2][1] = LOG_ZERO; // log(0) 為 -inf
    transitionMatrix[2][2] = logf(0.9f);
}

// log sum 避免 underflow
inline float logSum(float logA, float logB) {
    if (logA == LOG_ZERO) return logB;
    if (logB == LOG_ZERO) return logA;
    if (logA > logB) {
        return logA + logf(1 + expf(logB - logA));
    } else {
        return logB + logf(1 + expf(logA - logB));
    }
}

// 計算加權發射機率（對數尺度）
float weightedEmissionProbabilityLog(const std::vector<float> &readProb, int hapIndex,
                                       const std::vector<std::vector<std::vector<float>>> &emissionMatrix, int state) {
    float logProbability = LOG_ZERO;
    for (int i = 0; i < 4; ++i) {
        if (readProb[i] > 0) {
            logProbability = logSum(logProbability, logf(readProb[i]) + emissionMatrix[state][i][hapIndex]);
        }
    }
    return logProbability;
}

float pairHMMForward(const std::vector<std::vector<float>> &readProbMatrix, const std::string &haplotype,
                      const std::vector<std::vector<std::vector<float>>> &emissionMatrix,
                      const std::vector<std::vector<float>> &transitionMatrix) {
    int m = readProbMatrix.size();
    int n = haplotype.size();

    // 初始化前向矩陣（對數尺度）
    std::vector<float> prevM(n + 1, LOG_ZERO), currM(n + 1, LOG_ZERO);
    std::vector<float> prevI(n + 1, LOG_ZERO), currI(n + 1, LOG_ZERO);
    std::vector<float> prevD(n + 1, LOG_ZERO), currD(n + 1, LOG_ZERO);

    // 初始化第一列（free deletions）
    for (int j = 0; j <= n; ++j) {
        prevD[j] = logf(1.0f / n);
    }

    // 填充矩陣
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            // 將 haplotype 字元轉換為索引 (A=0, C=1, G=2, T=3, -=4)
            int hapIndex = (haplotype[j - 1] == 'A') ? 0 :
                           (haplotype[j - 1] == 'C') ? 1 :
                           (haplotype[j - 1] == 'G') ? 2 :
                           (haplotype[j - 1] == 'T') ? 3 : 4;

            float logEmissM = weightedEmissionProbabilityLog(readProbMatrix[i - 1], hapIndex, emissionMatrix, 0);
            float logEmissI = weightedEmissionProbabilityLog(readProbMatrix[i - 1], 4, emissionMatrix, 1);

            currM[j] = logEmissM + logSum(logSum(prevM[j - 1] + transitionMatrix[0][0],
                                                 prevI[j - 1] + transitionMatrix[1][0]),
                                          prevD[j - 1] + transitionMatrix[2][0]);

            currI[j] = logEmissI + logSum(prevM[j] + transitionMatrix[0][1],
                                          prevI[j] + transitionMatrix[1][1]);

            currD[j] = logSum(currM[j - 1] + transitionMatrix[0][2],
                              currD[j - 1] + transitionMatrix[2][2]);
        }

        // 更新上一列為當前列，進入下一輪迭代
        std::swap(prevM, currM);
        std::swap(prevI, currI);
        std::swap(prevD, currD);

        // 將當前列重置為 LOG_ZERO
        std::fill(currM.begin(), currM.end(), LOG_ZERO);
        std::fill(currI.begin(), currI.end(), LOG_ZERO);
        std::fill(currD.begin(), currD.end(), LOG_ZERO);
    }
    
    // 計算總對數機率
    float totalLogLikelihood = LOG_ZERO;
    for (int j = 1; j <= n; ++j) {
        totalLogLikelihood = logSum(totalLogLikelihood, logSum(prevM[j], prevI[j]));
    }

    return totalLogLikelihood;
}

int main() {
    std::vector<int> lengths = {100, 1000, 10000, 100000};  // 測試不同長度
    std::vector<std::vector<float>> transitionMatrix(STATES, std::vector<float>(STATES));
    initializeTransitionMatrix(transitionMatrix);

    std::vector<std::vector<std::vector<float>>> emissionMatrix(
        STATES, std::vector<std::vector<float>>(ALPHABETS, std::vector<float>(ALPHABETS)));
    initializeEmissionMatrix(emissionMatrix);

    for (int len : lengths) {
        // 建立 read probability matrix，簡單均分機率
        std::vector<std::vector<float>> readProbMatrix(len, std::vector<float>(4, 0.25f));
        std::string haplotype(len, 'A');  // 簡化 haplotype，全部為 'A'

        auto start = std::chrono::high_resolution_clock::now();
        float logLikelihood = pairHMMForward(readProbMatrix, haplotype, emissionMatrix, transitionMatrix);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;

        std::cout << "Length: " << len << ", Time: " << duration.count()
                  << " seconds, Log-Likelihood: " << logLikelihood << std::endl;
    }
    std::cout << "Press Enter to exit...";
    std::cin.get();
    return 0;
}
