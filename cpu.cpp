#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

const int STATES = 3; // Number of hidden states: Match (M), Insert (I), Delete (D)
const int ALPHABETS = 5; // Number of visible alphabets: A, C, G, T, -

const double LOG_ZERO = -1e300; // A very small value to represent log(0)

// Function to initialize the emission matrix
void initializeEmissionMatrix(std::vector<std::vector<std::vector<double>>> &emissionMatrix) {
    emissionMatrix[0][0][0] = log(0.9);
    emissionMatrix[0][1][1] = log(0.8);
    emissionMatrix[0][2][2] = log(0.9);
    emissionMatrix[0][3][3] = log(0.7);
    emissionMatrix[0][4][4] = log(0.1);

    emissionMatrix[1][0][4] = log(0.1);
    emissionMatrix[1][1][4] = log(0.1);
    emissionMatrix[1][2][4] = log(0.1);
    emissionMatrix[1][3][4] = log(0.1);
    emissionMatrix[1][4][4] = log(0.6);

    emissionMatrix[2][4][0] = log(0.2);
    emissionMatrix[2][4][1] = log(0.2);
    emissionMatrix[2][4][2] = log(0.2);
    emissionMatrix[2][4][3] = log(0.2);
    emissionMatrix[2][4][4] = log(0.2);
}

// Function to initialize the transition matrix
void initializeTransitionMatrix(std::vector<std::vector<double>> &transitionMatrix) {
    transitionMatrix[0][0] = log(0.9);
    transitionMatrix[0][1] = log(0.1);
    transitionMatrix[0][2] = LOG_ZERO; // log(0) is -inf

    transitionMatrix[1][0] = log(0.1);
    transitionMatrix[1][1] = log(0.8);
    transitionMatrix[1][2] = log(0.1);

    transitionMatrix[2][0] = log(0.1);
    transitionMatrix[2][1] = LOG_ZERO; // log(0) is -inf
    transitionMatrix[2][2] = log(0.9);
}

// Log sum function to avoid underflow
inline double logSum(double logA, double logB) {
    if (logA == LOG_ZERO) return logB;
    if (logB == LOG_ZERO) return logA;
    if (logA > logB) {
        return logA + log(1 + exp(logB - logA));
    } else {
        return logB + log(1 + exp(logA - logB));
    }
}

// Function to calculate weighted emission probability in log scale
double weightedEmissionProbabilityLog(const std::vector<double> &readProb, int hapIndex, const std::vector<std::vector<std::vector<double>>> &emissionMatrix, int state) {
    double logProbability = LOG_ZERO;
    for (int i = 0; i < 4; ++i) {
        if (readProb[i] > 0) {
            logProbability = logSum(logProbability, log(readProb[i]) + emissionMatrix[state][i][hapIndex]);
        }
    }
    return logProbability;
}

double pairHMMForward(const std::vector<std::vector<double>> &readProbMatrix, const std::string &haplotype,
                      const std::vector<std::vector<std::vector<double>>> &emissionMatrix,
                      const std::vector<std::vector<double>> &transitionMatrix) {
    int m = readProbMatrix.size();
    int n = haplotype.size();

    // Initialize forward matrices in log scale
    std::vector<double> prevM(n + 1, LOG_ZERO), currM(n + 1, LOG_ZERO);
    std::vector<double> prevI(n + 1, LOG_ZERO), currI(n + 1, LOG_ZERO);
    std::vector<double> prevD(n + 1, LOG_ZERO), currD(n + 1, LOG_ZERO);

    // Initialize first row (free deletions)
    for (int j = 0; j <= n; ++j) {
        prevD[j] = log(1.0 / n);
    }

    // Fill in the matrices
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            // Convert haplotype character to index (A=0, C=1, G=2, T=3, -=4)
            int hapIndex = (haplotype[j - 1] == 'A') ? 0 : (haplotype[j - 1] == 'C') ? 1 : (haplotype[j - 1] == 'G') ? 2 : (haplotype[j - 1] == 'T') ? 3 : 4;

            double logEmissM = weightedEmissionProbabilityLog(readProbMatrix[i - 1], hapIndex, emissionMatrix, 0);
            double logEmissI = weightedEmissionProbabilityLog(readProbMatrix[i - 1], 4, emissionMatrix, 1);
            //double logEmissD = emissionMatrix[2][4][hapIndex];
            //printf("logEmissM[%d][%d] = %e, logEmissI[%d][%d] = %e, logEmissD[%d][%d] = %e\n",i, j, logEmissM, i, j, logEmissI, i, j, logEmissD);
            currM[j] = logEmissM + logSum(logSum(prevM[j - 1] + transitionMatrix[0][0],
                                                 prevI[j - 1] + transitionMatrix[1][0]),
                                          prevD[j - 1] + transitionMatrix[2][0]);

            currI[j] = logEmissI + logSum(prevM[j] + transitionMatrix[0][1],
                                          prevI[j] + transitionMatrix[1][1]);

            currD[j] =logSum(currM[j - 1] + transitionMatrix[0][2],
                                          currD[j - 1] + transitionMatrix[2][2]);
            //printf("dpM[%d][%d] = %e, dpI[%d][%d] = %e, dpD[%d][%d] = %e\n", i,j,currM[j],i,j ,currI[j],i,j,currD[j]);
            /*printf("dpM[%d][%d] = %e\n", i, j, currM[j]);
            printf("dpI[%d][%d] = %e\n", i, j, currI[j]);
            printf("dpD[%d][%d] = %e\n", i, j, currD[j]);*/
        }

        // Update previous column with current column for the next iteration
        std::swap(prevM, currM);
        std::swap(prevI, currI);
        std::swap(prevD, currD);

        // Reset current columns to LOG_ZERO
        std::fill(currM.begin(), currM.end(), LOG_ZERO);
        std::fill(currI.begin(), currI.end(), LOG_ZERO);
        std::fill(currD.begin(), currD.end(), LOG_ZERO);
    }
    
    // Calculate total likelihood in log scale
    double totalLogLikelihood = LOG_ZERO;
    for (int j = 1; j <= n; ++j) {
        totalLogLikelihood = logSum(totalLogLikelihood, logSum(prevM[j], prevI[j]));
    }

    return totalLogLikelihood;
}

int main() {
    std::vector<int> lengths = {100,1000,10000};  // Lengths to test
    std::vector<std::vector<double>> transitionMatrix(STATES, std::vector<double>(STATES));
    initializeTransitionMatrix(transitionMatrix);

    std::vector<std::vector<std::vector<double>>> emissionMatrix(STATES, std::vector<std::vector<double>>(ALPHABETS, std::vector<double>(ALPHABETS)));
    initializeEmissionMatrix(emissionMatrix);

    for (int len : lengths) {
        // Create a read probability matrix with uniform probabilities for simplicity
        std::vector<std::vector<double>> readProbMatrix(len, std::vector<double>(4, 0.25));
        std::string haplotype(len, 'A');  // Simplified haplotype, all As for simplicity
        /*for (int i = 0; i < readProbMatrix.size(); ++i) {
        for (int j = 0; j < readProbMatrix[i].size(); ++j) {
            printf("readProbMatrix[%d][%d]: %f\n", i, j, readProbMatrix[i][j]);
        }
    }
    
    for (int i = 0; i < emissionMatrix.size(); ++i) {
        for (int j = 0; j < emissionMatrix[i].size(); ++j) {
            for (int k = 0; k < emissionMatrix[i][j].size(); ++k) {
                printf("emissionMatrix[%d][%d][%d]: %f\n", i, j, k, emissionMatrix[i][j][k]);
            }
        }
    }*/

        auto start = std::chrono::high_resolution_clock::now();
        double logLikelihood = pairHMMForward(readProbMatrix, haplotype, emissionMatrix, transitionMatrix);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Length: " << len << ", Time: " << duration.count() << " seconds, Log-Likelihood: " << logLikelihood << std::endl;
    }

    return 0;
}
