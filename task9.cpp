#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <algorithm>
#include <cstdlib>

using namespace std;

static void checkMultiplication(const vector<int>& A, const vector<int>& B, vector<int>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum_val = 0;
            for (int k = 0; k < n; ++k) {
                sum_val += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum_val;
        }
    }
}

static void initRandMatrix(vector<int>& M) {
    int dim = (int) sqrt(M.size());
    random_device seed;
    mt19937 gen(seed());
    uniform_int_distribution<> dist(0, 5);
    for (int i = 0; i < dim * dim; ++i) {
        M[i] = dist(gen);
    }
}

static void parallelBlockProduct(vector<int>& A, vector<int>& B, vector<int>& C,
                                 vector<int>& Atemp, vector<int>& Btemp,
                                 int smallBlockDim, int blockDim,
                                 int myProcRow, int myProcCol, int sqrtProcs,
                                 MPI_Comm colComm, MPI_Comm rowComm) {
    for (int offset = 0; offset < blockDim; offset += smallBlockDim) {
        for (int x = 0; x < smallBlockDim; x++) {
            for (int y = 0; y < blockDim; y++) {
                Btemp[(x + smallBlockDim * myProcRow) * blockDim + y] = B[(x + offset) * blockDim + y];
                Atemp[(x + smallBlockDim * myProcCol) * blockDim + y] = A[y * blockDim + (x + offset)];
            }
        }

        for (int c_block = 0; c_block < sqrtProcs; c_block++) {
            for (int seg = 0; seg < smallBlockDim; seg++) {
                MPI_Bcast(&Btemp[(seg + smallBlockDim * c_block) * blockDim], blockDim, MPI_INT, c_block, colComm);
            }
        }

        for (int r_block = 0; r_block < sqrtProcs; r_block++) {
            for (int seg = 0; seg < smallBlockDim; seg++) {
                MPI_Bcast(&Atemp[(seg + smallBlockDim * r_block) * blockDim], blockDim, MPI_INT, r_block, rowComm);
            }
        }

        for (int i = 0; i < blockDim; i++) {
            for (int j = 0; j < blockDim; j++) {
                int part_sum = 0;
                for (int k = 0; k < smallBlockDim * sqrtProcs; k++) {
                    part_sum += Atemp[k * blockDim + i] * Btemp[k * blockDim + j];
                }
                C[i * blockDim + j] += part_sum;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int worldProcs, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldProcs);

    int dimN = atoi(argv[1]);
    int smallBlockDim = (int) sqrt(dimN);
    int sqrtProcs = (int) sqrt(worldProcs);
    int blockDim = dimN / sqrtProcs;

    vector<int> A(blockDim * blockDim), B(blockDim * blockDim), C(blockDim * blockDim, 0);

    initRandMatrix(A);
    initRandMatrix(B);

    double startTime = MPI_Wtime();

    MPI_Comm gridComm, rowComm, colComm;
    int gridDims[2] = { sqrtProcs, sqrtProcs };
    int periods[2] = { 0, 0 };
    MPI_Cart_create(MPI_COMM_WORLD, 2, gridDims, periods, 0, &gridComm);

    int coords[2];
    MPI_Cart_coords(gridComm, rank, 2, coords);
    int myProcRow = coords[0];
    int myProcCol = coords[1];

    int rowSub[2] = {0, 1};
    int colSub[2] = {1, 0};

    MPI_Cart_sub(gridComm, rowSub, &rowComm);
    MPI_Cart_sub(gridComm, colSub, &colComm);

    vector<int> Atemp(smallBlockDim * sqrtProcs * blockDim, 0);
    vector<int> Btemp(smallBlockDim * sqrtProcs * blockDim, 0);

    parallelBlockProduct(A, B, C, Atemp, Btemp, smallBlockDim, blockDim, myProcRow, myProcCol, sqrtProcs, colComm, rowComm);

    double endTime = MPI_Wtime();
    double localTime = endTime - startTime;
    double maxTime;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, gridComm);

    vector<int> gatherA, gatherB, gatherC;
    if (rank == 0) {
        gatherA.resize(worldProcs * blockDim * blockDim);
        gatherB.resize(worldProcs * blockDim * blockDim);
        gatherC.resize(worldProcs * blockDim * blockDim);
    }

    MPI_Gather(A.data(), blockDim * blockDim, MPI_INT,
               gatherA.data(), blockDim * blockDim, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(B.data(), blockDim * blockDim, MPI_INT,
               gatherB.data(), blockDim * blockDim, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(C.data(), blockDim * blockDim, MPI_INT,
               gatherC.data(), blockDim * blockDim, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<int> finalA(dimN * dimN), finalB(dimN * dimN), finalC(dimN * dimN);

        for (int p = 0; p < worldProcs; p++) {
            int rr = p / sqrtProcs;
            int cc = p % sqrtProcs;
            for (int i = 0; i < blockDim; i++) {
                for (int j = 0; j < blockDim; j++) {
                    finalA[(rr * blockDim + i) * dimN + cc * blockDim + j] = gatherA[p * blockDim * blockDim + i * blockDim + j];
                    finalB[(rr * blockDim + i) * dimN + cc * blockDim + j] = gatherB[p * blockDim * blockDim + i * blockDim + j];
                    finalC[(rr * blockDim + i) * dimN + cc * blockDim + j] = gatherC[p * blockDim * blockDim + i * blockDim + j];
                }
            }
        }

        cout << "Максимальное время: " << maxTime << " c\n";

        vector<int> checkRes(dimN * dimN);

        checkMultiplication(finalA, finalB, checkRes, dimN);

        cout << ((finalC == checkRes) ? "Результат корректен" : "Есть ошибка!") << "\n";
    }

    MPI_Finalize();
    return 0;
}
