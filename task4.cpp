#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <stdlib.h>
#include <sys/resource.h>
#include <iomanip>

using namespace std;

void Multiply_seq(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


void Multiply_veq(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t c = vld1q_f32(&C[i * N + j]);
            for (int k = 0; k < N; k++) {
                float32x4_t b = vld1q_f32(&B[k * N + j]);
                float32x4_t a = vdupq_n_f32(A[i * N + k]);
                c = vmlaq_f32(c, a, b);
            }
            vst1q_f32(&C[i * N + j], c);
        }
    }
}


void fill_matrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


int main() {
    vector<int> sizes = {512, 1024, 2048};

    for (int N : sizes) {
        vector<float> A(N * N), B(N * N), C_seq(N * N), C_veq(N * N);

        fill_matrix(A.data(), N);
        fill_matrix(B.data(), N);

        fill(C_seq.begin(), C_seq.end(), 0.0f);
        fill(C_veq.begin(), C_veq.end(), 0.0f);

        cout << "Multiply_seq for N = " << N << ": ";
        clock_t start_time = clock();
        Multiply_seq(A.data(), B.data(), C_seq.data(), N);
        clock_t end_time = clock();
        cout << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << " seconds" << endl;

        cout << "Multiply_veq for N = " << N << ": ";
        start_time = clock();
        Multiply_veq(A.data(), B.data(), C_veq.data(), N);
        end_time = clock();
        cout << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC << " seconds" << endl;

    }

    return 0;
}