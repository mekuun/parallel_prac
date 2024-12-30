#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#define N 128
#define MAX_ITER 1000
#define EPSILON 1e-5

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    int i, j, iter;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) printf("Размер сетки должен быть кратен числу процессов.\n");
        MPI_Finalize();
        return -1;
    }

    int local_N = N / size;
    double* f = (double*) malloc((local_N + 2) * N * sizeof(double));
    double* f_new = (double*) malloc((local_N + 2) * N * sizeof(double));

    srand(time(NULL) + rank);

    for (i = 1; i <= local_N; i++) {
        for (j = 0; j < N; j++) {
            f[i * N + j] = rand() / (double)RAND_MAX;
            f_new[i * N + j] = f[i * N + j];
        }
    }

    double start_time = MPI_Wtime();
    double max_diff, global_diff;
    double* temp;

    for (iter = 0; iter < MAX_ITER; iter++) {
        MPI_Status status;

        if (size > 1) {
            // Передача данных между соседними процессами
            if (rank < size - 1) {
                MPI_Sendrecv(f + local_N * N, N, MPI_DOUBLE, rank + 1, 0,
                             f + (local_N + 1) * N, N, MPI_DOUBLE, rank + 1, 0,
                             MPI_COMM_WORLD, &status);
            }
            if (rank > 0) {
                MPI_Sendrecv(f + N, N, MPI_DOUBLE, rank - 1, 0,
                             f, N, MPI_DOUBLE, rank - 1, 0,
                             MPI_COMM_WORLD, &status);
            }
        }

        max_diff = 0.0;

        for (j = 1; j < N - 1; j++) {
            for (i = 1; i <= local_N; i++) {
                f_new[i * N + j] = 0.25 * (
                        f[(i - 1) * N + j] + f[(i + 1) * N + j] +
                        f[i * N + (j - 1)] + f[i * N + (j + 1)]
                );
                double diff = fabs(f_new[i * N + j] - f[i * N + j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        temp = f;
        f = f_new;
        f_new = temp;

        MPI_Allreduce(&max_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global_diff < EPSILON) {
            if (rank == 0) printf("Решение сошлось за %d итераций.\n", iter + 1);
            break;
        }
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    double max_elapsed_time;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double* full_f = NULL;
    if (rank == 0) {
        full_f = (double*) malloc(N * N * sizeof(double));
    }

    MPI_Gather(f + N, local_N * N, MPI_DOUBLE, full_f, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Решение:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%f ", full_f[i * N + j]);
            }
            printf("\n");
        }
        printf("Время выполнения: %f секунд.\n", max_elapsed_time);
        free(full_f);
    }

    free(f);
    free(f_new);
    MPI_Finalize();
    return 0;
}
