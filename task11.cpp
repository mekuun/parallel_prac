#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define N 20000

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dim[2];
    switch (size) {
        case 1:   dim[0] = 1; dim[1] = 1; break;
        case 2:   dim[0] = 2; dim[1] = 1; break;
        case 4:   dim[0] = 2; dim[1] = 2; break;
        case 8:   dim[0] = 4; dim[1] = 2; break;
        case 12:  dim[0] = 4; dim[1] = 3; break;
        case 16:  dim[0] = 4; dim[1] = 4; break;
    }

    int row = rank / dim[1];
    int col = rank % dim[1];

    int Ny = N / dim[0];
    int Nx = N / dim[1];

    double* A = (double*)malloc(Ny * Nx * sizeof(double));
    double* c = (double*)calloc(Ny, sizeof(double));

    srand((unsigned int)(time(NULL) + rank * 12345));

    for (int i = 0; i < Ny * Nx; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }

    double* win_buf = NULL;
    MPI_Win win;
    MPI_Win_allocate(2 * N * sizeof(double), sizeof(double),
                     MPI_INFO_NULL, MPI_COMM_WORLD, &win_buf, &win);

    double* b_global = win_buf;
    double* c_global = win_buf + N;

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            b_global[i] = (double)rand() / RAND_MAX;
        }
        for (int i = 0; i < N; i++) {
            c_global[i] = 0.0;
        }
    }

    MPI_Win_fence(0, win);

    double t0 = MPI_Wtime();

    double* b_local = (double*)malloc(N * sizeof(double));

    MPI_Get(b_local, N, MPI_DOUBLE, 0, 0, N, MPI_DOUBLE, win);
    MPI_Win_fence(0, win);

    for (int i = 0; i < Ny; i++) {
        int global_i = row * Ny + i;
        double sum = 0.0;
        for (int j = 0; j < Nx; j++) {
            int global_j = col * Nx + j;
            sum += A[i * Nx + j] * b_local[global_j];
        }
        c[i] = sum;
    }

    MPI_Win_fence(0, win);

    for (int i = 0; i < Ny; i++) {
        int global_i = row * Ny + i;
        MPI_Accumulate(&c[i], 1, MPI_DOUBLE, 0, (N + global_i), 1, MPI_DOUBLE, MPI_SUM, win);
    }

    MPI_Win_fence(0, win);

    double t = MPI_Wtime() - t0;

    if (rank == 0) {
        printf("Время (P=%d): %lf секунд\n", size, t);

    }

    free(b_local);
    free(A);
    free(c);

    MPI_Win_free(&win);

    MPI_Finalize();
    return 0;
}
