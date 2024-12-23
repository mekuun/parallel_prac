#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define N 48

using namespace std;

void initialize_grid(vector<double> &grid, int local_size) {
    srand(time(0) + grid.size());
    for (int i = 0; i < local_size; ++i) {
        grid[i] = rand() / (double)RAND_MAX;
    }
}

//создали срезы
void create_derived_types(int Nx, int Ny, int Nz, MPI_Datatype &yz_plane, MPI_Datatype &xz_plane, MPI_Datatype &xy_plane) {
    MPI_Type_vector(Ny * Nz, 1, 1, MPI_DOUBLE, &yz_plane);
    MPI_Type_commit(&yz_plane);

    MPI_Type_vector(Nx, Nz, Ny * Nz, MPI_DOUBLE, &xz_plane);
    MPI_Type_commit(&xz_plane);

    MPI_Type_vector(Nx * Ny, 1, Nz, MPI_DOUBLE, &xy_plane);
    MPI_Type_commit(&xy_plane);
}

void jacobi_iteration(const vector<double> &old_grid, vector<double> &new_grid, int Nx, int Ny, int Nz) {
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int k = 1; k < Nz - 1; ++k) {
                int idx = i * Ny * Nz + j * Nz + k;
                new_grid[idx] = (old_grid[(i - 1) * Ny * Nz + j * Nz + k] +
                        old_grid[(i + 1) * Ny * Nz + j * Nz + k] +
                        old_grid[i * Ny * Nz + (j - 1) * Nz + k] +
                        old_grid[i * Ny * Nz + (j + 1) * Nz + k] +
                        old_grid[i * Ny * Nz + j * Nz + (k - 1)] +
                        old_grid[i * Ny * Nz + j * Nz + (k + 1)]) / 6.0;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);
    int periods[3] = {0, 0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int Nx = N / dims[0];
    int Ny = N / dims[1];
    int Nz = N / dims[2];

    if (N % dims[0] || N % dims[1] || N % dims[2]) {
        if (rank == 0) {
            cerr << "Размер сетки должен делиться на топологию процессов!" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    vector<double> old_grid(Nx * Ny * Nz, 0.0);
    vector<double> new_grid(Nx * Ny * Nz, 0.0);
    initialize_grid(old_grid, Nx * Ny * Nz);

    int neighbors[6];
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[0], &neighbors[1]); // X
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[2], &neighbors[3]); // Y
    MPI_Cart_shift(cart_comm, 2, 1, &neighbors[4], &neighbors[5]); // Z

    MPI_Datatype yz_plane, xz_plane, xy_plane;
    create_derived_types(Nx, Ny, Nz, yz_plane, xz_plane, xy_plane);

    double start_time = MPI_Wtime();

    for (int iter = 0; iter < 100; ++iter) {
        MPI_Request requests[12];
        int req_count = 0;

        // по X
        if (neighbors[0] != MPI_PROC_NULL) {
            MPI_Isend(&old_grid[0], 1, yz_plane, neighbors[0], 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(&old_grid[0], 1, yz_plane, neighbors[0], 1, cart_comm, &requests[req_count++]);
        }
        if (neighbors[1] != MPI_PROC_NULL) {
            MPI_Isend(&old_grid[(Nx - 1) * Ny * Nz], 1, yz_plane, neighbors[1], 1, cart_comm, &requests[req_count++]);
            MPI_Irecv(&old_grid[(Nx - 1) * Ny * Nz], 1, yz_plane, neighbors[1], 0, cart_comm, &requests[req_count++]);
        }

        // по Y
        if (neighbors[2] != MPI_PROC_NULL) {
            MPI_Isend(&old_grid[0], 1, xz_plane, neighbors[2], 2, cart_comm, &requests[req_count++]);
            MPI_Irecv(&old_grid[0], 1, xz_plane, neighbors[2], 3, cart_comm, &requests[req_count++]);
        }
        if (neighbors[3] != MPI_PROC_NULL) {
            MPI_Isend(&old_grid[(Ny - 1) * Nz], 1, xz_plane, neighbors[3], 3, cart_comm, &requests[req_count++]);
            MPI_Irecv(&old_grid[(Ny - 1) * Nz], 1, xz_plane, neighbors[3], 2, cart_comm, &requests[req_count++]);
        }

        // по Z
        if (neighbors[4] != MPI_PROC_NULL) {
            MPI_Isend(&old_grid[0], 1, xy_plane, neighbors[4], 4, cart_comm, &requests[req_count++]);
            MPI_Irecv(&old_grid[0], 1, xy_plane, neighbors[4], 5, cart_comm, &requests[req_count++]);
        }
        if (neighbors[5] != MPI_PROC_NULL) {
            MPI_Isend(&old_grid[(Nz - 1)], 1, xy_plane, neighbors[5], 5, cart_comm, &requests[req_count++]);
            MPI_Irecv(&old_grid[(Nz - 1)], 1, xy_plane, neighbors[5], 4, cart_comm, &requests[req_count++]);
        }

        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // Якоби
        jacobi_iteration(old_grid, new_grid, Nx, Ny, Nz);

        old_grid.swap(new_grid); // Меняем массивы местами
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        cout << "Время выполнения: " << elapsed_time << " секунд." << endl;
    }

    MPI_Type_free(&yz_plane);
    MPI_Type_free(&xz_plane);
    MPI_Type_free(&xy_plane);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
