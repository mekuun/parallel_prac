#include <mpi.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <utility>

using namespace std;

const int ROWS = 1024;
const int COLS = 1024;
const int MAX_ITERATIONS = 1000;
const double ALIVE_PROB = 0.2;

void initialize_grid(vector<vector<int> >& grid) {
    srand(static_cast<unsigned>(time(NULL)));
    for (size_t i = 0; i < grid.size(); ++i) {
        for (size_t j = 0; j < grid[i].size(); ++j) {
            grid[i][j] = (rand() < ALIVE_PROB * RAND_MAX) ? 1 : 0;
        }
    }
}

int count_neighbors(const vector<vector<int> >& grid, int x, int y) {
    static const pair<int, int> directions[] = {
            make_pair(-1, -1), make_pair(-1, 0), make_pair(-1, 1),
            make_pair(0, -1), make_pair(0, 1),
            make_pair(1, -1), make_pair(1, 0), make_pair(1, 1)
    };

    int count = 0;
    for (size_t i = 0; i < sizeof(directions) / sizeof(directions[0]); ++i) {
        int dx = directions[i].first;
        int dy = directions[i].second;
        int nx = (x + dx + grid.size()) % grid.size();
        int ny = (y + dy + grid[0].size()) % grid[0].size();
        count += grid[nx][ny];
    }
    return count;
}

int calculate_rows(int rank, int num_procs) {
    int base_rows = ROWS / num_procs;
    return base_rows + (rank < ROWS % num_procs ? 1 : 0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double start_time = MPI_Wtime();

    int local_rows = calculate_rows(rank, num_procs);
    int start_row = rank * (ROWS / num_procs) + (rank < ROWS % num_procs ? rank : ROWS % num_procs);

    vector<vector<int> > local_grid(local_rows + 2, vector<int>(COLS, 0));
    vector<vector<int> > next_grid(local_rows + 2, vector<int>(COLS, 0));

    vector<vector<int> > global_grid;
    if (rank == 0) {
        global_grid.resize(ROWS, vector<int>(COLS));
        initialize_grid(global_grid);
    }

    vector<int> send_counts(num_procs), displs(num_procs);
    for (int i = 0, offset = 0; i < num_procs; ++i) {
        send_counts[i] = calculate_rows(i, num_procs) * COLS;
        displs[i] = offset;
        offset += send_counts[i];
    }

    vector<int> local_data(local_rows * COLS);
    MPI_Scatterv(rank == 0 ? &global_grid[0][0] : NULL, &send_counts[0], &displs[0], MPI_INT,
                 &local_data[0], local_rows * COLS, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
        copy(local_data.begin() + i * COLS, local_data.begin() + (i + 1) * COLS, local_grid[i + 1].begin());
    }

    int iteration = 0, total_alive = 0;

    while (iteration < MAX_ITERATIONS) {
        int top = (rank == 0) ? num_procs - 1 : rank - 1;
        int bottom = (rank == num_procs - 1) ? 0 : rank + 1;

        MPI_Request requests[4];
        MPI_Isend(&local_grid[1][0], COLS, MPI_INT, top, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&local_grid[local_rows][0], COLS, MPI_INT, bottom, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Irecv(&local_grid[0][0], COLS, MPI_INT, top, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(&local_grid[local_rows + 1][0], COLS, MPI_INT, bottom, 0, MPI_COMM_WORLD, &requests[3]);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        int local_alive = 0;
        for (int i = 1; i <= local_rows; ++i) {
            for (int j = 0; j < COLS; ++j) {
                int neighbors = count_neighbors(local_grid, i, j);
                if (local_grid[i][j] == 1) {
                    next_grid[i][j] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
                } else {
                    next_grid[i][j] = (neighbors == 3) ? 1 : 0;
                }
                local_alive += next_grid[i][j];
            }
        }

        for (int i = 1; i <= local_rows; ++i) {
            for (int j = 0; j < COLS; ++j) {
                local_grid[i][j] = next_grid[i][j];
            }
        }

        MPI_Allreduce(&local_alive, &total_alive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (total_alive == 0) break;
        iteration++;
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        cout << "Игра завершилась на итерации номер " << iteration  << endl;
        cout << "Всего живых: " << total_alive
        cout << "Время выполнения: " << elapsed_time << " секунд." << endl;
    }

    MPI_Finalize();
    return 0;
}
