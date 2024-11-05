#include <iostream>
#include <omp.h>
#include <random>
#include <ctime>
#include <cmath>

using namespace std;

int random_walk(int a, int b, int x, double p) {
    int steps = 0;
    thread_local std::mt19937 generator(std::random_device{}());
    uniform_real_distribution<double> distribution(0.0, 1.0);

    while (x > a && x < b) {
        mt19937 generator(seed);
        uniform_real_distribution<double> distribution(0.0, 1.0);
        if (rnd < p) x++;
        else x--;
        steps++;
    }
    if (x == b) return steps;
    return -1 * steps;
}

int main(int argc, char *argv[]) {
    double p = 0.5;
    int N, P, a = 1, b = 300, x = 150;
    cin >> N >> P;

    int count_b = 0;
    int total_steps = 0;

    omp_set_num_threads(P);

    double start_time = omp_get_wtime();

#pragma omp parallel for reduction(+:count_b, total_steps)
    for (int i = 0; i < N; ++i) {
        int steps = random_walk(a, b, x, p);
        if (steps > 0) count_b++;
        total_steps += abs(steps);
    }

    double end_time = omp_get_wtime();

    double probability_b = (double) count_b / N;
    double average_lifetime = (double) total_steps / N;
    double execution_time = end_time - start_time;

    cout << "Вероятность достижения b: " << probability_b << endl;
    cout << "Среднее время жизни одной частицы: " << average_lifetime << endl;
    cout << "Время работы основного цикла: " << execution_time << " секунд" << endl;

    return 0;
}
