#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int num_threads;
int num_intervals;
double step;
double sum = 0;
pthread_mutex_t mutex;


double get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1000000.0;
}


void* calculate_pi(void* arg) {
    int thread_id = *(int*)arg;
    double local_sum = 0;
    int i;

    int start = (num_intervals / num_threads) * thread_id;
    int end = (thread_id == num_threads - 1) ? num_intervals : start + (num_intervals / num_threads);

    for (i = start; i < end; i++) {
        double x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    pthread_mutex_lock(&mutex);
    sum += local_sum;
    pthread_mutex_unlock(&mutex);

    pthread_exit(0);
}

int main(int argc, char** argv) {

    if (argc != 3) {
        return -1;
    }

    num_intervals = atoi(argv[1]);
    num_threads = atoi(argv[2]);

    step = 1.0 / num_intervals;

    pthread_mutex_init(&mutex, NULL);

    pthread_t threads[num_threads];
    int thread_ids[num_threads];

    double start = get_time();

    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, &calculate_pi, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    double end = get_time();
    double time_used = end - start;
    printf("Elapsed time: %f s\n", time_used);

    sum *= step;
    printf("%.15f\n", sum);

    return 0;
}
