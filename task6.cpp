#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#define TRESHOLD 5000
using namespace std;
int compare_ints(const void* a, const void* b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;

    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;  // Размер левого подмассива
    int n2 = right - mid;     // Размер правого подмассива

    vector<int> leftArr(n1);
    vector<int> rightArr(n2);

    for (int i = 0; i < n1; ++i)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; ++j)
        rightArr[j] = arr[mid + 1 + j];

    int i = 0;    // Индекс для leftArr
    int j = 0;    // Индекс для rightArr
    int k = left; // Индекс для arr

    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            ++i;
        } else {
            arr[k] = rightArr[j];
            ++j;
        }
        ++k;
    }
    while (i < n1) {
        arr[k] = leftArr[i];
        ++i;
        ++k;
    }
    while (j < n2) {
        arr[k] = rightArr[j];
        ++j;
        ++k;
    }
}

void MergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

#pragma omp task shared(arr) if (right - left > TRESHOLD)
        {
            MergeSort(arr, left, mid);
        }

#pragma omp task shared(arr) if (right - left > TRESHOLD)
        {
            MergeSort(arr, mid + 1, right);
        }

#pragma omp taskwait
        merge(arr, left, mid, right);
    }
}

int main() {
    int N = 1000000;

for (int p = 1; p < 40; p += 3) {
    vector<int> A(N);

    srand(time(nullptr));

    for (int i = 0; i < N; ++i) {
        A[i] = rand() % 1000;
    }

    vector<int> B = A;

    omp_set_num_threads(p);

    double start_time = omp_get_wtime();

#pragma omp parallel
    {
#pragma omp single
        {
            MergeSort(A, 0, N - 1);
        }
    }

    double end_time = omp_get_wtime();
    double merge_sort_time = end_time - start_time;
    start_time = omp_get_wtime();

    qsort(B.data(), N, sizeof(int), compare_ints);

    end_time = omp_get_wtime();
    double std_sort_time = end_time - start_time;
    cout << "число нитей: " << p << endl;
    cout << "Время параллельной сортировки слиянием: " << merge_sort_time << " секунд" << endl;
    cout << "Время сортировки qsort: " << std_sort_time << " секунд" << endl;

    bool is_equal = true;
    for (int i = 0; i < N; ++i) {
        if (A[i] != B[i]) {
            is_equal = false;
            break;
        }
    }
    if (is_equal) {
        cout << "Массивы отсортированы одинаково." << endl;
    } else {
        cout << "Ошибка: массивы отсортированы по-разному!" << endl;
    }

}
    return 0;
}
