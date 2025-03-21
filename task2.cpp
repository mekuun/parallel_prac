#include <iostream>
#include <pthread.h>
#include <queue>
#include <unistd.h>

using std::cin, std::cout;

class NoMoreItemsException {};

template <typename T>
class MyConcurrentQueue {
public:
    MyConcurrentQueue(size_t maxSize) : maxSize(maxSize), done(false) {
        pthread_mutex_init(&mutex, nullptr);
        pthread_cond_init(&cond_full, nullptr);
        pthread_cond_init(&cond_empty, nullptr);
    }

    ~MyConcurrentQueue() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond_full);
        pthread_cond_destroy(&cond_empty);
    }

    void put(const T& item) {
        pthread_mutex_lock(&mutex);
        while (queue.size() >= maxSize) {
            pthread_cond_wait(&cond_full, &mutex);
        }
        queue.push(item);
        std::cout << "Written: " << item << std::endl;
        pthread_cond_signal(&cond_empty);
        pthread_mutex_unlock(&mutex);
    }

    T get() {
        pthread_mutex_lock(&mutex);
        while (queue.empty() && !done) {
            pthread_cond_wait(&cond_empty, &mutex);
        }
        if (queue.empty() && done) {
            pthread_mutex_unlock(&mutex);
            throw NoMoreItemsException();
        }
        T item = queue.front();
        queue.pop();
        std::cout << "Read: " << item << std::endl;
        pthread_cond_signal(&cond_full);
        pthread_mutex_unlock(&mutex);
        return item;
    }

    void setDone() {
        pthread_mutex_lock(&mutex);
        done = true;
        pthread_cond_broadcast(&cond_empty);
        pthread_mutex_unlock(&mutex);
    }

private:
    std::queue<T> queue;
    size_t maxSize;
    pthread_mutex_t mutex;
    pthread_cond_t cond_full;
    pthread_cond_t cond_empty;
    bool done;
};

void* writer(void* arg) {
    MyConcurrentQueue<int>* queue = static_cast<MyConcurrentQueue<int>*>(arg);
    for (int i = 0; i < 10; ++i) {
        queue->put(i);
        sleep(1);
    }
    return nullptr;
}

void* reader(void* arg) {
    MyConcurrentQueue<int>* queue = static_cast<MyConcurrentQueue<int>*>(arg);
    try {
        while (true) {
            int item = queue->get();
            sleep(1);
        }
    } catch (const NoMoreItemsException&) {
    }
    return nullptr;
}

int main() {
    MyConcurrentQueue<int> queue(5);

    int num_readers, num_writers;

    cin >> num_readers >> num_writers;

    pthread_t readers[num_readers];
    pthread_t writers[num_writers];

    for (int i = 0; i < num_writers; ++i) {
        pthread_create(&writers[i], nullptr, writer, &queue);
    }

    for (int i = 0; i < num_readers; ++i) {
        pthread_create(&readers[i], nullptr, reader, &queue);
    }

    for (int i = 0; i < num_writers; ++i) {
        pthread_join(writers[i], nullptr);
    }

    queue.setDone();

    for (int i = 0; i < num_readers; ++i) {
        pthread_join(readers[i], nullptr);
    }

    return 0;
}
