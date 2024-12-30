#include <iostream>
#include <vector>
#include <cmath>
#include <papi.h>

class GraphCSR {
    int vertexCount;
    unsigned int edgeCount;
    std::vector<unsigned int> rowOffsets;
    std::vector<int> columnIndices;
    std::vector<double> edgeWeights;

public:
    void loadGraph(const char* filename) {
        FILE *file = fopen(filename, "rb");
        fread(reinterpret_cast<char*>(&vertexCount), sizeof(int), 1, file);
        fread(reinterpret_cast<char*>(&edgeCount), sizeof(unsigned int), 1, file);

        std::cout << "Vertices: " << vertexCount << ", Edges: " << edgeCount << std::endl;

        rowOffsets.resize(vertexCount + 1);
        columnIndices.resize(edgeCount);
        edgeWeights.resize(edgeCount);

        fread(reinterpret_cast<char*>(rowOffsets.data()), sizeof(unsigned int), vertexCount + 1, file);
        fread(reinterpret_cast<char*>(columnIndices.data()), sizeof(int), edgeCount, file);
        fread(reinterpret_cast<char*>(edgeWeights.data()), sizeof(double), edgeCount, file);
        fclose(file);
    }

    void displayVertex(int index) const {
        for (int i = rowOffsets[index]; i < rowOffsets[index + 1]; i++) {
            std::cout << columnIndices[i] << " " << edgeWeights[i] << std::endl;
        }
    }

    int findVertexWithMaxWeight() const {
        int targetVertex = -1;
        float maxTotalWeight = -1;

        for (int i = 0; i < vertexCount; i++) {
            float currentWeight = 0;
            for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; j++) {
                if (columnIndices[j] % 2 == 0) {
                    currentWeight += edgeWeights[j];
                }
            }
            if (currentWeight > maxTotalWeight) {
                maxTotalWeight = currentWeight;
                targetVertex = i;
            }
        }
        return targetVertex;
    }

    int findVertexWithMaxRank() const {
        int vertexWithMaxRank = -1;
        float highestRank = -1;

        for (int i = 0; i < vertexCount; i++) {
            float rank = 0;
            for (int j = rowOffsets[i]; j < rowOffsets[i + 1]; j++) {
                float vertexWeight = 0;
                for (int k = rowOffsets[columnIndices[j]]; k < rowOffsets[columnIndices[j] + 1]; k++) {
                    vertexWeight += edgeWeights[k] * (rowOffsets[columnIndices[j] + 1] - rowOffsets[columnIndices[j]]);
                }
                rank += edgeWeights[j] * vertexWeight;
            }
            if (rank > highestRank) {
                highestRank = rank;
                vertexWithMaxRank = i;
            }
        }
        return vertexWithMaxRank;
    }
};

#define TEST_COUNT 5

int main() {
    const char* testFiles[TEST_COUNT] = {
            "synt",
            "road_graph",
            "stanford",
            "youtube",
            "syn_rmat"
    };

    int papiEventSet = PAPI_NULL, eventCode;
    long long counterValues[3];

    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&papiEventSet);
    PAPI_add_event(papiEventSet, PAPI_L1_TCM);
    PAPI_add_event(papiEventSet, PAPI_L2_TCM);

    char customEventName[] = "perf::PERF_COUNT_HW_CACHE_REFERENCES";
    PAPI_event_name_to_code(customEventName, &eventCode);
    PAPI_add_event(papiEventSet, eventCode);

    for (int test = 0; test < TEST_COUNT; test++) {
        GraphCSR graph;
        graph.loadGraph(testFiles[test]);

        PAPI_start(papiEventSet);
        std::cout << "Vertex with max weight (Algorithm 1): " << graph.findVertexWithMaxWeight() + 1 << std::endl;
        PAPI_stop(papiEventSet, counterValues);
        std::cout << "L1 Cache Misses: " << counterValues[0] << std::endl;
        std::cout << "L2 Cache Misses: " << counterValues[1] << std::endl;
        std::cout << "Cache References: " << counterValues[2] << std::endl;

        PAPI_reset(papiEventSet);
        PAPI_start(papiEventSet);
        std::cout << "Vertex with max rank (Algorithm 2): " << graph.findVertexWithMaxRank() + 1 << std::endl;
        PAPI_stop(papiEventSet, counterValues);
        std::cout << "L1 Cache Misses: " << counterValues[0] << std::endl;
        std::cout << "L2 Cache Misses: " << counterValues[1] << std::endl;
        std::cout << "Cache References: " << counterValues[2] << std::endl;
    }

    PAPI_destroy_eventset(&papiEventSet);
    PAPI_shutdown();
    return 0;
}
