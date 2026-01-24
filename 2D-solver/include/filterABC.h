#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <cstdio>
#include <cstdlib>   // std::malloc, std::free
#include <cstddef>   // std::size_t
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",         \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        std::exit(1);                                         \
    }                                                         \
} while(0)

struct matABC {
    float R1;
    int K; // real poles
    int L; // complex pairs

    float* A;    float* zeta;          // size K   (rename f -> zeta if you want)
    float* B;    float* C;             // size L
    float* alpha; float* beta;         // size L
};

// Call this when done (or on error after partial alloc)
static void free_fit(matABC& fit) {
    std::free(fit.A);     std::free(fit.zeta);
    std::free(fit.B);     std::free(fit.C);
    std::free(fit.alpha); std::free(fit.beta);
    fit = {}; // zero everything
}

static matABC alloc_fit(int K, int L) {
    matABC fit{};
    fit.K = K; fit.L = L;

    fit.A     = (float*)std::malloc((std::size_t)K * sizeof(float));
    fit.zeta  = (float*)std::malloc((std::size_t)K * sizeof(float));
    fit.B     = (float*)std::malloc((std::size_t)L * sizeof(float));
    fit.C     = (float*)std::malloc((std::size_t)L * sizeof(float));
    fit.alpha = (float*)std::malloc((std::size_t)L * sizeof(float));
    fit.beta  = (float*)std::malloc((std::size_t)L * sizeof(float));

    if ((K && (!fit.A || !fit.zeta)) || (L && (!fit.B || !fit.C || !fit.alpha || !fit.beta))) {
        free_fit(fit);
        throw std::runtime_error("malloc failed while allocating fit arrays");
    }
    return fit;
}

// Tiny reader: numbers only; whole-line comments start with '#'
matABC load_fit_c_like(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open " + path);

    std::vector<float> x;
    x.reserve(128);

    std::string tok;
    while (in >> tok) {
        if (!tok.empty() && tok[0] == '#') { std::getline(in, tok); continue; }
        x.push_back(std::stod(tok));
    }

    if (x.size() < 3) throw std::runtime_error("Need at least: R1 K L");

    const float R1 = x[0];
    const int K = (int)x[1];
    const int L = (int)x[2];

    if (K < 0 || L < 0) throw std::runtime_error("K/L must be non-negative");
    if (x[1] != (float)K || x[2] != (float)L) throw std::runtime_error("K and L must be integers");

    const std::size_t need = 3u + 2u*(std::size_t)K + 4u*(std::size_t)L;
    if (x.size() != need)
        throw std::runtime_error("Wrong number of values (expected " + std::to_string(need) +
                                 ", got " + std::to_string(x.size()) + ")");

    matABC fit = alloc_fit(K, L);
    fit.R1 = R1;

    std::size_t i = 3;
    for (int k = 0; k < K; ++k) { fit.A[k] = x[i++]; fit.zeta[k] = x[i++]; }
    for (int l = 0; l < L; ++l) {
        fit.B[l] = x[i++]; fit.C[l] = x[i++]; fit.alpha[l] = x[i++]; fit.beta[l] = x[i++];
    }
    return fit;
}

namespace fs = std::filesystem;

// You already have these:
matABC load_fit_c_like(const std::string& path);
void free_fit(matABC& fit);

// Returns number of loaded materials in outCount.
// Allocates:
//   outNames: array of std::string  (file names)
//   outFits : array of matABC     (loaded coeffs)
// Caller must delete[] outNames, delete[] outFits, and free_fit() each fit.
static void load_all_mats_from_dir(
    const std::string& dir,
    std::string*& outNames,
    matABC*& outFits,
    int& outCount
) {
    outNames = nullptr;
    outFits  = nullptr;
    outCount = 0;

    // Pass 1: count
    int count = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() == ".fit") ++count;
    }
    if (count == 0) return;

    // Allocate C-style arrays
    outNames = new std::string[count];     // ok
    outFits  = new matABC[count]();      // () value-inits -> pointers become null
    outCount = count;

    // Pass 2: load
    int i = 0;
    try {
        for (const auto& e : fs::directory_iterator(dir)) {
            if (!e.is_regular_file()) continue;
            if (e.path().extension() != ".fit") continue;

            const std::string path = e.path().string();
            outNames[i] = e.path().filename().string();   // or store full path if you prefer
            outFits[i]  = load_fit_c_like(path);          // copies pointers (fine)
            ++i;
        }
    } catch (...) {
        // Clean up partially loaded
        for (int j = 0; j < i; ++j) free_fit(outFits[j]);
        delete[] outFits;
        delete[] outNames;
        outFits = nullptr;
        outNames = nullptr;
        outCount = 0;
        throw;
    }

    // If directory changed between passes, i might be < count.
    // Keep it strict/simple: shrink logically.
    outCount = i;
}

static matABC* upload_fits_to_gpu(const matABC* fits, int n_mat) {
    // Host mirror that will contain DEVICE pointers
    matABC* h_dev = new matABC[n_mat](); // value-init -> pointers become nullptr

    for (int i = 0; i < n_mat; ++i) {
        h_dev[i].R1 = fits[i].R1;
        h_dev[i].K  = fits[i].K;
        h_dev[i].L  = fits[i].L;

        const int K = fits[i].K;
        const int L = fits[i].L;

        // K arrays: A, zeta
        if (K > 0) {
            CUDA_CHECK(cudaMalloc(&h_dev[i].A,    (size_t)K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&h_dev[i].zeta, (size_t)K * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(h_dev[i].A,    fits[i].A,    (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_dev[i].zeta, fits[i].zeta, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
        }

        // L arrays: B, C, alpha, beta
        if (L > 0) {
            CUDA_CHECK(cudaMalloc(&h_dev[i].B,     (size_t)L * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&h_dev[i].C,     (size_t)L * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&h_dev[i].alpha, (size_t)L * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&h_dev[i].beta,  (size_t)L * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(h_dev[i].B,     fits[i].B,     (size_t)L * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_dev[i].C,     fits[i].C,     (size_t)L * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_dev[i].alpha, fits[i].alpha, (size_t)L * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_dev[i].beta,  fits[i].beta,  (size_t)L * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // Now copy the struct array (with device pointers inside) to device
    matABC* d_fits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fits, (size_t)n_mat * sizeof(matABC)));
    CUDA_CHECK(cudaMemcpy(d_fits, h_dev, (size_t)n_mat * sizeof(matABC), cudaMemcpyHostToDevice));

    delete[] h_dev;
    return d_fits;
}

static void free_fits_on_gpu(matABC* d_fits, int n_mat) {
    // Copy back to recover the device pointers
    matABC* h_dev = new matABC[n_mat];
    CUDA_CHECK(cudaMemcpy(h_dev, d_fits, (size_t)n_mat * sizeof(matABC), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_mat; ++i) {
        CUDA_CHECK(cudaFree(h_dev[i].A));
        CUDA_CHECK(cudaFree(h_dev[i].zeta));
        CUDA_CHECK(cudaFree(h_dev[i].B));
        CUDA_CHECK(cudaFree(h_dev[i].C));
        CUDA_CHECK(cudaFree(h_dev[i].alpha));
        CUDA_CHECK(cudaFree(h_dev[i].beta));
    }

    delete[] h_dev;
    CUDA_CHECK(cudaFree(d_fits));
}
