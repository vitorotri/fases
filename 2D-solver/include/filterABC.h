#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>   // std::malloc, std::free
#include <cstddef>   // std::size_t

struct FitSoA_C {
    float R1;
    int K; // real poles
    int L; // complex pairs

    float* A;    float* zeta;          // size K   (rename f -> zeta if you want)
    float* B;    float* C;             // size L
    float* alpha; float* beta;         // size L
};

// Call this when done (or on error after partial alloc)
static void free_fit(FitSoA_C& fit) {
    std::free(fit.A);     std::free(fit.zeta);
    std::free(fit.B);     std::free(fit.C);
    std::free(fit.alpha); std::free(fit.beta);
    fit = {}; // zero everything
}

static FitSoA_C alloc_fit(int K, int L) {
    FitSoA_C fit{};
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
FitSoA_C load_fit_c_like(const std::string& path) {
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

    FitSoA_C fit = alloc_fit(K, L);
    fit.R1 = R1;

    std::size_t i = 3;
    for (int k = 0; k < K; ++k) { fit.A[k] = x[i++]; fit.zeta[k] = x[i++]; }
    for (int l = 0; l < L; ++l) {
        fit.B[l] = x[i++]; fit.C[l] = x[i++]; fit.alpha[l] = x[i++]; fit.beta[l] = x[i++];
    }
    return fit;
}
