#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <cstdlib>

__host__ __device__ int flat(int i, int j, int Nx){
    return (j * Nx) + i;
}

static void skip_to_data(std::ifstream& in) {
    // Skip whitespace, blank lines, and full comment lines starting with '#'
    while (true) {
        in >> std::ws;
        int c = in.peek();
        if (c == '#') {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (c == '\n') {
            in.get();
            continue;
        }
        break;
    }
}

static void die_if_fail(const std::ifstream& in, const std::string& what, const std::string& filename) {
    if (!in) {
        std::cerr << "Parse failed while reading " << what << " from file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }
}

void read_fit_file(
    const std::string& filename,
    int m,              // material index: 0..nMat-1
    int K_m,
    int L_m,
    float* R_inf,
    float* Ak,
    float* zeta,
    float* Bl,
    float* Cl,
    float* alpha,
    float* beta,
    int nMat            // number of materials (boundary excluded)
) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    // R_inf
    skip_to_data(in);
    in >> R_inf[m];
    die_if_fail(in, "R_inf", filename);

    // A_k, zeta_k
    skip_to_data(in);
    for (int k = 0; k < K_m; ++k) {
        in >> Ak[ flat(k, m, K_m) ] >> zeta[ flat(k, m, K_m) ];
        die_if_fail(in, "A_k zeta_k row " + std::to_string(k), filename);
    }

    // B_l, C_l, alpha_l, beta_l
    skip_to_data(in);
    for (int l = 0; l < L_m; ++l) {
        in >> Bl[ flat(l, m, L_m) ]
           >> Cl[ flat(l, m, L_m) ]
           >> alpha[ flat(l, m, L_m) ]
           >> beta[ flat(l, m, L_m) ];
        die_if_fail(in, "B_l C_l alpha_l beta_l row " + std::to_string(l), filename);
    }
}

// build bmap and return nB
// re-implement on GPU if slow
int build_boundary_map(uint8_t *oid, int N, int *bmap){
    int nB = 0; // number of boundary cells
    // bmap has size N
    for (int c = 0; c < N; c++) {
        if (oid[c] != 0xFF || o_ID != 0){
            bmap[c] = nB++;
        } else {
            bmap[c] = -1;
        }
    }
    printf("\n[OK] Built boundary map\n");
    return nB;
}
