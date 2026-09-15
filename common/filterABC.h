#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <cstdlib>

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
    int m,
    int K_m,
    int L_m,
    float* R_inf,
    float* Ak,
    float* zeta,
    float* Bl,
    float* Cl,
    float* alpha,
    float* beta
) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    int K_local = 0, L_local = 0;

    // R_inf
    skip_to_data(in);
    in >> R_inf[m];
    die_if_fail(in, "R_inf", filename);

    // K L
    skip_to_data(in);
    in >> K_local >> L_local;
    die_if_fail(in, "K L", filename);

    if (K_local < 0 || K_local > K_m) {
        std::cerr << "Invalid K_local=" << K_local << " (K_m=" << K_m
                  << ") in file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }
    if (L_local < 0 || L_local > L_m) {
        std::cerr << "Invalid L_local=" << L_local << " (L_m=" << L_m
                  << ") in file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    // A_k zeta_k
    skip_to_data(in);
    for (int k = 0; k < K_local; ++k) {
        in >> Ak[ flat(k, m, K_m) ]       // stride = K_m
           >> zeta[ flat(k, m, K_m) ];
        die_if_fail(in, "A_k zeta_k row " + std::to_string(k), filename);
    }

    // B_l C_l alpha_l beta_l
    skip_to_data(in);
    for (int l = 0; l < L_local; ++l) {
        in >> Bl[ flat(l, m, L_m) ]       // stride = L_m
           >> Cl[ flat(l, m, L_m) ]
           >> alpha[ flat(l, m, L_m) ]
           >> beta[ flat(l, m, L_m) ];
        die_if_fail(in, "B_l C_l alpha_l beta_l row " + std::to_string(l), filename);
    }
}


// build bmap and return nB
// re-implement on GPU if too slow
int build_boundary_map(uint8_t *oid, int N, int *bmap){
    int nB = 0; // number of boundary cells
    // bmap has size N, as well as oid
    // loop around every cell
    for (int c = 0; c < N; c++) {
        if (oid[c] != 0xFF && oid[c] != 0){
            bmap[c] = nB++;
        } else {
            bmap[c] = -1;
        }
    }
    printf("\n[OK] Built boundary map\n");
    return nB;
}

// Compact index map for split-field PML cells: any cell inside an ENABLED absorbing band
// of uniform thickness n_pml (cells) on every axis. pmap[c] = 0..nP-1 for PML, -1 interior.
// Faces match sPML: L/R = x-lo/hi, B/F = y-lo/hi, D/U = z-lo/hi.
int build_pml_map(int Nx, int Ny, int Nz, int n_pml,
                  float U_PML, float D_PML, float R_PML, float L_PML, float F_PML, float B_PML,
                  int *pmap){
    int nP = 0;
    for (int k = 0; k < Nz; k++){
        for (int j = 0; j < Ny; j++){
            for (int i = 0; i < Nx; i++){
                size_t c = i + Nx*(j + k*Ny);
                int inx = (L_PML != 0.0f && i <  n_pml) || (R_PML != 0.0f && i >= Nx - n_pml);
                int iny = (B_PML != 0.0f && j <  n_pml) || (F_PML != 0.0f && j >= Ny - n_pml);
                int inz = (D_PML != 0.0f && k <  n_pml) || (U_PML != 0.0f && k >= Nz - n_pml);
                pmap[c] = (inx || iny || inz) ? nP++ : -1;
            }
        }
    }
    printf("\n[OK] Built PML map (%d cells)\n", nP);
    return nP;
}
