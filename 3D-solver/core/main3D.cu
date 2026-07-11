#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <new>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <sstream>
#include <unistd.h>
#include <cuda_runtime.h>
#include "ism3D.h"
#include "raytrace3D.h"
#include "functions3D.cuh"

unsigned long long available_ram_bytes() {
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    return (unsigned long long)pages * (unsigned long long)page_size;
}

int main(int argc, char *argv[]){
	if (argc < 5) {
        printf("Usage: %s <scene> <config> <src> <rec> [poles] [--estimate]\n", argv[0]);
        return 1;
    }

    // --estimate: parse the scene, compute V / T60 / f_Schroeder, write the two
    // single-value binaries for the GUI Setup tab, and exit BEFORE any field
    // allocation / FDTD / ISM. Fast, and needs no poles.txt (T60 uses only .mat alpha).
    bool estimate_only = false;
    bool dump_rays = false;   // --dump-rays [N] [M]: write ray_paths.txt for the GUI overlay and exit
    int  dump_nrays = 0;      // optional ray count after --dump-rays (0 => fall back)
    int  dump_order = 0;      // optional max reflection order after the count (0 => default 5)
    for (int i = 1; i < argc; ++i){
        if (std::strcmp(argv[i], "--estimate")  == 0) estimate_only = true;
        if (std::strcmp(argv[i], "--dump-rays") == 0){
            dump_rays = true;
            if (i+1 < argc) dump_nrays = std::atoi(argv[i+1]);   // 0 if the next token isn't a number
            if (i+2 < argc) dump_order = std::atoi(argv[i+2]);
        }
    }

    // -------------------------------------------------------------------------
    // Project-aware paths. The executable may live in build/, but the CASE
    // files live under the project root. Derive the project root from argv[2]
    // when argv[2] is .../config/config.txt. This keeps all physics/math
    // untouched and fixes standalone execution from any working directory.
    // -------------------------------------------------------------------------

    auto canonical_input = [](const char* raw) -> std::filesystem::path {
        std::filesystem::path p(raw);
        if (p.is_relative()) p = std::filesystem::current_path() / p;
        return p.lexically_normal();
    };

    std::filesystem::path scene_path  = canonical_input(argv[1]);
    std::filesystem::path config_path = canonical_input(argv[2]);
    std::filesystem::path src_path    = canonical_input(argv[3]);
    std::filesystem::path rec_path    = canonical_input(argv[4]);

    std::filesystem::path project_root = std::filesystem::current_path();
    if (config_path.filename() == "config.txt" &&
        config_path.parent_path().filename() == "config") {
        project_root = config_path.parent_path().parent_path();
    }

    std::filesystem::path poles_path = config_path.parent_path() / "poles.txt";
    std::filesystem::path fit_dir    = project_root / "input" / "fit";
    std::filesystem::path output_dir = project_root / "output";
    std::filesystem::path imp_dir = project_root / "input" / "imp";
    std::filesystem::path mat_dir = project_root / "input" / "mat";

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        fprintf(stderr, "[!] Cannot create output directory: %s (%s)\n",
                output_dir.string().c_str(), ec.message().c_str());
        return EXIT_FAILURE;
    }

    auto out_file = [&](const std::string& name) -> std::string {
        return (output_dir / name).string();
    };

	float t_max, dt, h, f_cutoff; // max time, time-step, grid spacing, max frequency
	float *p0, *u0, *v0, *w0, *p1, *u1, *v1, *w1;// *p2, *u2, *v2, *w2; // host field arrays
	float *d_p0, *d_u0, *d_v0, *d_w0, *d_p1, *d_u1, *d_v1, *d_w1, *d_p2, *d_u2, *d_v2, *d_w2; // device arrays
	float *c_rl, *dd, *beta_rl; // WENO5 arrays
	float rho, c, Temp; // density, speed of sound, temperature
	float Lx, Ly, Lz; // domain dimensions
	float U_PML, D_PML, R_PML, L_PML, F_PML, B_PML; // Perfectly Matched Layer activation keys
    int Nx, Ny, Nz, n; // number of points in x, y, time step
    
    // timing variables
    clock_t tic, toc;
    double elapsed_time;
    
    // read argv[2] file
    std::vector<float> conf_arr;
    std::ifstream file(config_path);
    std::string line;
    while (std::getline(file, line)){
    	// Truncate at # (remove comments after float)
    	size_t hash_pos = line.find('#');
    	if (hash_pos != std::string::npos) {
        	line.erase(hash_pos);  // Everything after # is gone
    	}
    	std::istringstream iss(line);
		float f;
		if (iss >> f) conf_arr.push_back(f);
	}
	file.close();
	file.clear();
	
    // domain variables
    h = conf_arr[0]; // uniform grid spacing [m]
    t_max = conf_arr[1]; // maximum simulation time [s]
    f_cutoff = conf_arr[2]; // [Hz] (cutoff frequency)
    
    // PML keys
    U_PML = (float)conf_arr[6];
    D_PML = (float)conf_arr[7];
    R_PML = (float)conf_arr[8];
    L_PML = (float)conf_arr[9];
    F_PML = (float)conf_arr[10];
    B_PML = (float)conf_arr[11];
    
    printf("\nU_PML = %d	D_PML = %d\n", (int)U_PML, (int)D_PML);
    printf("\nL_PML = %d	R_PML = %d\n", (int)L_PML, (int)R_PML);
    printf("\nF_PML = %d	B_PML = %d\n", (int)F_PML, (int)B_PML);
    
    n = 0; // initial time-step
    
    // ------------------------ load OBJ/PLY/XYZ/etc file and create scene variables ------------------------
    
    // Remove stale outputs from the project output directory, not from cwd.
    // S.bin / oid.bin are kept: they double as a cache (see scene_cache.txt below)
    for (const char* fn : {"pressureX.bin", "pressureY.bin", "pressureZ.bin",
                           "header.txt", "pre_header.txt"}) {
        std::remove(out_file(fn).c_str());
    }
        
    std::string scene_str = scene_path.string();
    const char *path = scene_str.c_str();
    int nObj = 0; // number of objects in the scene (initialization)
    float *S = NULL; // signed distance function array or voxels, size Nx*Ny, flattened
    float *d_S = NULL;
    float tol = 3.0f*h; // tolerance
    uint8_t *oid = NULL; // for surface attribution
    uint8_t* d_oid = NULL;
    
    // ---------------------------------- get Device(s) information ----------------------------------------
    //
    // Important: always check the return value from cudaGetDeviceCount().
    // If the CUDA runtime cannot talk to the installed driver, nDevices is not
    // a trustworthy value. The old code ignored the error and could report
    // "no CUDA devices found" even when nvidia-smi could see the GPU.

    const char *visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
    printf("\n CUDA_VISIBLE_DEVICES=%s\n", visible_devices ? visible_devices : "<unset>");

    int nDevices = 0;
    cudaError_t devErr = cudaGetDeviceCount(&nDevices);
    if (devErr != cudaSuccess) {
        fprintf(stderr, "\n[!] FAILED: cudaGetDeviceCount() returned %d (%s): %s\n",
                (int)devErr, cudaGetErrorName(devErr), cudaGetErrorString(devErr));

        if (devErr == cudaErrorInsufficientDriver) {
            fprintf(stderr, "[!] The executable was probably built with a CUDA runtime newer "
                            "than the installed NVIDIA driver supports. Rebuild with CUDA 11.x "
                            "for driver 470.x, or update the NVIDIA driver.\n");
        } else if (devErr == cudaErrorNoDevice) {
            fprintf(stderr, "[!] The CUDA runtime did not expose any CUDA-capable GPU to this process.\n");
        } else if (devErr == cudaErrorInitializationError) {
            fprintf(stderr, "[!] CUDA runtime initialization failed. Check driver state, permissions, "
                            "and whether another environment/library path is interfering.\n");
        }

        fprintf(stderr, "\n");
        return -1;
    }

    if (nDevices <= 0){
        fprintf(stderr, "\n[!] FAILED: no CUDA devices found by the CUDA runtime\n\n");
        return -1;
    }

    int selectedDevice = 0;
    cudaError_t setErr = cudaSetDevice(selectedDevice);
    if (setErr != cudaSuccess) {
        fprintf(stderr, "\n[!] FAILED: cudaSetDevice(%d) returned %d (%s): %s\n\n",
                selectedDevice, (int)setErr, cudaGetErrorName(setErr), cudaGetErrorString(setErr));
        return -1;
    }

    // Force context creation now, so device/runtime problems are reported here
    // rather than much later as a misleading allocation/kernel failure.
    cudaError_t initErr = cudaFree(0);
    if (initErr != cudaSuccess) {
        fprintf(stderr, "\n[!] FAILED: CUDA context initialization returned %d (%s): %s\n\n",
                (int)initErr, cudaGetErrorName(initErr), cudaGetErrorString(initErr));
        return -1;
    }

    for (int i = 0; i < nDevices; ++i) {
        cudaDeviceProp prop{};
        cudaError_t propErr = cudaGetDeviceProperties(&prop, i);
        if (propErr != cudaSuccess) {
            fprintf(stderr, "\n[!] WARNING: cudaGetDeviceProperties(%d) returned %d (%s): %s\n",
                    i, (int)propErr, cudaGetErrorName(propErr), cudaGetErrorString(propErr));
            continue;
        }

        printf("\n Device %d%s: %s (CC %d.%d, %zu bytes global)\n",
               i, (i == selectedDevice) ? " [selected]" : "", prop.name,
               prop.major, prop.minor, prop.totalGlobalMem);
    }
    
    // -------------------------------------- process OBJ scene ------------------------------------
    
    float A_tot; // total surface area of domain
    std::vector<float> A; // vector of surface area per material
    std::vector<std::string> names; // global naming
    Scene3D sc;
   	if (ends_with_dotobj(path)){
   	
   		printf("\n[OK] processing scene...\n");

		if(!parse_obj3d(path, &sc, h)) return -1;

		// --dump-rays: pure-geometry trace. Build the BVH from the just-parsed mesh,
		// shoot the debug rays, write ray_paths.txt for the GUI overlay, and exit --
		// BEFORE the SDF/OID build, since the ray tracer never uses them.
		if (dump_rays){
			RTScene rtS = rt_build(sc);
			std::vector<vec3> srcs;   // all source positions from src.txt (world coords)
			{
				std::ifstream sf(src_path); std::string ln;
				while (std::getline(sf, ln)){
					if (ln.empty() || ln[0] == '#') continue;
					std::istringstream is(ln); float x,y,z;
					if (is >> x >> y >> z) srcs.push_back(v3(x,y,z));
				}
			}
			if (srcs.empty()) srcs.push_back(v3(0,0,0));
			int Ndbg = (dump_nrays > 0) ? dump_nrays
			         : ((conf_arr.size() > 14) ? std::max(1, (int)conf_arr[14]) : 200);
			int Mdbg = (dump_order > 0) ? dump_order : 5;
			rt_dump_paths(rtS, srcs, Ndbg, Mdbg, out_file("ray_paths.txt").c_str());
			printf("\n[OK] dumped %d ray paths (max order %d) -> output/ray_paths.txt\n", Ndbg, Mdbg);
			scene_free(&sc);
			return 0;
		}

		A = mesh_group_areas(sc, &A_tot); // A[i] is aligned with Z[i].alpha[b], for band b (layered Z, no BoundingBox)
		// created in the ISM section

		// S.bin/oid.bin double as a cache, keyed by scene mtime + h + tol
		std::error_code fec;
		std::ostringstream key_s;
		key_s << scene_str << " "
		      << std::filesystem::last_write_time(scene_path, fec).time_since_epoch().count()
		      << " " << h << " " << tol;
		std::string cache_key = key_s.str(), old_key;

		std::ifstream ck(out_file("scene_cache.txt"));
		bool cached = std::getline(ck, old_key) && old_key == cache_key
		           && (ck >> Nx >> Ny >> Nz >> Lx >> Ly >> Lz);
		ck.close();
		if (cached){
			size_t Ncells = (size_t)Nx*Ny*Nz;
			S   = (float*)malloc(Ncells*sizeof(float));
			oid = (uint8_t*)malloc(Ncells);
			FILE *fS = fopen(out_file("S.bin").c_str(), "rb");
			FILE *fO = fopen(out_file("oid.bin").c_str(), "rb");
			cached = S && oid && fS && fO
			      && fread(S, sizeof(float), Ncells, fS) == Ncells
			      && fread(oid, 1, Ncells, fO) == Ncells;
			if (fS) fclose(fS);
			if (fO) fclose(fO);
			if (!cached){ free(S); S = NULL; free(oid); oid = NULL; }
		}

		if (cached){
			printf("\n[OK] scene unchanged, reusing cached S.bin / oid.bin\n");
		}
		else {
			if(!build_sdf(&sc, h, &Nx, &Ny, &Nz, &Lx, &Ly, &Lz, &S)){
			// grid_dims_checked already printed the reason + suggested h
			scene_free(&sc);
			return -1;
			}

			if(!build_oid_dense(sc, S, Nx, Ny, Nz, h, tol, &oid)){   // GPU, CPU fallback
				fprintf(stderr,"Failed to build OID map\n");
				free(S); scene_free(&sc);
				return -1;
			}

			FILE *file_S = fopen(out_file("S.bin").c_str(), "wb");
			if (file_S) {
				fwrite(S, sizeof(float), (size_t)Nx*Ny*Nz, file_S);
				fclose(file_S);
			}

			FILE *file_oid = fopen(out_file("oid.bin").c_str(), "wb");
			if (file_oid){
				fwrite(oid, sizeof(uint8_t), (size_t)Nx*Ny*Nz, file_oid);
				fclose(file_oid);
			}

			// stamp the cache only after both bins are on disk
			std::ofstream(out_file("scene_cache.txt"), std::ios::trunc)
			    << cache_key << "\n" << Nx << " " << Ny << " " << Nz << " "
			    << Lx << " " << Ly << " " << Lz << "\n";
		}

		cudaMalloc(&d_oid, (size_t)Nx*Ny*Nz*sizeof(uint8_t));
		cudaMemcpy(d_oid, oid, (size_t)Nx*Ny*Nz*sizeof(uint8_t), cudaMemcpyHostToDevice);
		
		nObj = scene_layer_count_3d(sc);
		scene_layer_names_3d(sc, names);
		
		cudaMalloc(&d_S, (size_t)Nx*Ny*Nz*sizeof(float));
		cudaMemcpy(d_S, S, (size_t)Nx*Ny*Nz*sizeof(float), cudaMemcpyHostToDevice);
		
    }
    // process DXF test (not implemented yet)
    else {
    	return -1;
    }
    
    size_t n_air = count_air_voxels(S, (size_t)Nx*Ny*Nz);
	float V = n_air * h*h*h; // effective air volume of domain
	
	// ------------------------------------- atmospheric values ------------------------------------------------
	
	// should to to config later
	
	float R_spec = 287.05; // specific gas constant for air [J] [kg]^-1 [K]^-1
	float gamma = 1.4; // specific heat ratio for air
	float alti = (conf_arr.size() > 13) ? conf_arr[13] : 0.0f; // altitude [m]; 0 if absent (older configs)
	
	rho = 1.225; // average sea level density for air [kg/m3]. For dry air, proportional to p0/R_spec*T
	Temp = conf_arr[3] + 273.15f; // temperature [K] = Celsius_temp + 273.15
	c = sqrt(gamma*R_spec*Temp); // speed of sound [m/s]
	// c = sqrt(K/rho) in case of liquids
    
     printf("\n[OK] Speed of sound c = %.2f m/s\n",c);
    
     // attenuation coefficients for sound pressure according to ISO 9613-1
	std::vector<float> m_air(8);
	float p_a = p_atm(alti,Temp);
	float octave_bands[8] = {63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0};
	float hh = 1.1; // average concentration of water vapour
	for (int i = 0; i < 8; i++){
		m_air[i] = alpha_air(p_a, hh, Temp, octave_bands[i]);
		printf("[OK] m_air_%d = %f \n", i, m_air[i]);
	}
    
    // ---------------------------------- Estimated T60s and f_Schroeder ---------------------------------------
    
    // later for ISM parsing, must call here to estimate T60s and give user f_Schroeder, since it depends on c
    // as well
    auto Z = ism_load_impedances_layered(imp_dir, names, mat_dir); // names[0]="BoundingBox"; handles the +1
    // compute estimated T60's
	std::vector<float> T60(8, 0.5f);
	estimate_T60s(T60, V, c, A_tot, A, Z, m_air);
	// T60 at 500 Hz. Guard the rigid-room sentinel (estimate_T60s writes 1e9 for a
	// near-zero-absorption band): a hard-walled 500 Hz band has no diffuse regime, so
	// f_Schroeder runs to infinity -> emit 0.0f and let the GUI skip it (fS > 0 check).
	float f_Schroeder = (T60[3] < 1e6f) ? 2000.0f * std::sqrt(T60[3] / V) : 0.0f;

	// Single-value binaries in the config dir for the GUI Setup tab: Vol.bin drives
	// the recommended ray count, f_Schroeder.bin the recommended cutoff frequency.
	{
		std::filesystem::path cdir = config_path.parent_path();
		FILE* fVol = fopen((cdir / "Vol.bin").string().c_str(), "wb");
		if (fVol){ fwrite(&V, sizeof(float), 1, fVol); fclose(fVol); }
		FILE* fSch = fopen((cdir / "f_Schroeder.bin").string().c_str(), "wb");
		if (fSch){ fwrite(&f_Schroeder, sizeof(float), 1, fSch); fclose(fSch); }
		FILE* fT60 = fopen((cdir / "T60.bin").string().c_str(), "wb"); // 8 octave bands
		if (fT60){ fwrite(T60.data(), sizeof(float), 8, fT60); fclose(fT60); }
		FILE* fA = fopen((cdir / "A_tot.bin").string().c_str(), "wb"); // total surface area [m^2]
		if (fA){ fwrite(&A_tot, sizeof(float), 1, fA); fclose(fA); }

	}
	printf("\n[OK] estimated T60s\n\n");
	for (int b = 0; b < 8; b++) printf("%.3f", T60[b]);

	// --estimate fast path: stop before the heavy solver stages.
	
	if (estimate_only){
		printf("\n[OK] estimate: V = %.3f m^3, f_Schroeder = %.2f Hz\n", V, f_Schroeder);
		free(S); free(oid); scene_free(&sc);
		return 0;
	}

    FILE *pre_header; // file for data output (sdf_plot.py TST)
	pre_header = fopen(out_file("pre_header.txt").c_str(), "w");
	if (pre_header != NULL){
    	fprintf(pre_header, "%d %d %d %f %d\n", Nx, Ny, Nz, h, (int)(tol/h));
    }
    fclose(pre_header);
    
    
    // ----------------------------- Print scene dimensions ---------------------------------------
    
    size_t N = Nx * Ny * Nz; // number of grid cells

    // Fail fast before the big allocations: the solver still needs ~36 B/voxel host
    // (bmap + 8 field arrays; S/oid already resident) and ~52 B/voxel device
    // (bmap + 12 field arrays). Swapping past this point is what freezes the machine.
    {
        unsigned long long avail = available_ram_bytes();
        size_t gpu_free = 0, gpu_total = 0;
        cudaMemGetInfo(&gpu_free, &gpu_total);
        double need_host = 36.0 * (double)N, need_dev = 52.0 * (double)N;
        double over = 0.0;
        if (avail > 0     && need_host > 0.7 * (double)avail)    over = std::max(over, need_host / (0.7 * (double)avail));
        if (gpu_total > 0 && need_dev  > 0.9 * (double)gpu_free) over = std::max(over, need_dev  / (0.9 * (double)gpu_free));
        if (over > 0.0) {
            fprintf(stderr, "\n[!] h = %g gives %zu voxels: ~%.2f GB host / ~%.2f GB device needed, "
                            "~%.2f GB host / ~%.2f GB device free. Increase h to >= %.4g.\n",
                    h, N, need_host / 1e9, need_dev / 1e9,
                    (double)avail / 1e9, (double)gpu_free / 1e9, h * cbrt(over));
            return -1;
        }
    }

    printf("\n\n[OK] Nx = %d  Ny = %d  Nz = %d\n", Nx, Ny, Nz);
    printf("\n[Ok] Lx = %f  Ly = %f  Lz = %f\n", Lx, Ly, Lz);
    
    // ----------------------- LOAD filter coefficiets for materials ------------------------------------
    
    // ----- host arrays for materials -----
    
    float *Rinf;
    float *Ak, *zeta;
    float *Bl, *Cl, *alpha, *beta;
    
    int nMat = nObj - 1; // exclude bounding box
    try {
    	Rinf = new float[nMat]();
    } catch (const std::bad_alloc&){
    	fprintf(stderr, "\n[!] Allocation for R_inf[] failed, not enough memory\n");
    	return -1;
    }
    
    // read K_m and L_m from file
    
    int K_m, L_m; // k_max and L_max from all the fits
    
    // read poles.txt file
    file.open(poles_path);
    if (!file.is_open()) {
        fprintf(stderr, "[!] Cannot open poles file: %s\n", poles_path.string().c_str());
        return EXIT_FAILURE;
    }
    file >> K_m >> L_m;
	file.close();
	file.clear();
	
	// minimum reasonable size safety
	if (K_m < 1) K_m = 1;
	if (L_m < 1) L_m = 1;
    
    try {
		Ak 		= new float[K_m * nMat]();
		zeta 	= new float[K_m * nMat]();
		
		Bl 		= new float[L_m * nMat]();
    	Cl 		= new float[L_m * nMat]();
    	alpha 	= new float[L_m * nMat]();
    	beta 	= new float[L_m * nMat]();
    } catch (const std::bad_alloc&){
    	fprintf(stderr, "\n[!] Allocation for remaining materials' coefficients, not enough memory\n");
    	return -1;
    }
    
    printf("\nChecking materials\n\n");
    for (size_t i = 0; i < names.size(); ++i){
  		printf("%zu -> %s\n", i, names[i].c_str());
  	}

    
    for (int m = 0; m < nMat; ++m) {
    	std::string filename = (fit_dir / (names[m+1] + ".fit")).string(); // names[0] = BoundingBox
    	read_fit_file(filename, m, K_m, L_m, Rinf, Ak, zeta, Bl, Cl, alpha, beta);
	}
	

	// ----- device arrays for materials -----

	float *d_Rinf;
    float *d_Ak, *d_zeta;
    float *d_Bl, *d_Cl, *d_alpha, *d_beta;
    
    cudaMalloc(&d_Rinf, nMat * sizeof(float));
	cudaMemcpy(d_Rinf, Rinf, nMat * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc(&d_Ak, K_m * nMat * sizeof(float));
	cudaMemcpy(d_Ak, Ak, K_m * nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_zeta, K_m * nMat * sizeof(float));
	cudaMemcpy(d_zeta, zeta, K_m * nMat * sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc(&d_Bl, L_m * nMat * sizeof(float));
	cudaMemcpy(d_Bl, Bl, L_m * nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_Cl, L_m * nMat * sizeof(float));
	cudaMemcpy(d_Cl, Cl, L_m * nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_alpha, L_m * nMat * sizeof(float));
	cudaMemcpy(d_alpha, alpha, L_m * nMat * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc(&d_beta, L_m * nMat * sizeof(float));
	cudaMemcpy(d_beta, beta, L_m * nMat * sizeof(float), cudaMemcpyHostToDevice);
	
	// ----- free host arrays -----
	
	delete[] Rinf;
	delete[] Ak;
	delete[] zeta;
	delete[] Bl;
	delete[] Cl;
	delete[] alpha;
	delete[] beta;
	
	// ----- boundary map for phi and psi (ADE) -----
	
	// host
	int *h_bmap = (int*)malloc(N * sizeof(int)); // similar to oid, but with many int's
	int nB = build_boundary_map(oid, N, h_bmap);
	
	// device
	int *d_bmap;
	cudaMalloc(&d_bmap, N * sizeof(int));
	cudaMemcpy(d_bmap, h_bmap, N * sizeof(int), cudaMemcpyHostToDevice);
	
	// ----- allocate and initialize phi and psi (device side only) -----
	
	float *phi, *psi_1, *psi_2;
	float *phi_old, *psi_1_old, *psi_2_old;
	float *phi_nu, *psi_1_nu, *psi_2_nu;
	
	cudaMalloc(&phi,   nB * K_m * sizeof(float));
	cudaMalloc(&psi_1, nB * L_m * sizeof(float));
	cudaMalloc(&psi_2, nB * L_m * sizeof(float));
	
	cudaMemset(phi,   0, nB * K_m * sizeof(float));
	cudaMemset(psi_1, 0, nB * L_m * sizeof(float));
	cudaMemset(psi_2, 0, nB * L_m * sizeof(float));
	
	cudaMalloc(&phi_old,   nB * K_m * sizeof(float));
	cudaMalloc(&psi_1_old, nB * L_m * sizeof(float));
	cudaMalloc(&psi_2_old, nB * L_m * sizeof(float));
	
	cudaMemset(phi_old,   0, nB * K_m * sizeof(float));
	cudaMemset(psi_1_old, 0, nB * L_m * sizeof(float));
	cudaMemset(psi_2_old, 0, nB * L_m * sizeof(float));
	
	cudaMalloc(&phi_nu,   nB * K_m * sizeof(float));
	cudaMalloc(&psi_1_nu, nB * L_m * sizeof(float));
	cudaMalloc(&psi_2_nu, nB * L_m * sizeof(float));
	
	cudaMemset(phi_nu,   0, nB * K_m * sizeof(float));
	cudaMemset(psi_1_nu, 0, nB * L_m * sizeof(float));
	cudaMemset(psi_2_nu, 0, nB * L_m * sizeof(float));
    
    // ---------------------------------------------------------------------------------------------------------

    // Allocate main host memory arrays and initilize to 0.0f
    // (budget already verified by the upfront check after N was computed)
    try {
    	p0 = new float[N](); // pressure deviations from p0
    	u0 = new float[N](); // x component of particle velocity 
    	v0 = new float[N](); // y component of particle velocity 
    	w0 = new float[N](); // z component of particle velocity
    } catch (const std::bad_alloc&){
    	fprintf(stderr, "\n[!] Allocation for initial pressures and velocities failed, not enough memory\nTry a larger grid spacing or smaller cutoff frequency\n");
    	return -1;
    }

	float C_CFL = conf_arr[4]; // Courant number: pick values < 0.5 (not equal to)
	dt = C_CFL*h/c; // Courant-Friedrichs-Lewy condition
	printf("\n	dt = %.9f\n", dt);
	
	// Check if Nyquist-Shannon Theorem is satisfied
	float fs = 1/dt; // sampling frequency
	float f_time = 0.5*fs; // = 0.5/dt maximum representable temporal frequency
	float f_space = 0.5*c/h; // maximum representable spatial frequency (from Nyquist, lambda_min = 2*dx)
	float f_max = std::min(f_time, f_space); // de facto maximum frequency being simulated in the mesh
	printf("\n[OK] f_max = %f Hz (compare with desired cutoff limit)\n\n", f_max);
	if (!(f_max >= f_cutoff)){
		printf("[!] Sampling frequency = %f, too low, aborting...\n\n",fs);
		return -1;
	}
	
	// rule of thumb for good boundary modelling etc: 10 ~ 20 cells per minimum wavelength (from f_cutoff)
	float lambda = c/f_cutoff;
	if (lambda / h < 10.0f){
		printf("[!] Resolution too low, aborting...\n\n");
		return -1;
	}
	
	size_t Nt = (size_t) (t_max/dt) + 1; // number of iterations

    // Allocate device memory
    if (cudaMalloc(&d_p0, N*sizeof(float)) != cudaSuccess
	 || cudaMalloc(&d_u0, N*sizeof(float)) != cudaSuccess
	 || cudaMalloc(&d_v0, N*sizeof(float)) != cudaSuccess
	 || cudaMalloc(&d_w0, N*sizeof(float)) != cudaSuccess){
    		printf("[!] CUDA error allocating device memory\n");
    	return -1;
	}
	cudaMemset(d_p0, 0.0, N * sizeof(float));
	cudaMemset(d_u0, 0.0, N * sizeof(float));
	cudaMemset(d_v0, 0.0, N * sizeof(float));
	cudaMemset(d_w0, 0.0, N * sizeof(float));
	
	// ---------------------------------- kernel options and extra arrays --------------------------------
	
	
	float eps = 1e-7; // the smaller the eps, the more precise the BCs work, but cannot be the smallest otherwise have inf. In C, floats can go up to 1e-38. With 1e-5 workd really bad, with 1e-10 works well, depite below machine precision for float, so results may be wrong even with convergence!!!
	
	try {
		p1 = new float[N](); // sound pressure
    	u1 = new float[N](); // x component of particle velocity 
    	v1 = new float[N](); // y component of particle velocity
    	w1 = new float[N](); // z component of particle velocity
    } catch (const std::bad_alloc&){
    	fprintf(stderr, "\n[!] Allocation for remaining host arrays failed, not enough memory\nTry a larger grid spacing or smaller cutoff frequency\n");
    	return -1;
    }
	
	cudaMalloc(&d_p1, N*sizeof(float));
	cudaMalloc(&d_u1, N*sizeof(float));
	cudaMalloc(&d_v1, N*sizeof(float));
	cudaMalloc(&d_w1, N*sizeof(float));
		
	cudaMemset(d_p1, 0.0, N * sizeof(float));
	cudaMemset(d_u1, 0.0, N * sizeof(float));
	cudaMemset(d_v1, 0.0, N * sizeof(float));
	cudaMemset(d_w1, 0.0, N * sizeof(float));
	
	// WENO5 extra arrays
	
    cudaMalloc(&d_p2, N*sizeof(float));
	cudaMalloc(&d_u2, N*sizeof(float));
	cudaMalloc(&d_v2, N*sizeof(float));
	cudaMalloc(&d_w2, N*sizeof(float));
		
	cudaMemset(d_p2, 0.0, N * sizeof(float));
	cudaMemset(d_u2, 0.0, N * sizeof(float));
	cudaMemset(d_v2, 0.0, N * sizeof(float));
	cudaMemset(d_w2, 0.0, N * sizeof(float));
			
	cudaMallocManaged(&c_rl, 12*sizeof(float)); // coefficients
	cudaMallocManaged(&beta_rl, 9*sizeof(float)); // coefficients for second term of beta_r
	cudaMallocManaged(&dd, 3*sizeof(float)); // array
			
	c_rl[flat(0,0,12)] = 11.0/6.0; c_rl[flat(0,1,12)] = -7.0/6.0; c_rl[flat(0,2,12)] = 1.0/3.0;
	c_rl[flat(1,0,12)] = 1.0/3.0; c_rl[flat(1,1,12)] = 5.0/6.0; c_rl[flat(1,2,12)] = -1.0/6.0;
	c_rl[flat(2,0,12)] = -1.0/6.0; c_rl[flat(2,1,12)] = 5.0/6.0; c_rl[flat(2,2,12)] = 1.0/3.0;
	c_rl[flat(3,0,12)] = 1.0/3.0; c_rl[flat(3,1,12)] = -7.0/6.0; c_rl[flat(3,2,12)] = 11.0/6.0;
			
	beta_rl[flat(0,0,9)] = 3.0; beta_rl[flat(0,1,9)] = -4.0; beta_rl[flat(0,2,9)] = 1.0;
	beta_rl[flat(1,0,9)] = 1.0; beta_rl[flat(1,1,9)] = 0.0; beta_rl[flat(1,2,9)] = -1.0;
	beta_rl[flat(2,0,9)] = 1.0; beta_rl[flat(2,1,9)] = -4.0; beta_rl[flat(2,2,9)] = 3.0;
			
	dd[0] = 3.0/10.0; dd[1] = 3.0/5.0; dd[2] = 1.0/10.0;
	
		
	cudaDeviceSynchronize();

	
	// --------------------- Parse sources and receiver positions from argv[3] and argv[4] ---------------------
    
    // parse source(s)
    
    int i_src, j_src, k_src; // discretized position of source
    
    std::vector<float> src_arr; // source array on host
    std::vector<float> src_pressure; // sound pressure values for sources
    
    
    file.open(src_path);
	if (!file.is_open()){
    	fprintf(stderr, "[!] Failed to open sources: %s\n", src_path.string().c_str());
    	return -1;
	}
	while (std::getline(file, line)){
    	if (line.empty() || line[0] == '#') continue;
    	
    	float val_x, val_y, val_z, val_s;
    	std::istringstream iss(line);
    	
    	if (iss >> val_x >> val_y >> val_z >> val_s){
        	i_src = (int)(val_x / h);
        	j_src = (int)(val_y / h);
        	k_src = (int)(val_z / h);
        	
        	if (i_src >= 0 && i_src < Nx && j_src >= 0 && j_src < Ny && k_src >= 0 && k_src < Nz){
        		// map point source(s) as intial condition(s) / source injection, no delay(s)
            	p0[flat3(i_src, j_src, k_src, Nx, Ny)] = val_s;
            	p1[flat3(i_src, j_src, k_src, Nx, Ny)] = -val_s;
            	
            	// push positions into source array
        		src_arr.push_back(val_x);
        		src_arr.push_back(val_y);
        		src_arr.push_back(val_z);
        		
        		// sound pressure values into pressure array
        		src_pressure.push_back(val_s);
        	}
    	}
	}
	file.close();
	file.clear();
	
    
    // parse receiver(s)
    
    int i_rec, j_rec, k_rec; // receiver position
    
    std::vector<int> i_recV, j_recV, k_recV; // vectors for multiple receivers discrete positions (called multiple times on main loop)
    std::vector<float> rec_arr; // receiver array on host with actual positions, 2D flattened
    
    file.open(rec_path);
	if (!file.is_open()){
    	fprintf(stderr, "[!] Failed to open receivers: %s\n", rec_path.string().c_str());
    	return -1;
	}
	while (std::getline(file, line)) {
    	if (line.empty() || line[0] == '#') continue;
    	float val_x, val_y, val_z;
    	std::istringstream iss(line);
    	if (iss >> val_x >> val_y >> val_z){
        	i_rec = (int)(val_x / h);
        	j_rec = (int)(val_y / h);
        	k_rec = (int)(val_z / h);
        	
        	if (!(i_rec >= 0 && i_rec < Nx && j_rec >= 0 && j_rec < Ny && k_rec >= 0 && k_rec < Nz)){
        		printf("\n[!] Receiver(s) outside domain, will now stop...\n");
            	return -1;
        	}
        	else{
        		i_recV.push_back(i_rec);
        		j_recV.push_back(j_rec);
        		k_recV.push_back(k_rec);
        		
        		// push positions into receiver array
        		rec_arr.push_back(val_x);
        		rec_arr.push_back(val_y);
        		rec_arr.push_back(val_z);
        	}
    	}
	}
	file.close();
	file.clear();
	
	
	// ----------------------- Define kernel parameters and copy sources to device --------------------------
	
	
	// Kernel launch parameters
	dim3 TP(32,4,4); // row-major order, 32*4*4 = 512 threads/block
	dim3 BP((Nx + TP.x - 1)/TP.x, (Ny + TP.y - 1)/TP.y, (Nz + TP.z - 1)/TP.z);

	// Copy initial data to device
    if (cudaMemcpy(d_p0, p0, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_u0, u0, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_v0, v0, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_w0, w0, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess){
    		printf("[!] CUDA error allocating device memory\n");
    	return -1;
    }
    if (cudaMemcpy(d_p1, p1, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_u1, u1, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_v1, v1, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_w1, w1, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess){
    		printf("[!] CUDA error allocating device memory\n");
    	return -1;
    }
	
	
	// --------------------------------- Memory assessment ---------------------------------
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("[!] CUDA error: %s\n", cudaGetErrorString(err));
	}
	
	size_t anim_step = (int)(0.025*Nt);
	if (anim_step == 0) anim_step = 1; // Nt < 40 would make n % anim_step divide by zero (fix this)
	size_t frames = Nt/anim_step; // make denominator smaller for more frames
	printf("\n[OK] anim_step = %d\n\n",anim_step);
	size_t anim_s = frames*N*sizeof(float); // animation size in bytess
	
	if ((anim_s/1000000000.0) > 5.2){
		printf("\n[!] Memory to animate is %f, larger than 5.2 GB, exiting...\n\n\n",anim_s/1000000000.0);
		return -1;
	}
	
	// Can handle much more memory if save animation in chuncks and assemble together:
	// e.g., the last sound pressure field is the initial field for the next instant in time.


	// ---------------------------------- animation arrays ---------------------------------
	
	int i0 = i_src;
	size_t stride_X = (size_t)Ny * (size_t)Nz;
	std::vector<float> p_X(stride_X); // reuse each frame
	
	int j0 = j_src;
	size_t stride_Y = (size_t)Nx * (size_t)Nz;
	std::vector<float> p_Y(stride_Y); // reuse each frame
	
	int k0 = k_src;
	size_t stride_Z = (size_t)Nx * (size_t)Ny;
	std::vector<float> p_Z(stride_Z); // reuse each frame
	
	// open BIN file for writing pressure fields
	FILE *file_pX = fopen(out_file("pressureX.bin").c_str(), "ab");
	if (file_pX == NULL) {
    	printf("[!] Error opening pressureX.bin file!\n");
    	exit(1);
	}
	FILE *file_pY = fopen(out_file("pressureY.bin").c_str(), "ab");
	if (file_pY == NULL) {
    	printf("[!] Error opening pressureY.bin file!\n");
    	exit(1);
	}
	FILE *file_pZ = fopen(out_file("pressureZ.bin").c_str(), "ab");
	if (file_pZ == NULL) {
    	printf("[!] Error opening pressureZ.bin file!\n");
    	exit(1);
	}
	
	// open BIN files for writing pressure at multiple receivers
	std::vector<std::ofstream> files_rec(i_recV.size());
	// i_recV.size() should be the same as j_recV.size() and k_recV.size()
	for (int ct = 0; ct < i_recV.size(); ct++){
    	std::string str = out_file("IR" + std::to_string(ct) + ".bin");
    	files_rec[ct].open(str, std::ios::app | std::ios::binary);
	}
	
	
	size_t pbin_size = 0; // for plotting animation correctly
	
	/*
	 *
	 *
	 * MAIN LOOP START
	 *
	 *
	 */
	
	printf("\n[OK] Simulation start\n\n");
	tic = clock();

	while(n < Nt){
				
		// Launch WENO3 kernel + SSP-RK3
				
		// (0)
			
			
		kernel_WENO_step0<<<BP, TP>>>(Nx, Ny, Nz, d_p0, d_u0, d_v0, d_w0, d_p1, d_u1, d_v1, d_w1, rho, c, h, dt, c_rl, eps, dd, beta_rl, d_S, U_PML, D_PML, R_PML, L_PML, F_PML, B_PML);
			
		kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u1, d_v1, d_w1, d_p1, Nx, Ny, Nz, h, d_S, d_oid, rho, c, n, dt, d_u0, d_v0, d_w0, d_p0, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 0.0f, 1.0f, 1.0f, phi, phi_old, phi_old, psi_1, psi_1_old, psi_1_old, psi_2, psi_2_old, psi_2_old); // right arrays are the same as middle (can be passed twice as dummy function arguments)
			
		// (1)
			
		kernel_WENO_step1<<<BP, TP>>>(Nx, Ny, Nz, d_p0, d_u0, d_v0, d_w0, d_p1, d_u1, d_v1, d_w1, d_p2, d_u2, d_v2, d_w2, rho, c, h, dt, c_rl, eps, dd, beta_rl, d_S, U_PML, D_PML, R_PML, L_PML, F_PML, B_PML);

		kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u2, d_v2, d_w2, d_p2, Nx, Ny, Nz, h, d_S, d_oid, rho, c, n, dt, d_u1, d_v1, d_w1, d_p1, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 3.0f, 1.0f, 4.0f, phi_nu, phi_old, phi, psi_1_nu, psi_1_old, psi_1, psi_2_nu, psi_2_old, psi_2);
			
		// (2)
			
		kernel_WENO_step2<<<BP, TP>>>(Nx, Ny, Nz, d_p0, d_u0, d_v0, d_w0, d_p1, d_u1, d_v1, d_w1, d_p2, d_u2, d_v2, d_w2, rho, c, h, dt, c_rl, eps, dd, beta_rl, d_S, U_PML, D_PML, R_PML, L_PML, F_PML, B_PML);
			
		kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u1, d_v1, d_w1, d_p1, Nx, Ny, Nz, h, d_S, d_oid, rho, c, n, dt, d_u2, d_v2, d_w2, d_p2, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 1.0f, 2.0f, 3.0f, phi, phi_old, phi_nu, psi_1, psi_1_old, psi_1_nu, psi_2, psi_2_old, psi_2_nu);

		// sync
		cudaDeviceSynchronize();

		// faster than cudammemcoy() DeviceToDevice
		std::swap(d_p0, d_p1);
		std::swap(d_u0, d_u1);
		std::swap(d_v0, d_v1);
		std::swap(d_w0, d_w1);
			
		std::swap(phi_old, phi);
		std::swap(psi_1_old, psi_1);
		std::swap(psi_2_old, psi_2);
			
		print_progress(n, Nt, 50); // gui.py loads progress by querrying the output of this function!
		
		// Copy result back to host for creating animation
		cudaMemcpy(p0, d_p0, N*sizeof(float), cudaMemcpyDeviceToHost);
			
		// write animation slices
		if (n % anim_step == 0){
				
			// YZ slice at i0: dst = j + Ny*k
			#pragma omp parallel for collapse(2) schedule(static)
			for (int k = 0; k < Nz; ++k){
  				for (int j = 0; j < Ny; ++j){
    				size_t dst = (size_t)j + (size_t)Ny * (size_t)k;
    				size_t src = (size_t)i0 + (size_t)Nx * ((size_t)j + (size_t)Ny * (size_t)k);
    				p_X[dst] = p0[src];
  				}
			}
			fwrite(p_X.data(), sizeof(float), (size_t)Ny*(size_t)Nz, file_pX);
			
			// XZ slice at j0: dst = i + Nx*k
			#pragma omp parallel for collapse(2) schedule(static)
			for (int k = 0; k < Nz; ++k){
  				for (int i = 0; i < Nx; ++i){
    				size_t dst = (size_t)i + (size_t)Nx * (size_t)k;
    				size_t src = (size_t)i + (size_t)Nx * ((size_t)j0 + (size_t)Ny * (size_t)k);
    				p_Y[dst] = p0[src];
  				}
			}
			fwrite(p_Y.data(), sizeof(float), (size_t)Nx*(size_t)Nz, file_pY);
			
			// XY slice in k0 (for z axis, contiguous in row-major order), faster the way below
			size_t start = (size_t)Nx * (size_t)Ny * (size_t)k0;
			std::memcpy(p_Z.data(), p0 + start, stride_Z * sizeof(float));
			fwrite(p_Z.data(), sizeof(float), stride_Z, file_pZ);
    			
    		// increment real size of binnaries
    		pbin_size += 1;
		}
        		
			
		// write to file for each receiver and iteration
		// i_recV.size() should be the same as j_recV.size() and k_recV.size()
		for (int ct = 0; ct < i_recV.size(); ct++){
			files_rec[ct].write(
				reinterpret_cast<const char*>(&p0[flat3(i_recV[ct], j_recV[ct], k_recV[ct], Nx, Ny)]),
				sizeof(float)
			);
		}
        		
		n += 1; // update time-step
	}
		
    // Free host memory
    delete[] p0; delete[] u0; delete[] v0; delete[] w0;
    delete[] p1; delete[] u1; delete[] v1; delete[] w1;
    	
    // Free device memory
    cudaFree(d_p0); cudaFree(d_u0); cudaFree(d_v0); cudaFree(d_w0);
    cudaFree(d_p1); cudaFree(d_u1); cudaFree(d_v1); cudaFree(d_w1);
    cudaFree(d_p2); cudaFree(d_u2); cudaFree(d_v2); cudaFree(d_w2);
    cudaFree(c_rl); cudaFree(dd);
    
    fflush(file_pX); fflush(file_pY); fflush(file_pZ);
    
    toc = clock();
    
    elapsed_time = ((double)(toc - tic)/60.0)/CLOCKS_PER_SEC; // in minutes
    printf("\n[OK] Simulation time: %f min \n\n", elapsed_time); 
    
    fclose(file_pX); fclose(file_pY); fclose(file_pZ);
    
    FILE *header; // file for data output
	header = fopen(out_file("header.txt").c_str(), "w");
	if (header != NULL){
    	fprintf(header, "%d %d %d %d %f %f %f %d %d %f %s %f %d\n", Nx, Ny, Nz, Nt, Lx, Ly, Lz, pbin_size, anim_step, dt, scene_str.c_str(), h, (int)(tol/h));
    	fclose(header);
    }
    
    printf("\n[OK] Printed header.txt for plot.py\n");
    
    // ----- free remaining arrays -----
    
    cudaFree(d_Rinf);
    cudaFree(d_Ak); cudaFree(d_zeta);
    cudaFree(d_Bl); cudaFree(d_Cl); cudaFree(d_alpha); cudaFree(d_beta);
    
    cudaFree(phi); cudaFree(psi_1); cudaFree(psi_2);
    cudaFree(phi_old); cudaFree(psi_1_old); cudaFree(psi_2_old);
    cudaFree(phi_nu); cudaFree(psi_1_nu); cudaFree(psi_2_nu);
    
    cudaFree(beta_rl);
    cudaFree(d_bmap);
    
    free(S);
    free(oid);
    free(h_bmap);
    
    cudaFree(d_S);
    cudaFree(d_oid);
    
    printf("\n[OK] Freed wave-based arrays\n");  
    
    // --------------------------------------------- ISM Section -----------------------------------------------
	
	// ISM max reflection order: config.txt line 13 (conf_arr[12]); 3 if absent
	ISMConfig cfg;
	cfg.frame = ISMFrame::OBJECT_SPACE; // apply (p+shift)*scale to src/rec, identical to the walls
	int max_order = (conf_arr.size() > 12) ? (int)conf_arr[12] : 3;
	if (max_order < 0) max_order = 0;
	printf("\n[OK] ISM max_order = %d\n", max_order);
	ISMResult Q = ism_solve(sc, src_arr, rec_arr, {}, Z, max_order, src_pressure, cfg); // gets main values for IRs

	// Ray-traced late tail: build BVH scene ONCE (before the mesh is freed); the
	// tracer continues the ISM past max_order into the reverberant tail.
	RTScene rtS = rt_build(sc);

	scene_free(&sc); // free scene here: ism_solve and rt_build are its last users

	
	// determine number of valid bands for ISM / Ray-Tracing (easier array allocation later on)
	int N_bands = 8; // number of bands for ISM / Ray-Tracing, start with total
	for (int i = 0; i < 8; i++){
		if (f_cutoff <= octave_bands[i]){
			N_bands -= i;
			break;
		}
	}
	
	fs = 44100.0; // new sample rate
	dt = 1.0/fs; // new time-step size
	Nt = (int)(t_max/dt); // new number of samples
	int Nw = 1024; // Hann window size for the fractional-delay deposit (sub-sample arrival timing)
	
	// ----- ray-traced late reverberant tail (continues the ISM past max_order) -----
	// Flat absorption table for the BVH tracer: alpha_gb[gid*8 + band]. Z is indexed
	// by material == group_id (names[gid+1]), exactly like the .fit loop above.
	std::vector<float> alpha_gb((size_t)rtS.ngroups * 8, 0.0f);
	for (int g = 0; g < rtS.ngroups && g < (int)Z.size(); g++){
		for (int b = 0; b < 8; ++b){
			alpha_gb[(size_t)g*8 + b] = Z[g].alpha[b];
		}
	}

	// recv_radius sets the detection-sphere size; too small starves the tail, too big
	// smears it. Nrays trades tail smoothness for time. Raise both if the tail is noisy.
	// ray count: user value from config.txt line 15 (conf_arr[14]) if present, else
	// a safe default. The GUI seeds that field from Vol.bin (Vorlander <k>).
	int   rt_nrays       = (conf_arr.size() > 14) ? std::max(1000, (int)conf_arr[14]) : 20000;
	int   rt_max_order   = 200;	// energy floor kills most rays far sooner
	float rt_recv_radius = 0.15f; // [m] receiver radius (keep it around 0.10 m to match ray-tracer)
	// per-group scattering: read the single value from input/sctr/<name>.sctr and expand
	// to 8 octave bands (rt_scatter_bands). names[g+1] is material g (names[0]=BoundingBox).
	std::filesystem::path sctr_dir = project_root / "input" / "sctr";
	std::vector<float> sctr_gb((size_t)rtS.ngroups * 8, 0.0f);
	std::vector<float> sctr_g((size_t)rtS.ngroups, 0.01f);   // GLOBAL s per group (the stochastic draw)
	for (int g = 0; g < rtS.ngroups; g++){
		std::string mname = (g + 1 < (int)names.size()) ? names[g + 1] : std::string();
		float gval = 0.01f;   // default scattering if no .sctr file
		if (!mname.empty()){
			std::ifstream sf((sctr_dir / (mname + ".sctr")).string());
			std::string ln;
			while (std::getline(sf, ln)){
				size_t hp = ln.find('#'); if (hp != std::string::npos) ln.erase(hp);
				std::istringstream is(ln); float v;
				if (is >> v){
					gval = v;
					break;
				}
			}
		}
		sctr_g[(size_t)g] = gval;
		// rt_scatter_bands floors every band at ~0.01 (its first row), so gval<=0 would still
		// inject ~1% diffuse through the next-event deposit while the stochastic draw (sctr_g)
		// is off. Force a TRUE zero here so "no scattering" is actually none (validation).
		if (gval <= 0.f){
			for (int b = 0; b < 8; ++b) sctr_gb[(size_t)g*8 + b] = 0.f;
		} else {
			float s8[8]; rt_scatter_bands(gval, s8);
			for (int b = 0; b < 8; ++b) sctr_gb[(size_t)g*8 + b] = s8[b];
		}
	}

	// RT late field: two ENERGY buffers (specular / diffuse), rooted per bin at deposit.
	std::vector<float> e_spec, e_diff;
	rt_trace_late(rtS, src_arr, src_pressure, rec_arr, alpha_gb.data(), m_air.data(), sctr_gb.data(), sctr_g.data(),
	              c, fs, (int)Nt, rt_nrays, rt_max_order, rt_recv_radius, e_spec, e_diff);

	// compute IRs
	// for each band
	for (int b = 0; b < 8; b++){
		if (f_cutoff > octave_bands[b]) continue; // skip band
		
		// allocate memory for high frequency IRs: for each rec, first index sample is time, second is
		// receiver's index (flattened)
		std::vector<float> p_hf(Nt * (size_t)Q.Nr);
		// Fractional-delay deposit (windowed sinc), ref. Reuben Thomas (Thesis), Fu & Li (2016).
		// Places an arrival at its exact SUB-SAMPLE time -- this is a TIMING tool, independent
		// of coherence (coherence is set by the sign of g). Incoherent audible IR: deposit each
		// arrival's magnitude with a random sign, deposit(tau, rt_sign(...)*mag, ...). During the
		// energy-echogram validation below it is bypassed (arrivals hard-binned as energy), hence
		// uncomment the deposit() lines to switch the ISM to the timed path.
		auto deposit = [&](float tau, float g, int rr, float dist){
			int tt = (int)tau;
			for (int n0 = -(int)(0.5*Nw); n0 <= (int)(0.5*Nw); n0++){
				int nn = tt + n0;
				if (nn < 0 || nn >= (int)Nt) continue;
				float xx = nn - tau;
				p_hf[flat(nn,rr,Nt)] += g * 0.5 * (1.0 + std::cos(2.0*M_PI*xx/Nw)) * sinc(M_PI*xx);
			}
		};

		// for each receiver
		for (int r = 0; r < Q.Nr; r++){
			int t_cross = 0.0; // crossover time index, for late tail reverberation
			std::vector<int> arrivals; // sample index of every deposited arrival, for the crossover
			//std::vector<float> e_ism(Nt, 0.0); // INCOHERENT ISM energy per bin (GA: sum energy, not pressure)
			// sum contributions of sources (real and virtual)
			for (int s = 0; s < Q.Ns; s++){
				// direct: guard the deposit only - do NOT skip the image loop
    			float d_0 = Q.direct_at(s, r); // 0.0f if blocked
    			if (d_0 > 0.0f){
        			float tau_0 = fs * d_0 / c;
        			int t_0 = (int)tau_0;
        			if (t_0 < Nt){
        				float gS_0 = Q.src_pressure[s] / (4.0*M_PI*d_0);
        				gS_0 *= std::exp(-m_air[b]*d_0);
        				// INCOHERENT (GA) energy deposit at the integer sample (validation path).
        				// no fractional delay
            			//e_ism[t_0] += (float)gS_0 * gS_0;
            			// fractional delay
            			deposit(tau_0,  rt_sign((unsigned)(t_0*2654435761u+r*40503u+b*97u))*gS_0, r, d_0);

            			if (t_0 > t_cross) t_cross = t_0;
            			arrivals.push_back(t_0);
        			}
    			}

				// sum contributions of image sources for a given real source
				// for each image-source 'iS' in container Q.images_at()
				for (const ISMImagePath& iS : Q.images_at(s, r)){
					float d = iS.distance;
					float tau = fs * d / c;
					int t = (int)tau;
					if (t >= Nt) continue; // safety guard
					if (t > t_cross) t_cross = t;
					arrivals.push_back(t);
					float P = 1.0; // iterated product
					// for each surface k crossed by the image source
					for(int k = 0; k < iS.angle_rad.size(); k++){
						float theta = iS.angle_rad[k];
						//float theta = 0.0; // amplitude validation
						float R = R_oblique(iS.alpha[k][b], std::cos(theta));
						// scattered fraction leaves the specular path: keep sqrt(1-s) of the
						// pressure (energy x (1-s)), matching the RT (1-s) split.
						int gk = iS.group[k];
						if (gk >= 0 && gk < rtS.ngroups)
							R *= std::sqrt(std::max(0.0f, 1.0f - sctr_gb[(size_t)gk*8 + b]));
						P *= R;
					}
					// .amp is a factor rangining [0,1] that handles curved surfaces (part of the
					// parser's heuristics)

					float gS = Q.src_pressure[s] * iS.amp * P / (4.0*M_PI*d);
					gS *= std::exp(-m_air[b]*d);
					// INCOHERENT (GA) energy deposit at the integer sample (validation path).
					// no fractional delay
					//e_ism[t] += (float)gS * gS;
					// fractional delay
					deposit(tau, rt_sign((unsigned)(t*2654435761u+r*40503u+b*97u))*gS, r, d);
				}
			}
			
			// Crossover where the computed arrival count falls 2 sigma (Poisson) below the
			// theoretical count Ns*(4/3)*pi*(c*t)^3/V: the ISM is complete before that point
			// and misses a significant echo density after it. Fallback: last arrival (t_cross
			// as accumulated above). Later ISM arrivals still sit on top as specular echoes.
			std::sort(arrivals.begin(), arrivals.end());
			for (int i = 0; i < (int)arrivals.size(); i++){
				float ct = c * (arrivals[i] + 1) / fs;
				float theo = Q.Ns * 4.0f * (float)M_PI * ct*ct*ct / (3.0f * V);
				if (theo - (i + 1) > 2.0f * std::sqrt(theo)){
					t_cross = arrivals[i];
					break;
				}
			}
			
			// ray-tracer late field. e_diff (scattered) enters at ALL times -- it is energy
			// the specular ISM lacks; e_spec (specular) only AFTER the ISM crossover, to
			// avoid duplicating the ISM early reflections. Sum the energies per bin, root
			// -> pressure, deposit as incoherent noise (random phase / sign).
			// INCOHERENT (geometrical-acoustics) combine: ISM + RT scattered + RT specular
			// energies ADD per bin, then sqrt -> pressure magnitude. Phase is discarded, so
			// coinciding arrivals add in energy (sqrt(sum a^2)), matching the ray-tracer and
			// the incoherent rir_ism.py. Random sign for the audible IR (comment for
			// amplitude validation).
			for (int t = 1; t < (int)Nt; t++){
				//float e = e_ism[t] + e_diff[((size_t)b*Q.Nr + r)*Nt + t]; // amplitude valiation
				float e = e_diff[((size_t)b*Q.Nr + r)*Nt + t];
				if (t > t_cross) e += e_spec[((size_t)b*Q.Nr + r)*Nt + t];
				if (e <= 0.0) continue;
				p_hf[flat(t, r, Nt)] += rt_sign((unsigned)(t*2654435761u + r*40503u + b*97u)) * (float)std::sqrt(e);
				//p_hf[flat(t, r, Nt)] += (float)std::sqrt(e); // amplitude validation
			}

			// this receiver is done for this band -> write its slice to associated binary file
        	std::string fname = out_file("RIR" + std::to_string(r) + "_"
                                    		+ std::to_string((int)octave_bands[b]) + ".bin");
        	FILE* f = std::fopen(fname.c_str(), "wb");
        	if (f){
           		std::fwrite(p_hf.data() + (size_t)r*Nt, sizeof(float), (size_t)Nt, f);
           		std::fclose(f);
        	} else {
           		std::fprintf(stderr, "\n[ISM] could not open %s for writing\n", fname.c_str());
        	}
		}
	}
	
	printf("\n[OK] Ray-tracer and ISM finished\n");

    // ---------------------------------------------------------------------------------------------------------
    
    printf("\n[OK] Simulation end\n\n"); 
    
    return 0;
}
