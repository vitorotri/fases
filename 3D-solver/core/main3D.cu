#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <new>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <sstream>
#include <unistd.h>
#include <cuda_runtime.h>
#include "ism3D.h"
#include "functions3D.cuh"

unsigned long long available_ram_bytes() {
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    return (unsigned long long)pages * (unsigned long long)page_size;
}

int main(int argc, char *argv[]){
	if (argc < 5) {
        printf("Usage: %s <scene> <config> <src> <rec>\n", argv[0]);
        return 1;
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
    std::filesystem::path nMat_path = config_path.parent_path() / "nMat";

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "[!] Cannot create output directory: " << output_dir
                  << " (" << ec.message() << ")\n";
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
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("Error opening file");
        return -1;
    }
    int nObj = 0; // number of objects in the scene (initialization)
    //int file_opt = 0; // o for OBJ, 1 for VOX/PLY (initialization)
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
    std::cout << "\n CUDA_VISIBLE_DEVICES="
              << (visible_devices ? visible_devices : "<unset>") << std::endl;

    int nDevices = 0;
    cudaError_t devErr = cudaGetDeviceCount(&nDevices);
    if (devErr != cudaSuccess) {
        std::cerr << "\n[!] FAILED: cudaGetDeviceCount() returned "
                  << static_cast<int>(devErr) << " ("
                  << cudaGetErrorName(devErr) << "): "
                  << cudaGetErrorString(devErr) << "\n";

        if (devErr == cudaErrorInsufficientDriver) {
            std::cerr << "[!] The executable was probably built with a CUDA runtime newer "
                         "than the installed NVIDIA driver supports. Rebuild with CUDA 11.x "
                         "for driver 470.x, or update the NVIDIA driver.\n";
        } else if (devErr == cudaErrorNoDevice) {
            std::cerr << "[!] The CUDA runtime did not expose any CUDA-capable GPU to this process.\n";
        } else if (devErr == cudaErrorInitializationError) {
            std::cerr << "[!] CUDA runtime initialization failed. Check driver state, permissions, "
                         "and whether another environment/library path is interfering.\n";
        }

        std::cerr << std::endl;
        return -1;
    }

    if (nDevices <= 0){
        std::cerr << "\n[!] FAILED: no CUDA devices found by the CUDA runtime\n\n";
        return -1;
    }

    int selectedDevice = 0;
    cudaError_t setErr = cudaSetDevice(selectedDevice);
    if (setErr != cudaSuccess) {
        std::cerr << "\n[!] FAILED: cudaSetDevice(" << selectedDevice << ") returned "
                  << static_cast<int>(setErr) << " ("
                  << cudaGetErrorName(setErr) << "): "
                  << cudaGetErrorString(setErr) << "\n\n";
        return -1;
    }

    // Force context creation now, so device/runtime problems are reported here
    // rather than much later as a misleading allocation/kernel failure.
    cudaError_t initErr = cudaFree(0);
    if (initErr != cudaSuccess) {
        std::cerr << "\n[!] FAILED: CUDA context initialization returned "
                  << static_cast<int>(initErr) << " ("
                  << cudaGetErrorName(initErr) << "): "
                  << cudaGetErrorString(initErr) << "\n\n";
        return -1;
    }

    for (int i = 0; i < nDevices; ++i) {
        cudaDeviceProp prop{};
        cudaError_t propErr = cudaGetDeviceProperties(&prop, i);
        if (propErr != cudaSuccess) {
            std::cerr << "\n[!] WARNING: cudaGetDeviceProperties(" << i << ") returned "
                      << static_cast<int>(propErr) << " ("
                      << cudaGetErrorName(propErr) << "): "
                      << cudaGetErrorString(propErr) << "\n";
            continue;
        }

        std::cout << "\n Device Number: " << i << (i == selectedDevice ? " [selected]" : "") << std::endl;
        std::cout << " Device Name: " << prop.name << std::endl;
        std::cout << " Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << " Total Global Memory (bytes): " << prop.totalGlobalMem << std::endl;
        std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << " Memory Clock Rate (kHz): " << prop.memoryClockRate << std::endl;
        std::cout << " Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << " L2 Cache Size (bytes): " << prop.l2CacheSize << std::endl;
        std::cout << " Shared Memory per Block (bytes): " << prop.sharedMemPerBlock << std::endl;
        std::cout << " Shared Memory per SM (bytes): " << prop.sharedMemPerMultiprocessor << std::endl;
    }
    
    // -------------------------------------- process OBJ scene ------------------------------------
    
    float A_tot; // total surface area of domain
    std::vector<float> A; // vector of surface area per material
    std::vector<std::string> names; // global naming
    Scene3D sc;
   	if (ends_with_dotobj(path)){
   	
   		printf("\n[OK] processing scene...\n");

		if(!parse_obj3d(path, &sc, h)) return -1;
		
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
    
    FILE *pre_header; // file for data output (sdf_plot.py TST)
	pre_header = fopen(out_file("pre_header.txt").c_str(), "w");
	if (pre_header != NULL){
    	fprintf(pre_header, "%d %d %d %f %d\n", Nx, Ny, Nz, h, (int)(tol/h));
    }
    fclose(pre_header);
    
    /*
    // force for test hypotetical scene with BCs without parsing (no need to comment above)
    Lx = 3.0; Ly = 4.0; Lz = 2.8;
    Nx = (int)(Lx/h); Ny = (int)(Ly/h); Nz = (int)(Lz/h);
    */
    
    // ----------------------------- Print scene parameters ---------------------------------------
    
    size_t N = Nx * Ny * Nz; // number of grid cells
    
    printf("\n[OK] Nx = %d  Ny = %d  Nz = %d\n", Nx, Ny, Nz);
    printf("\n[Ok] Lx = %f  Ly = %f  Lz = %f\n", Lx, Ly, Lz);
    
    // ----------------------- LOAD filter coefficiets for materials ------------------------------------
    
    // ----- host arrays for materials -----
    
    float *Rinf;
    float *Ak, *zeta;
    float *Bl, *Cl, *alpha, *beta;
    
    int nMat = nObj - 1; // exclude bounding box
    // test for using same material for different objects when they match
    /*
    int32_t nMat = 0;
    std::ifstream f_nMat(nMat_path, std::ios::binary);
    f_nMat.read(reinterpret_cast<char*>(&nMat), sizeof(int32_t));
    std::cout << "\n[OK] nMat = " << nMat << "\n";
    */
    try {
    	Rinf = new float[nMat]();
    } catch (const std::bad_alloc&){
    	std::cerr << "\n[!] Allocation for R_inf[] failed, not enough memory\n";
    }
    
    // read K_m and L_m from file
    
    int K_m, L_m; // k_max and L_max from all the fits
    
    // read poles.txt file
    file.open(poles_path);
    if (!file.is_open()) {
        std::cerr << "[!] Cannot open poles file: " << poles_path << std::endl;
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
    	std::cerr << "\n[!] Allocation for remaining materials' coefficients, not enough memory\n";
    }
    
    printf("\nChecking materials\n\n");
    for (size_t i = 0; i < names.size(); ++i){
  		std::cout << i << " -> " << names[i] << "\n";
  	}

    
    for (int m = 0; m < nMat; ++m) {
    	std::string filename = (fit_dir / (names[m+1] + ".fit")).string(); // names[0] = BoundingBox
    	//std::cout << "\nReading: " << filename << "\n";	
    	read_fit_file(filename, m, K_m, L_m, Rinf, Ak, zeta, Bl, Cl, alpha, beta);
    	/*
		// simple print test
    	std::printf("Rinf[%d] = %.15f\n", m, (float)Rinf[m]);
    	std::printf("Ak(0,%d)  = %.15f   zeta(0,%d) = %.15f\n",
                	m, (float)Ak[flat(0, m, K_m)],
                	m, (float)zeta[flat(0, m, K_m)]);
    	std::printf("Ak(1,%d)  = %.15f   zeta(1,%d) = %.15f\n\n",
                	m, (float)Ak[flat(1, m, K_m)],
                	m, (float)zeta[flat(1, m, K_m)]);
        */
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


    // ------------------------------------- atmospheric values ------------------------------------------------
	
	// should to to config later
	
	float R_spec = 287.05; // specific gas constant for air [J] [kg]^-1 [K]^-1
	float gamma = 1.4; // specific heat ratio for air
	
	rho = 1.225; // average sea level density for air [kg/m3]. For dry air, proportional to p0/R_spec*T
	Temp = conf_arr[3] + 273.15f; // temperature [K] = Celsius_temp + 273.15
	c = sqrt(gamma*R_spec*Temp); // speed of sound [m/s]
	// c = sqrt(K/rho) in case of liquids
    
    // later for ISM parsing, must call here to estimate T60s and give user f_Schroeder, since it depends on c
    // as well
    ISMConfig cfg;
	cfg.frame = ISMFrame::OBJECT_SPACE; // apply (p+shift)*scale to src/rec, identical to the walls
    auto Z = ism_load_impedances_layered(imp_dir, names, mat_dir); // names[0]="BoundingBox"; handles the +1
    // compute estimated T60's
	std::vector<float> T60(8, 0.5f);
	estimate_T60s(T60, V, c, A_tot, A, Z);
	printf("\n[OK] estimated T60 per band [s]:");
	for (int b = 0; b < 8; b++) printf(" %.3f", T60[b]);
	printf("\n");
    
    printf("\n	c = %.2f m/s\n",c);
    
    // ---------------------------------------------------------------------------------------------------------

    // Allocate main host memory arrays and initilize to 0.0f
    unsigned long long avail = available_ram_bytes();
    std::size_t needed = 4 * N * sizeof(float);
    if (needed > 0.7 * avail) {
        std::cerr << "\n[!] Requested memory too large, skipping allocation\nTry a larger grid spacing or smaller cutoff frequency";
        return -1;
    }
    try {
    	p0 = new float[N](); // pressure deviations from p0
    	u0 = new float[N](); // x component of particle velocity 
    	v0 = new float[N](); // y component of particle velocity 
    	w0 = new float[N](); // z component of particle velocity
    } catch (const std::bad_alloc&){
    	std::cerr << "\n[!] Allocation for initial pressures and velocities failed, not enough memory\nTry a larger grid spacing or smaller cutoff frequency\n";
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
	
	float dm; // device memory
	
	float eps = 1e-7; // the smaller the eps, the more precise the BCs work, but cannot be the smallest otherwise have inf. In C, floats can go up to 1e-38. With 1e-5 workd really bad, with 1e-10 works well, depite below machine precision for float, so results may be wrong even with convergence!!!
	
    if (needed > 0.7 * avail) {
        std::cerr << "\n[!] Requested memory too large, skipping allocation\nTry a larger grid spacing or smaller cutoff frequency\n";
        return -1;
    }
	try {
		p1 = new float[N](); // sound pressure
    	u1 = new float[N](); // x component of particle velocity 
    	v1 = new float[N](); // y component of particle velocity
    	w1 = new float[N](); // z component of particle velocity
    } catch (const std::bad_alloc&){
    	std::cerr << "\n[!] Allocation for remaining host arrays failed, not enough memory\nTry a larger grid spacing or smaller cutoff frequency\n";
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
	
	dm = (12*N + 12+9+3 + nMat*(2*K_m + 4*L_m) + nB*(3*K_m + 6*L_m))*sizeof(float);
	dm += N*(sizeof(uint8_t) + sizeof(int)); // oid and bmap
		
	cudaDeviceSynchronize();

	
	// --------------------- Parse sources and receiver positions from argv[3] and argv[4] ---------------------
    
    // parse source(s)
    
    //float *p_time = new float[Nt](); // timed single source
    //float *d_p_time;
    int i_src, j_src, k_src; // discretized position of source
    
    std::vector<float> src_arr; // source array on host
    std::vector<float> src_pressure; // sound pressure values for sources
    
    //float src_y; // for raised cosine
    
    file.open(src_path);
	if (!file.is_open()){
    	std::cerr << "[!] Failed to open sources: " << src_path << "\n";
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
            	
            	// single timed source
            	//p_time[0] = val_s;
            	//p_time[1] = -val_s;
            	//src_y = val_y;
            	//p_time[2] = 0.25 * val_s;
            	
            	// single plane wave
            	//for (int i = 0; i < Nx; i++){
            	//	for (int k = 0; k < Nz; k++){
            	//		p0[flat3(i, j_src, k, Nx, Ny)] = 0.1f*val_s;
            	//		u0[flat3(i, j_src, k, Nx, Ny)] = val_s;
            	//	}
            	//}
        	}
    	}
	}
	file.close();
	file.clear();
	
	/*
	// raised cosine
	float W_full = 0.04f;
	float ww = 0.5f * W_full;
	float A = 0.1f;
    for (int k = 0; k < Nz; ++k){
    	for (int j = 0; j < Ny; ++j){
        	for (int i = 0; i < Nx; ++i){
            	// y-location of cell center
            	float yy = (j + 0.5f) * h;
	
            	float g  = raised_cosine(yy, src_y, ww);
            	float pI = A * g;
	
            	int Cijk = flat3(i, j, k, Nx, Ny);
	
            	p0[Cijk] = pI;
            	v0[Cijk] = pI / (rho * c);  // +y traveling wave
        	}
    	}
	}
	*/
	/*
	// oscillating source
	for (int nn = 0; nn < Nt; nn++){
		p_time[nn] = sin(2.0*M_PI*1000.0*nn*dt);
	}
    */
    
    // parse receiver(s)
    
    int i_rec, j_rec, k_rec; // receiver position
    
    std::vector<int> i_recV, j_recV, k_recV; // vectors for multiple receivers discrete positions (called multiple times on main loop)
    std::vector<float> rec_arr; // receiver array on host with actual positions, 2D flattened
    
    file.open(rec_path);
	if (!file.is_open()){
    	std::cerr << "[!] Failed to open receivers: " << rec_path << "\n";
    	return -1;
	}
	while (std::getline(file, line)) {
    	if (line.empty() || line[0] == '#') continue;
    	float val_x, val_y, val_z;
    	std::istringstream iss(line);
    	if (iss >> val_x >> val_y >> val_z){
    		//printf("\nrec values ---> %f %f %f\n", val_x, val_y, val_z);
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
	
	/*
	// occupancy check
	int blockSize1D;
	int minGridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize1D, kernel_WENO_step0, 0, 0);
	printf("\nOptimal block size (1D): %d\n", blockSize1D); // 512 on the laptop GTX 1050
	printf("Minimum grid size for full occupancy: %d\n", minGridSize);
	*/
	
	// Kernel launch parameters
	dim3 TP(32,4,4); // row-major order. From the occupancy output right above, 32 * 4 * 4 = 512
	dim3 BP((Nx + TP.x - 1)/TP.x, (Ny + TP.y - 1)/TP.y, (Nz + TP.z - 1)/TP.z);

	// Copy initial data to device if not using p_time[]
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
	
	// if using p_time
    //cudaMalloc(&d_p_time, Nt*sizeof(float));
    //cudaMemcpy(d_p_time, p_time, Nt*sizeof(float), cudaMemcpyHostToDevice);
	
	// --------------------------------- Memory assessment ---------------------------------
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("[!] CUDA error: %s\n", cudaGetErrorString(err));
	}
	printf("\n[OK] Memory to allocate in device is %f GB\n", dm/(1000000000));
	
	//printf("float is %zu", sizeof(float));
	size_t anim_step = (int)(0.025*Nt);
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
	
	//int i0 = Nx / 2;
	int i0 = i_src;
	size_t stride_X = (size_t)Ny * (size_t)Nz;
	std::vector<float> p_X(stride_X); // reuse each frame
	
	//int j0 = Ny / 2;
	int j0 = j_src;
	size_t stride_Y = (size_t)Nx * (size_t)Nz;
	std::vector<float> p_Y(stride_Y); // reuse each frame
	
	//int k0 = Nz / 2;
	int k0 = k_src;
	size_t stride_Z = (size_t)Nx * (size_t)Ny;
	std::vector<float> p_Z(stride_Z); // reuse each frame
	
	// -------------------------------------- Initial TV ------------------------------------
	
	float TV_p_, TV_u_, TV_v_, TV_w_, TV_0, TV_1;
	
	// compute initial Total Variations if not using p_time[]
	
	TV_p_ = TV(p0, Nx, Ny, Nz, h);
	TV_u_ = TV(u0, Nx, Ny, Nz, h);
	TV_v_ = TV(v0, Nx, Ny, Nz, h);
	TV_w_ = TV(w0, Nx, Ny, Nz, h);
	
	TV_0 = TV_p_ + TV_u_ + TV_v_ + TV_w_;
	
	// -------------------------------------------------------------------------------------

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
	
	//Nt = 100; // for testing
	
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
			
		//kernel_inject_source<<<1,1>>>(d_p0, d_p_time, i_src, j_src, k_src, Nx, Ny, n);
			
		kernel_WENO_step0<<<BP, TP>>>(Nx, Ny, Nz, d_p0, d_u0, d_v0, d_w0, d_p1, d_u1, d_v1, d_w1, rho, c, h, dt, c_rl, eps, dd, beta_rl, d_S, U_PML, D_PML, R_PML, L_PML, F_PML, B_PML);
			
		kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u1, d_v1, d_w1, d_p1, Nx, Ny, Nz, h, d_S, d_oid, rho, c, n, dt, d_u0, d_v0, d_w0, d_p0, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 0.0f, 1.0f, 1.0f, phi, phi_old, phi_old, psi_1, psi_1_old, psi_1_old, psi_2, psi_2_old, psi_2_old); // right arrays are the same as middle (can be passed twice as dummy function arguments)
		/*
		kernel_apply_periodic_x<<<BPx, TPx>>>(d_p1, d_u1, d_v1, d_w1, Nx, Ny, Nz);
		kernel_apply_periodic_z<<<BPz, TPz>>>(d_p1, d_u1, d_v1, d_w1, Nx, Ny, Nz);
		*/
		//kernel_test_BCs_no_parsing<<<BP, TP>>>(Nx, Ny, Nz, d_p1, d_u1, d_v1, d_w1);
			
		// (1)
			
		kernel_WENO_step1<<<BP, TP>>>(Nx, Ny, Nz, d_p0, d_u0, d_v0, d_w0, d_p1, d_u1, d_v1, d_w1, d_p2, d_u2, d_v2, d_w2, rho, c, h, dt, c_rl, eps, dd, beta_rl, d_S, U_PML, D_PML, R_PML, L_PML, F_PML, B_PML);

		kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u2, d_v2, d_w2, d_p2, Nx, Ny, Nz, h, d_S, d_oid, rho, c, n, dt, d_u1, d_v1, d_w1, d_p1, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 3.0f, 1.0f, 4.0f, phi_nu, phi_old, phi, psi_1_nu, psi_1_old, psi_1, psi_2_nu, psi_2_old, psi_2);
		/*
		kernel_apply_periodic_x<<<BPx, TPx>>>(d_p2, d_u2, d_v2, d_w2, Nx, Ny, Nz);
		kernel_apply_periodic_z<<<BPz, TPz>>>(d_p2, d_u2, d_v2, d_w2, Nx, Ny, Nz);
		*/
		//kernel_test_BCs_no_parsing<<<BP, TP>>>(Nx, Ny, Nz, d_p2, d_u2, d_v2, d_w2);
			
		// (2)
			
		kernel_WENO_step2<<<BP, TP>>>(Nx, Ny, Nz, d_p0, d_u0, d_v0, d_w0, d_p1, d_u1, d_v1, d_w1, d_p2, d_u2, d_v2, d_w2, rho, c, h, dt, c_rl, eps, dd, beta_rl, d_S, U_PML, D_PML, R_PML, L_PML, F_PML, B_PML);
			
		kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u1, d_v1, d_w1, d_p1, Nx, Ny, Nz, h, d_S, d_oid, rho, c, n, dt, d_u2, d_v2, d_w2, d_p2, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 1.0f, 2.0f, 3.0f, phi, phi_old, phi_nu, psi_1, psi_1_old, psi_1_nu, psi_2, psi_2_old, psi_2_nu);
		/*
		kernel_apply_periodic_x<<<BPx, TPx>>>(d_p1, d_u1, d_v1, d_w1, Nx, Ny, Nz);
		kernel_apply_periodic_z<<<BPz, TPz>>>(d_p1, d_u1, d_v1, d_w1, Nx, Ny, Nz);
		*/
		//kernel_test_BCs_no_parsing<<<BP, TP>>>(Nx, Ny, Nz, d_p1, d_u1, d_v1, d_w1);

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
        		
        // write to file for single receiver IR
		//fwrite(&p0[flat3(i_rec, j_rec, k_rec, Nx, Ny)], sizeof(float), 1, f_rec);
			
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
		
	// Copy final results back to host (for computing TVs / Energy)
	cudaMemcpy(p0, d_p0, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(u0, d_u0, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(v0, d_v0, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(w0, d_w0, N*sizeof(float), cudaMemcpyDeviceToHost);
	/*
	TV_p_ = TV(p0, Nx, Ny, Nz, h);
	TV_u_ = TV(u0, Nx, Ny, Nz, h);
	TV_v_ = TV(v0, Nx, Ny, Nz, h);
	TV_w_ = TV(w0, Nx, Ny, Nz, h);
		
	TV_1 = TV_p_ + TV_u_ + TV_v_ + TV_w_;
	printf("\n\nTV_0 = %f\n", TV_0);
	printf("\nTV_1 = %f\n", TV_1);
		
	is_TVD(TV_0, TV_1);
	*/
    // Free host memory
    delete[] p0; delete[] u0; delete[] v0; delete[] w0;
    delete[] p1; delete[] u1; delete[] v1; delete[] w1;
    //delete[] p2; delete[] u2; delete[] v2; delete[] w2;
    	
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
    files_rec.clear();
    fclose(fp);
    
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
    
    printf("\n[OK] Freed arrays\n");  
    
    // --------------------------------------------- ISM Section -----------------------------------------------
    
	// ISM max reflection order: config.txt line 13 (conf_arr[12]); 3 if absent (older configs)
	int max_order = (conf_arr.size() > 12) ? (int)conf_arr[12] : 3;
	if (max_order < 0) max_order = 0;
	printf("\n[OK] ISM max_order = %d\n", max_order);
	ISMResult Q = ism_solve(sc, src_arr, rec_arr, {}, Z, max_order, src_pressure, cfg); // gets main values for IRs
	
	scene_free(&sc); // free scene
	
	// determine number of valid bands for ISM (easier array allocation later on)
	int N_bands = 8; // number of bands for ISM, start with total
	float octave_bands[8] = {63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0};
	for (int i = 0; i < 8; i++){
		if (f_cutoff <= octave_bands[i]){
			N_bands -= i;
			break;
		}
	}
	//std::cout << "\nN_bands = " << N_bands << "\n";
	
	fs = 44100.0; // new sample rate
	dt = 1.0/fs; // new time-step size
	Nt = (int)(t_max/dt); // new number of samples
	int Nw = 1024; // Hann window size for fractional delay
	//std::cout << "Nw = " << Nw << "\n";
	
	// compute IRs
	// for each band
	#pragma omp parallel for
	for (int b = 0; b < 8; b++){
		if (f_cutoff > octave_bands[b]) continue; // skip band
		
		// allocate memory for ISM IRs: for each rec, first index sample is time, second is receiver's index
		// (flattened)
		std::vector<float> p_ism(Nt * (size_t)Q.Nr);
		// for each receiver
		for (int r = 0; r < Q.Nr; r++){
			int t_cross = 0.0; // crossover time index, for late tail reverberation (private to thread)
			std::vector<int> arrivals; // sample index of every deposited arrival, for the crossover
			float src_scale2 = 0.0f; // summed squared val_s: tail level scale (powers add)
			// sum contributions of sources (real and virtual)
			for (int s = 0; s < Q.Ns; s++){
				src_scale2 += Q.src_pressure[s] * Q.src_pressure[s];

				// direct: guard the deposit only - do NOT skip the image loop
    			float d_0 = Q.direct_at(s, r); // 0.0f if blocked
    			if (d_0 > 0.0f){
        			float tau_0 = fs * d_0 / c;
        			int t_0 = (int)tau_0;
        			if (t_0 < Nt){
        				float gS_0 = Q.src_pressure[s] / (4.0*M_PI*d_0);
        				// no fractional delay
            			//p_ism[flat(t_0, r, Nt)] += gS_0;
            			
            			// fractional delay
            			for (int n0 = -(int)(0.5*Nw); n0 <= (int)(0.5*Nw); n0++){
							int nn = t_0 + n0;
							if (nn < 0 || nn >= Nt) continue;
							float xx = nn - tau_0;
							p_ism[flat(nn,r,Nt)] += gS_0 * 0.5 * (1.0 + std::cos(2.0*M_PI*xx/Nw))*sinc(M_PI*xx);
						}
            			
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
					//std::cout << "t = " << t << "\n";
					if (t >= Nt) continue; // safety guard
					if (t > t_cross) t_cross = t;
					arrivals.push_back(t);
					float P = 1.0; // iterated product
					// for each surface k crossed by the image source
					for(int k = 0; k < iS.angle_rad.size(); k++){
						float theta = iS.angle_rad[k];
						//float theta = 0.0; // (test for normal incidence, shoebox comparison)
						float R = R_oblique(iS.alpha[k][b], theta);
						P *= R;
					}
					// .amp is a factor rangining [0,1] that handles curved surfaces (part of the
					// parser's heuristics)
					
					float gS = Q.src_pressure[s] * iS.amp * P / (4.0*M_PI*d);
					// no fractionl delay
					//p_ism[flat(t,r,Nt)] += gS;
					
					// fractional delay, ref. Reuben Thomas (Thesis), Fu & Li (2016)
					for (int n0 = -(int)(0.5*Nw); n0 <= (int)(0.5*Nw); n0++){
						int nn = t + n0;
						if (nn < 0 || nn >= Nt) continue;
						float xx = nn - tau;
						p_ism[flat(nn,r,Nt)] += gS * 0.5 * (1.0 + std::cos(2.0*M_PI*xx/Nw)) * sinc(M_PI*xx);
					}
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
				if (theo - (i + 1) > 2.0f * std::sqrt(theo)){ t_cross = arrivals[i]; break; }
			}

			// statistical tail, level-matched to the ISM's val_s/(4*pi*d) convention
			stat_tail(p_ism, Nt, t_cross, r, fs, c, V, T60[b], 1234u + 8u*r + b, std::sqrt(src_scale2));
			
			// this receiver is done for this band -> write its slice to associated binary file
        	std::string fname = out_file("RIR" + std::to_string(r) + "_"
                                    		+ std::to_string((int)octave_bands[b]) + ".bin");
        	FILE* f = std::fopen(fname.c_str(), "wb");
        	if (f){
           		std::fwrite(p_ism.data() + (size_t)r*Nt, sizeof(float), (size_t)Nt, f);
           		std::fclose(f);
        	} else {
           		std::fprintf(stderr, "\n[ISM] could not open %s for writing\n", fname.c_str());
        	}
		}	
	}
	// Compute late reverberation tail (statisticaly), ref. Svensson (1998)
	// Comment when comparing with analytic ISM for showbox-shaped rooms per frequency, with 6 distinct faces
	
    
    // ---------------------------------------------------------------------------------------------------------
    
    printf("\n[OK] Finished.\n\n"); 
    
    return 0;
}
