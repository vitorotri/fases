#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <sstream>
#include "functions3D.cuh"

int main(int argc, char *argv[]){
	if (argc < 5) {
        printf("Usage: %s <scene> <config> <src> <rec>\n", argv[0]);
        return 1;
    }

	printf("\n[!] Fit all materials first!\n");

	printf("\n[!] Call Python script showing scene config here?\n");
	
	printf("\n[!] Is poles.txt consistent?\n");

	float t_max, dt, h, f_cutoff; // max time, time-step, grid spacing, max frequency
	float *p0, *u0, *v0, *w0, *p1, *u1, *v1, *w1, *p2, *u2, *v2, *w2; // host field arrays
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
    std::ifstream file(argv[2]);
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
    
    n = 0; // initial time-step
    
    // ---------------------- load OBJ/PLY/XYZ/etc file and create scene ---------------------
    
    system("rm output/pressureX.bin output/pressureY.bin output/pressureZ.bin output/oid.bin output/S.bin output/header.txt");
        
    char *path = argv[1];
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror("Error opening file");
        return -1;
    }
    int nObj = 0; // number of obejcts in the scene (initialization)
    //int file_opt = 0; // o for OBJ, 1 for VOX/PLY (initialization)
    float *S = NULL; // signed distance function array or voxels, size Nx*Ny, flattened
    float *d_S = NULL;
    float tol = 3.0f*h; // tolerance

	/*
    float *R = NULL; // surfaces reflection coefficient
    float *d_R = NULL;
    float *T = NULL; // barrier transmission coefficient
    float *d_T = NULL;
    float *th = NULL; // barrier thickness
    float *d_th = NULL;
    */
    uint8_t *oid = NULL; // for surface attribution
    uint8_t* d_oid = NULL;
    
    // ------------------------------- get Device(s) information -----------------------------------
    
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0){
    	printf("\n[!] FAILED: no CUDA devices found\n\n");
    	return -1;
    }
    for (int i = 0; i < nDevices; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\n Device Number: " << i << std::endl;
        std::cout << " Device Name: " << prop.name << std::endl;
        std::cout << " Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << " Total Global Memory (bytes): " << prop.totalGlobalMem << std::endl;
        std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << " Memory Clock Rate (kHz): " << prop.memoryClockRate << std::endl;
        std::cout << " Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        std::cout << " L2 Cache Size (bytes): " << prop.l2CacheSize << std::endl;
        std::cout << " Shared Memory per Block (bytes): " << prop.sharedMemPerBlock << std::endl;
		std::cout << " Shared Memory per SM (bytes): "    << prop.sharedMemPerMultiprocessor << std::endl;
    }
    
    // -------------------------------------- process OBJ scene ------------------------------------
    
    std::vector<std::string> names; // global naming
    
   	if (ends_with_dotobj(path)){
   	
   		printf("\n[OK] processing scene...\n");
   		
    	Scene3D sc;

		if(!parse_obj3d(path, &sc, h)) return 1;
		
		build_sdf(&sc, h, &Nx, &Ny, &Nz, &Lx, &Ly, &Lz, &S);
		
		if(!build_oid_dense_cpu(sc, S, Nx, Ny, Nz, h, tol, &oid)){
    		fprintf(stderr,"Failed to build OID map\n");
    		return 1;
		}
		
		FILE *file_S = fopen("output/S.bin", "wb");
		if (file_S) {
    		fwrite(S, sizeof(float), (size_t)Nx*Ny*Nz, file_S);
    		fclose(file_S);
		}
		
		FILE *file_oid = fopen("output/oid.bin", "wb");
		if (file_oid){
    		fwrite(oid, sizeof(uint8_t), (size_t)Nx*Ny*Nz, file_oid);
    		fclose(file_oid);
		}
		
		cudaMalloc(&d_oid, (size_t)Nx*Ny*Nz*sizeof(uint8_t));
		cudaMemcpy(d_oid, oid, (size_t)Nx*Ny*Nz*sizeof(uint8_t), cudaMemcpyHostToDevice);
		
		nObj = scene_layer_count_3d(sc);
		scene_layer_names_3d(sc, names);
		
		cudaMalloc(&d_S, (size_t)Nx*Ny*Nz*sizeof(float));
		cudaMemcpy(d_S, S, (size_t)Nx*Ny*Nz*sizeof(float), cudaMemcpyHostToDevice);
		
		scene_free(&sc);
    }
    // process PLY/VOX test
    else {
    	return -1;
    }
    
    FILE *pre_header; // file for data output (sdf_plot.py TST)
	pre_header = fopen("output/pre_header.txt", "w");
	if (pre_header != NULL){
    	fprintf(pre_header, "%d %d %d %f %d\n", Nx, Ny, Nz, h, (int)(tol/h));
    	fclose(pre_header);
    }
    
    
    /*
    // force for test with BCs without parsing (no need to comment above)
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
    
    int nMat = nObj - 1; // exclude boundary
    
    Rinf = new float[nMat]();
    
    // read K_m and L_m from file
    
    int K_m, L_m; // k_max and L_max from all the fits
    
    // read poles.txt file
    file.open("config/poles.txt");
    if (!file.is_open()) {
        std::cerr << "[!] Cannot open config/poles.txt" << std::endl;
        return EXIT_FAILURE;
    }
    file >> K_m >> L_m;
	file.close();
	file.clear();
	
	// minimum reasonable size safety
	if (K_m < 1) K_m = 1;
	if (L_m < 1) L_m = 1;
    
	Ak 		= new float[K_m * nMat]();
	zeta 	= new float[K_m * nMat]();
	
	Bl 		= new float[L_m * nMat]();
    Cl 		= new float[L_m * nMat]();
    alpha 	= new float[L_m * nMat]();
    beta 	= new float[L_m * nMat](); 
    
    printf("\nChecking materials\n\n");
    for (size_t i = 0; i < names.size(); ++i){
  		std::cout << i << " -> " << names[i] << "\n";
  	}

    
    for (int m = 0; m < nMat; ++m) {
    	std::string filename = "input/fit/" + names[m+1] + ".fit"; // names[0] = Boundary
    	std::cout << "\nReading: " << filename << "\n";	
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


    // ------------------------------------- atmospheric values ------------------------------------------------
	
	// should to to config later
	
	float R_spec = 287.05f; // specific gas constant for air [J] [kg]^-1 [K]^-1
	float gamma = 1.4f; // specific heat ratio for air
	
	rho = 1.225f; // average sea level density for air [kg/m3]. For dry air, proportional to p0/R_spec*T
	Temp = conf_arr[3] + 273.15f; // temperature [K] = Celsius_temp + 273.15
	c = sqrt(gamma*R_spec*Temp); // speed of sound [m/s]
	// c = sqrt(K/rho) in case of liquids
    
    printf("\n	c = %.2f m/s\n",c);
    
    // ---------------------------------------------------------------------------------------------------------

    // Allocate remaining host memory and initilize to 0.0f
    p0 = new float[N](); // pressure deviations from p0
    u0 = new float[N](); // x component of particle velocity 
    v0 = new float[N](); // y component of particle velocity 
    w0 = new float[N](); // z component of particle velocity

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
	if (lambda / h < 12.0f){
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
	
	int kern;
	float dm;
	printf("\nChoose an option for time integration:\n\n 1) Godunov + Euler\n 2) WENO3 + SSP-RK3\n\n");
	scanf("%d", &kern);
	
	float eps = 1e-7; // the smaller the eps, the more precise the BCs work, but cannot be the smallest otherwise have inf. In C, floats can go up to 1e-38. With 1e-5 workd really bad, with 1e-10 works well, depite below machine precision for float, so results may be wrong even with convergence!!!
	
	p1 = new float[N](); // sound pressure
    u1 = new float[N](); // x component of particle velocity 
    v1 = new float[N](); // y component of particle velocity
    w1 = new float[N](); // z component of particle velocity
	
	cudaMalloc(&d_p1, N*sizeof(float));
	cudaMalloc(&d_u1, N*sizeof(float));
	cudaMalloc(&d_v1, N*sizeof(float));
	cudaMalloc(&d_w1, N*sizeof(float));
		
	cudaMemset(d_p1, 0.0, N * sizeof(float));
	cudaMemset(d_u1, 0.0, N * sizeof(float));
	cudaMemset(d_v1, 0.0, N * sizeof(float));
	cudaMemset(d_w1, 0.0, N * sizeof(float));
	
	dm = 8*N*sizeof(float);
	
	// WENO5 option
	
	if (kern == 2){
		
    	p2 = new float[N](); // sound pressure
    	u2 = new float[N](); // x component of particle velocity 
    	v2 = new float[N](); // y component of particle velocity
		w2 = new float[N](); // z component of particle velocity
		
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
	}
	
	// ----------------------- Parse sources and receivers from argv[3] and argv[4] --------------------
    
    // parse source(s)
    
    //float *p_time = new float[Nt]();
    //float *d_p_time;
    int i_src, j_src, k_src; // discretized position of source
    
    //float src_y; // for raised cosine
    
    file.open(argv[3]);
	if (!file.is_open()){
    	std::cerr << "[!] Failed to open: " << argv[3] << "\n";
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
    //i_rec = Nx/2; j_rec = Ny/2; k_rec = Nz/2; // safe default
    
    std::vector<int> i_recV, j_recV, k_recV; // vectors for multiple receivers
    
    file.open(argv[4]);
	if (!file.is_open()){
    	std::cerr << "[!] Failed to open: " << argv[4] << "\n";
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
        	}
    	}
	}
	file.close();
	file.clear();
	
	
	// ----------------------- Define kernel parameters and copy sources to device --------------------------
	
	int opt; // opt param for some user inputs
	
	
	
	// peridoc BCs trick (if used)
	dim3 TPz(16,16);
	dim3 BPz((Nx + TPz.x - 1)/TPz.x, (Ny + TPz.y - 1)/TPz.y);
	kernel_periodic_trick_z<<<BPz, TPz>>>(d_S, d_oid, Nx, Ny, Nz, h);
	cudaDeviceSynchronize();
	dim3 TPx(16,16);
	dim3 BPx((Ny + TPx.x - 1)/TPx.x, (Nz + TPx.y - 1)/TPx.y);
	kernel_periodic_trick_x<<<BPx, TPx>>>(d_S, d_oid, Nx, Ny, Nz, h);
	cudaDeviceSynchronize();
	cudaMemcpy(S, d_S, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(oid, d_oid, N*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	FILE *file_S = fopen("output/S.bin", "wb");
	if (file_S) {
    	fwrite(S, sizeof(float), (size_t)Nx*Ny*Nz, file_S);
    	fclose(file_S);
	}
	FILE *file_oid = fopen("output/oid.bin", "wb");
	if (file_oid){
    	fwrite(oid, sizeof(uint8_t), (size_t)Nx*Ny*Nz, file_oid);
    	fclose(file_oid);
	}
	
	
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
	printf("\n[OK] Memory to animate is %f GB\nProceed?\n\n 1) Yes\n 2) No\n\n", anim_s/(1000000000));
	scanf("%d", &opt);
	if (opt != 1) return -1;
	
	int lim_ID = 0; // limiter_ID
	if (kern != 2){
		printf("\nType a limiter option then press ENTER:\n\n 0) no slope\n 1) central\n 2) minmod\n 3) superbee\n 4) MC\n 5) van Leer\n 6) van Albada\n\n");
		scanf("%d", &lim_ID);
	}
	
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
	
	// compute initial TVs if not using p_time[]
	
	TV_p_ = TV(p0, Nx, Ny, Nz, h);
	TV_u_ = TV(u0, Nx, Ny, Nz, h);
	TV_v_ = TV(v0, Nx, Ny, Nz, h);
	TV_w_ = TV(w0, Nx, Ny, Nz, h);
	
	TV_0 = TV_p_ + TV_u_ + TV_v_ + TV_w_;
	
	//TV_0 = 3.0 * h * h * p_time[0];
	
	// --------------------------------- Initial energy ------------------------------------
	
	// if pulse is only p-based sinusoid (max is 1.0)
	//float E0 = 0.5 * p_time[0] * p_time[0] / (rho * c*c); // initial energy for delta pulse
	
	// -------------------------------------------------------------------------------------
	
	printf("\n[OK] Will now start %d iterations. Proceed?\n\n1) Yes\n2) No\n\n", Nt);
	scanf("%d", &opt);
	if (opt != 1) return -1;
	
	// open BIN file for writing pressure fields
	FILE *file_pX = fopen("output/pressureX.bin", "ab");
	if (file_pX == NULL) {
    	printf("[!] Error opening output/pressureX.bin file!\n");
    	exit(1);
	}
	FILE *file_pY = fopen("output/pressureY.bin", "ab");
	if (file_pY == NULL) {
    	printf("[!] Error opening output/pressureY.bin file!\n");
    	exit(1);
	}
	FILE *file_pZ = fopen("output/pressureZ.bin", "ab");
	if (file_pZ == NULL) {
    	printf("[!] Error opening output/pressureZ.bin file!\n");
    	exit(1);
	}
	
	/*
	// open BIN file for writing pressure at single receiver
	FILE *f_rec = fopen("output/IR.bin", "ab");
	if (f_rec == NULL) {
    	printf("[!] Error opening output/IR.bin file!\n");
    	exit(1);
	}
	*/
	
	// open BIN files for writing pressure at multuple receivers
	std::vector<std::ofstream> files_rec(i_recV.size());
	// i_recV.size() should be the same as j_recV.size() and k_recV.size()
	for (int ct = 0; ct < i_recV.size(); ct++) {
    	std::string str = "output/IR" + std::to_string(ct) + ".bin";
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

	// Godunov + Euler
	if (kern == 1){
		/*
		while(n < Nt){
		
		// Launch Euler kernel
			
		//kernel_Euler<<<BP, TP>>>(Nx, Ny, Nz, d_p0, d_u0, d_v0, d_w0, d_p1, d_u1, d_v1, d_w1, rho, c, h, dt, 2*lim_ID, 2*lim_ID + 1, d_S, U_PML, D_PML, R_PML, L_PML, F_PML, B_PML);
			
		//kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u1, d_v1, d_p1, Nx, Ny, h, d_S, d_R, d_oid, rho, c, n, dt, d_u0, d_v0, d_p0, tol, d_T, d_th, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, phi, psi_1, psi_2);
			
		// exchange arrays
		cudaMemcpy(d_p0, d_p1, N*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_u0, d_u1, N*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_v0, d_v1, N*sizeof(float), cudaMemcpyDeviceToDevice);
			
		cudaDeviceSynchronize();
			
		print_progress(n, Nt, 50);
	
		// Copy result back to host
		cudaMemcpy(p1, d_p1, N*sizeof(float), cudaMemcpyDeviceToHost);
			
		// write to file
		if (n % anim_step == 0){
			fwrite(p1, sizeof(float), N, file_pX);
        	fflush(file_pX);  // Ensure data is written to disk
        	
        	pbin_size += 1;
        	
        }
		
		n += 1; // update time-step
	}
	
	// Copy final results back to host (for computing TVs)
	
	cudaMemcpy(p1, d_p1, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(u1, d_u1, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(v1, d_v1, N*sizeof(float), cudaMemcpyDeviceToHost);
		
	TV_p_ = TV(p1, Nx, Ny, Nz, h);
	TV_u_ = TV(u1, Nx, Ny, Nz, h);
	TV_v_ = TV(v1, Nx, Ny, Nz, h);
	TV_w_ = TV(w1, Nx, Ny, Nz, h);
		
	TV_1 = TV_p_ + TV_u_ + TV_v_ + TV_w_;
	printf("\n\nTV_0 = %f\n", TV_0);
	printf("\nTV_1 = %f\n", TV_1);
		
	is_TVD(TV_0, TV_1);

    // Free memory
    delete[] p0, delete[] u0, delete[] v0;
    delete[] p1, delete[] u1, delete[] v1;
    cudaFree(d_p1); cudaFree(d_u1); cudaFree(d_v1); cudaFree(d_w1);
    cudaFree(d_p2); cudaFree(d_u2); cudaFree(d_v2); cudaFree(d_w2);
    */
    }
    // WENO5 + SSP-RK3
    else{
		while(n < Nt){
			/*
			// counter pressure for directive source
			// be mindfull as this can work like a local obstacle, so reflections may arise
			if (n < 10){
				p0[flat(src_x-3, src_y-3, Nx)] = 0.0;
				p0[flat(src_x-2, src_y-2, Nx)] = 0.0;
				p0[flat(src_x-1, src_y-1, Nx)] = 0.0; 
					
				p0[flat(src_x-1, src_y, Nx)] = 0.0;
				p0[flat(src_x, src_y-1, Nx)] = 0.0;
				p0[flat(src_x-2, src_y, Nx)] = 0.0;
				p0[flat(src_x, src_y-2, Nx)] = 0.0;
				p0[flat(src_x-3, src_y, Nx)] = 0.0;
				p0[flat(src_x, src_y-3, Nx)] = 0.0;
					
				cudaMemcpy(d_p, p, N*sizeof(float), cudaMemcpyHostToDevice);
			}
			*/
				
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
			
			print_progress(n, Nt, 50);
		
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
			
			// write to file for each receiver
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
		
		TV_p_ = TV(p0, Nx, Ny, Nz, h);
		TV_u_ = TV(u0, Nx, Ny, Nz, h);
		TV_v_ = TV(v0, Nx, Ny, Nz, h);
		TV_w_ = TV(w0, Nx, Ny, Nz, h);
		
		TV_1 = TV_p_ + TV_u_ + TV_v_ + TV_w_;
		printf("\n\nTV_0 = %f\n", TV_0);
		printf("\nTV_1 = %f\n", TV_1);
		
		is_TVD(TV_0, TV_1);
		
		/*
		float E1 = Energy(p0, u0, v0, w0, Nx, Ny, Nz, h, rho, c);
		
		printf("\nE0 = %.12f\n", E0);
		printf("\nE1 = %.12f\n", E1);
		is_conservative(E0, E1);
		*/
    	// Free host memory
    	delete[] p0; delete[] u0; delete[] v0; delete[] w0;
    	delete[] p1; delete[] u1; delete[] v1; delete[] w1;
    	delete[] p2; delete[] u2; delete[] v2; delete[] w2;
    	
    	// Free device memory
    	cudaFree(d_p0); cudaFree(d_u0); cudaFree(d_v0); cudaFree(d_w0);
    	cudaFree(d_p1); cudaFree(d_u1); cudaFree(d_v1); cudaFree(d_w1);
    	cudaFree(d_p2); cudaFree(d_u2); cudaFree(d_v2); cudaFree(d_w2);
    	cudaFree(c_rl); cudaFree(dd);
    	
    }
    
    fflush(file_pX); fflush(file_pY); fflush(file_pZ);
	//fflush(f_rec);
    
    toc = clock();
    
    elapsed_time = ((double)(toc - tic)/60.0)/CLOCKS_PER_SEC; // in minutes
    printf("\n[OK] Simulation time: %f min \n\n", elapsed_time); 
    
    fclose(file_pX); fclose(file_pY); fclose(file_pZ);
    //fclose(f_rec);
    files_rec.clear();
    fclose(fp);
    
    printf("[OK] Wrote to file for single receiver\n\n");
    
    FILE *header; // file for data output
	header = fopen("output/header.txt", "w");
	if (header != NULL){
    	fprintf(header, "%d %d %d %d %f %f %f %d %d %f %s %f %d\n", Nx, Ny, Nz, Nt, Lx, Ly, Lz, pbin_size, anim_step, dt, argv[1], h, (int)(tol/h));
    	fclose(header);
    }
    
    printf("\n[OK] Printed header.txt for plot.py\n");
    
    // ----- free arrays -----
    
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
    
    // -----------------------
    
    printf("\n[OK] Finished.\n\n"); 
    
    return 0;
}
