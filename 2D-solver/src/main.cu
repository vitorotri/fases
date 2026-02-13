#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>
#include <sstream>
#include "functions.cuh"
//#include <cuda_runtime.h>

int main(int argc, char *argv[]){
	if (argc < 4) {
        printf("Usage: %s <scene> <config> <src> <rec>\n", argv[0]);
        return 1;
    }

	printf("\n[!] Fit all materials first!\n");

	printf("\n[!] Call Python script showing scene config here?\n");

	float t_max, dt, h, f_cutoff; // max time, time-step, grid spacing, max frequency
	float *p0, *u0, *v0, *p1, *u1, *v1, *p2, *u2, *v2; // host arrays
	float *d_p0, *d_u0, *d_v0, *d_p1, *d_u1, *d_v1, *d_p2, *d_u2, *d_v2, *c_rl, *dd; // device arrays
	float rho, c, Temp; // density, speed of sound, temperature
	float Lx, Ly; // domain dimensions
	float N_PML, S_PML, E_PML, W_PML; // Perfectly Matched Layer activation keys
    int Nx, Ny, n; // number of points in x, y, time step
    
    // timing
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
    	float f;
    	std::istringstream(line) >> f;
    	conf_arr.push_back(f);
	}
	file.close();
	file.clear();
	
    // domain variables
    h = conf_arr[0]; // uniform grid spacing [m]
    t_max = conf_arr[1]; // maximum simulation time [s]
    f_cutoff = conf_arr[2]; // [Hz] (cutoff frequency)
    
    // PML keys
    N_PML = (float)conf_arr[6];
    S_PML = (float)conf_arr[7];
    E_PML = (float)conf_arr[8];
    W_PML = (float)conf_arr[9];
    
    n = 0; // initial time-step
    
    // ---------------------- load OBJ/PLY/XYZ/etc file and create scene ---------------------
    
    system("rm output/pressure.bin output/oid.bin output/S.bin output/header.txt");
    
    
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
    
    // -------------------------------------- process OBJ scene ------------------------------------
    
    std::vector<std::string> names; // global naming
    
   	if (ends_with_dotobj(path)){
    	//const char *path = "TST2.obj";  // your 2D OBJ like in the prompt
    	Scene2D sc;
    	if(!parse_obj2d(path, &sc)) return 1;
    	build_sdf(&sc, h, &Nx, &Ny, &Lx, &Ly, &S);

		
		FILE *file_S = fopen("output/S.bin", "wb");  // Open file for writing in binary mode
    	if (file_S) {
        	fwrite(S, sizeof(float), Nx*Ny, file_S); // Write array to file
        	fclose(file_S);
    	}
		/*
		// sanity test for S
		for (int i = 0; i < Nx*Ny; i++){
			if (isnan(S[i])) {
            	printf("[!] WARNING: S[%d] is NaN\n", i);
            }
		}
		*/
		if(!build_oid_dense_cpu(sc, S, Nx, Ny, h, tol, &oid)){
    		fprintf(stderr,"Failed to build OID map\n");
    		return 1;
		}
		/*
		// sanity test for oid
		for (int i = 0; i < Nx*Ny; i++){
            printf("oid[%d] = %u\n", i, oid[i]);
		}
		*/
		// oid[] have values ranging from 1...nObj, with 255 in the other cells
		FILE *file_oid = fopen("output/oid.bin", "wb");  // Open file for writing in binary mode
    	if (file_oid) {
        	fwrite(oid, sizeof(uint8_t), Nx*Ny, file_oid); // Write array to file
        	fclose(file_oid);
    	}
		
		// copy oid[] to device
		cudaMalloc(&d_oid, (size_t)Nx*Ny*sizeof(uint8_t));
		cudaMemcpy(d_oid, oid, (size_t)Nx*Ny, cudaMemcpyHostToDevice);
    	
		// get nObj from scene
    	nObj = scene_layer_count(sc); // number of layers/materials including Boundary
    	//printf("\n\nnObj = %d\n\n", nObj);
		scene_layer_names(sc, names);
		std::sort(names.begin() + 1, names.end()); // ordering
		
		//allocate single R, T, thickness arrays
		/*
		// allocate array of reflection coefficients per material/obj
		R = new float[nObj-1]; // exclude boundary
		
		// allocate array of transmission coefficients per material/obj
		T = new float[nObj-1]; // exclude boundary
		
		// allocate array of thickness per barrier
		th = new float[nObj-1]; // exclude boundary
    	*/
    	printf("\n[OK] Scene processing\n");
    	printf("\nDomain info:\n\nLx = %.6f m Ly = %.6f m\nNx = %d Ny = %d\n", Lx, Ly, Nx, Ny);
    	
    	/*
    	int counter_mat = 0; // 0 is boundary in "names"
    	float R_val, T_val, th_val;
    	int obj_typ;
    	while(counter_mat < nObj-1){
    		printf("\nAttribute object's type for %s\n", names[counter_mat+1].c_str());
    		printf("\n(1) Regular obstacle\n(2) Barrier\n(3) Medium interface\n\n");
    		scanf("%d", &obj_typ);
    		switch(obj_typ){
    			case 1:
    				printf("\nEnter with the R value for the obstacle %s\n\n", names[counter_mat+1].c_str());
    				scanf("%f", &R_val);
    				R[counter_mat] = R_val;
    				T[counter_mat] = 0.0f;
    				th[counter_mat] = 0.0f;
    				break;
    			case 2:
    				printf("\nEnter with the R value for object %s\n\n", names[counter_mat+1].c_str());
    				scanf("%f", &R_val);
    				R[counter_mat] = R_val;
    				printf("\nEnter with the T value for the object %s\n\n", names[counter_mat+1].c_str());
    				scanf("%f", &T_val);
    				T[counter_mat] = T_val;
    				check_energy_coeffs(R_val,T_val); // warning
    				printf("\nEnter the thickness value for the object %s\n\n", names[counter_mat+1].c_str());
    				scanf("%f", &th_val);
    				th[counter_mat] = th_val;
    				break;
    			case 3:
    				//
    				break;
    		}
    		counter_mat += 1;
    	}
    	*/
     	// allocate on device and then copy, signed distance field, Reflection, Transmission, barrier thickness
    	cudaMalloc(&d_S, (size_t)Nx*Ny*sizeof(float));
    	cudaMemcpy(d_S, S, (size_t)Nx*Ny*sizeof(float), cudaMemcpyHostToDevice);
    	/*
		cudaMalloc(&d_R, (nObj-1)*sizeof(float));
		cudaMemcpy(d_R, R, (nObj-1)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_T, (nObj-1)*sizeof(float));
		cudaMemcpy(d_T, T, (nObj-1)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc(&d_th, (nObj-1)*sizeof(float));
		cudaMemcpy(d_th, th, (nObj-1)*sizeof(float), cudaMemcpyHostToDevice);
		*/
    	scene_free(&sc);
    }
    // process PLY/VOX test
    else {
    	return -1;
    }
    
    FILE *pre_header; // file for data output (sdf_plot.py TST)
	pre_header = fopen("output/pre_header.txt", "w");
	if (pre_header != NULL){
    	fprintf(pre_header, "%d %d %f %d\n", Nx, Ny, h, (int)(tol/h));
    	fclose(pre_header);
    }
    
    
    // --------------------------------------------------------------------------------------------------
    
    int N = Nx*Ny; // number of grid cells
    
    // ----------------------- LOAD filter coefficiets for materials ------------------------------------
    
    // ----- host arrays for materials -----
    
    float *Rinf;
    float *Ak, *zeta;
    float *Bl, *Cl, *alpha, *beta;
    
    int nMat = nObj - 1; // exclude boundary
    
    Rinf = new float[nMat]();
    
    // read K_m and L_m from file
    
    int K_m, L_m; // k_max and L_max from all the fits
    
    //K_m = int(conf_arr[5]);
    //L_m = int(conf_arr[6]);
    
    // read poles.txt file
    file.open("config/poles.txt");
    if (!file.is_open()) {
        std::cerr << "[!] Cannot open config/file.txt" << std::endl;
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
    
    for (size_t i = 0; i < names.size(); ++i){
  		std::cout << i << " -> " << names[i] << "\n";
  	}

    
    for (int m = 0; m < nMat; ++m) {
    	std::string filename = "input/fit/" + names[m+1] + ".fit"; // names[0] = Boundary
    	std::cout << "Reading: " << filename << "\n";
	
    	read_fit_file(filename, m, K_m, L_m, Rinf, Ak, zeta, Bl, Cl, alpha, beta);
	
		// simple print test
    	std::printf("Rinf[%d] = %.15f\n", m, (float)Rinf[m]);
    	std::printf("Ak(0,%d)  = %.15f   zeta(0,%d) = %.15f\n",
                	m, (float)Ak[flat(0, m, K_m)],
                	m, (float)zeta[flat(0, m, K_m)]);
    	std::printf("Ak(1,%d)  = %.15f   zeta(1,%d) = %.15f\n\n",
                	m, (float)Ak[flat(1, m, K_m)],
                	m, (float)zeta[flat(1, m, K_m)]);
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
	
	// ----- boundary map for phi and psi -----
	
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
    
    printf("\nc = %f m/s\n",c);
    
    // ---------------------------------------------------------------------------------------------------------

    // Allocate remaining host memory and initilize to 0.0f
    p0 = new float[N](); // pressure deviations from p0
    u0 = new float[N](); // x component of particle velocity 
    v0 = new float[N](); // y component of particle velocity 

	float C_CFL = conf_arr[4]; // Courant number: pick values < 0.5 (not equal to)
	dt = C_CFL*h/c; // Courant-Friedrichs-Lewy condition
	printf("\ndt = %.9f\n", dt);
	
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
	printf("\nlambda = %f\n",lambda);
	if (lambda / h < 12.0f){
		printf("[!] Resolution too low, aborting...\n\n");
		return -1;
	}
	
	int Nt = (int) (t_max/dt) + 1; // number of iterations
	
	printf("\nNt = %d\n\n",Nt);

    // Allocate device memory
    if (cudaMalloc(&d_p0, N*sizeof(float)) != cudaSuccess
		|| cudaMalloc(&d_u0, N*sizeof(float)) != cudaSuccess
		|| cudaMalloc(&d_v0, N*sizeof(float)) != cudaSuccess){
    	printf("[!] CUDA error allocating device memory\n");
    	return -1;
	}
	
	
	// ----------------------- Parse sources and receivers from argv[3] and argv[4] --------------------
    
    // parse sources
    
    file.open(argv[3]);
	if (!file.is_open()){
    	std::cerr << "[!] Failed to open: " << argv[3] << "\n";
    	return -1;
	}
	while (std::getline(file, line)) {
    	if (line.empty() || line[0] == '#') continue;
    	float val_x, val_y, val_s;
    	std::istringstream iss(line);
    	if (iss >> val_x >> val_y >> val_s) {
        	int i_src = int(val_x / h);
        	int j_src = int(val_y / h);
        	if (i_src >= 0 && i_src < Nx && j_src >= 0 && j_src < Ny) {
            	p0[flat(i_src, j_src, Nx)] = val_s;
        	}
    	}
	}
	file.close();
	file.clear();
    
    // parse single receiver (for now), use array later
    
    int rec_x = Nx/2, rec_y = Ny/2; // safe default
    file.open(argv[4]);
	if (!file.is_open()){
    	std::cerr << "[!] Failed to open: " << argv[3] << "\n";
    	return -1;
	}
	while (std::getline(file, line)) {
    	if (line.empty() || line[0] == '#') continue;
    	float val_x, val_y;
    	std::istringstream iss(line);
    	if (iss >> val_x >> val_y) {
        	int i_rec = int(val_x / h);
        	int j_rec = int(val_y / h);
        	if (i_rec >= 0 && i_rec < Nx && j_rec >= 0 && j_rec < Ny) {
            	rec_x = i_rec;
            	rec_y = j_rec;
        	}
    	}
	}
	file.close();
	file.clear();
	
	
	// --------------------------- Fields initialization (sources and receiver) ----------------------
	
	
	int opt;
	/*
	printf("\nChoose pulse type:\n\n 1) Delta\n 2) Gaussian\n 3) Top-hat\n\n");
	scanf("%d",&opt);
	
	// for Lx = 4.07, Ly = 12.03 [m]
	int src_x = (int)(0.5f*Nx); int src_y = (int)(0.5f*Ny); // single source ut 0.5*Nx,0.2*Ny for numerical TST reflect
	int rec_x = (int)(0.75f*Nx); int rec_y = (int)(0.5f*Ny); // sound receiver at normal incidence (transmission)
	
	float amp = 0.001f; // amplitude
	if (opt == 1){
		// Delta pulse
		//p0[flat(src_x, src_y, Nx)] = amp; // [Pa]
		//p0[flat((int)(0.75f*Nx), src_y, Nx)] = amp; // 2nd source
		
		// Directional source
		//p0[flat(src_x+1, src_y+1, Nx)] = amp; // without pressure scheme seems not TVD
		//u0[flat(src_x+1, src_y, Nx)] = 0.04;
		//v0[flat(src_x, src_y+1, Nx)] = 0.04;
		
		// Delta distribution
		//delta(p0, 1.0/2.0, src_x, src_y, Nx, Ny, 390.0);
		
		
		// Plane wave
		//int src_x_0 = (int)(0.25f*Nx);
		//int src_x_1 = (int)(0.08*Nx);
		
		//for (int j = (int)(0.05*Ny); j < Ny-(int)(0.05*Ny); j++){ // depends on resolution
		for (int j = int(0.05f*Ny); j < int(0.95f*Ny); j++){
			//for (int i = src_x_0; i <= src_x_1; i++){
				u0[flat(src_x,j,Nx)] = amp;
				p0[flat(src_x,j,Nx)] = 0.1*amp;
			//}
		}
		
		
	}
	else if (opt == 2){
		// Gaussian profile
		//amp = 1.0; // 125
		
 		float sigma = 1.5; // 2
		float r2;
		for (int i = 0; i < Nx; i++){
			for (int j = 0; j < Ny; j++){
            	r2 = (i - src_x)*(i - src_x) + (j - src_y)*(j - src_y);
				p0[flat(i,j,Nx)] = amp*exp(-r2 / (2*sigma*sigma));
			}
		}
		
	}
	else {
		// Top-hat
		int ratio = 200;
		int ii, jj;
		for (int i = 0; i < Nx; i++){
			for (int j = 0; j < Ny; j++){
				ii = i - src_x;
				jj = j - src_y;
				if (sqrt(ii*ii + jj*jj) <= max(Nx,Ny)/ratio){
					p0[flat(i,j,Nx)] = 150.0;
				}
			}
		}
	}
	*/
	
	// ----------------------- Define kernel parameters and copy sources to device --------------------------
	
	// Kernel launch parameters
	dim3 TP(8,8); // multiples of 32, max is 1024 (TP.x * TP.y <= 1024) // may influence on number of resources requested, be carefull (here maximum is probably 16 x 16)
	dim3 BP((Nx + TP.x - 1)/TP.x,(Ny + TP.y - 1)/TP.y); // BLOCKS Per Grid
	
	// Copy initial data to device
    if (cudaMemcpy(d_p0, p0, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_u0, u0, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess
    	|| cudaMemcpy(d_v0, v0, N*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess){
    	printf("[!] CUDA error allocating device memory\n");
    	return -1;
    }
	
	// ---------------------------------- kernel options --------------------------------
	
	int kern;
	float dm;
	printf("\nChoose an option for time integration:\n\n 1) Godunov + Euler\n 2) WENO3 + SSP-RK3\n\n");
	scanf("%d", &kern);
	
	float eps = 1e-7; // the smaller the eps, the more precise the BCs work, but cannot be the smallest otherwise have inf. In C, floats can go up to 1e-38. With 1e-5 workd really bad, with 1e-10 works well, depite below machine precision for float, so results may be wrong even with convergence!!!
	
	p1 = new float[N](); // sound pressure
    u1 = new float[N](); // x component of particle velocity 
    v1 = new float[N](); // y component of particle velocity
	
	cudaMalloc(&d_p1, N*sizeof(float));
	cudaMalloc(&d_u1, N*sizeof(float));
	cudaMalloc(&d_v1, N*sizeof(float));
		
	cudaMemcpy(d_p1, p1, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_u1, u1, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v1, v1, N*sizeof(float), cudaMemcpyHostToDevice);
	
	dm = 6*N*sizeof(float);
	
	if (kern == 2){
		// auxiliar memory arrays (for SSP-RK2)
		
    	p2 = new float[N](); // sound pressure
    	u2 = new float[N](); // x component of particle velocity 
    	v2 = new float[N](); // y component of particle velocity
		
    	cudaMalloc(&d_p2, N*sizeof(float));
		cudaMalloc(&d_u2, N*sizeof(float));
		cudaMalloc(&d_v2, N*sizeof(float));
		
		cudaMemcpy(d_p2, p2, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_u2, u2, N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_v2, v2, N*sizeof(float), cudaMemcpyHostToDevice);
			
		cudaMallocManaged(&c_rl, 6*sizeof(float)); // coefficients
		cudaMallocManaged(&dd, 2*sizeof(float)); // array
			
		c_rl[flat(0,0,3)] = 1.5; c_rl[flat(0,1,3)] = -0.5;
		c_rl[flat(1,0,3)] = 0.5; c_rl[flat(1,1,3)] = 0.5;
		c_rl[flat(2,0,3)] = -0.5; c_rl[flat(2,1,3)] = 1.5;
			
		dd[0] = 2.0/3.0; dd[1] = 1.0/3.0;
		
		cudaDeviceSynchronize();
		
		dm = (6*N + 6 + 2)*sizeof(float);

	}
	
	// ---------------------------------------------------------------------------------------
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("[!] CUDA error: %s\n", cudaGetErrorString(err));
	}
	printf("\n[OK] Memory to allocate in device is %f GB\n", dm/(1000000000));
	
	//printf("float is %zu", sizeof(float));
	int anim_step = (int)(0.008*Nt);
	int frames = Nt/anim_step; // make denominator smaller for more frames
	printf("\n[OK] anim_step = %d\n\n",anim_step);
	float anim_s = (float)frames*N*sizeof(float); // animation size in bytess
	
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
	
	float TV_p_, TV_u_, TV_v_, TV_0, TV_1;
	
	// compute initial TVs
	
	TV_p_ = TV(p0, Nx, Ny, h);
	TV_u_ = TV(u0, Nx, Ny, h);
	TV_v_ = TV(v0, Nx, Ny, h);
	
	TV_0 = TV_p_ + TV_u_ + TV_v_;
	
	printf("\n[OK] Will now start %d iterations. Proceed?\n\n1) Yes\n2) No\n\n", Nt);
	scanf("%d", &opt);
	if (opt != 1) return -1;
	
	// open BIN file for writing pressure field
	FILE *fptr = fopen("output/pressure.bin", "ab");
	if (fptr == NULL) {
    	printf("[!] Error opening pressure.bin file!\n");
    	exit(1);
	}
	
	// open BIN file for writing pressure at single receiver
	FILE *f_rec = fopen("output/IR.bin", "ab");
	if (f_rec == NULL) {
    	printf("[!] Error opening pressure.bin file!\n");
    	exit(1);
	}
	
	//Nt = 100; // for testing
	/*
	// -------- oscillating source (take proper per if kern == 2) --------
	p0[flat(src_x, src_y, Nx)] += 0.0f;
	std::vector<float> p_osc(Nt, 0.0f);
	std::vector<float> tones = {30, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000};
	std::vector<float> phase(tones.size(), 0.0f);
	for (int n = 0; n < Nt; ++n) {
    	float s = 0.0f;
    	for (size_t i = 0; i < tones.size(); i++){
        	phase[i] += 2.0f * M_PI * tones[i] * dt;
        	if (phase[i] > 2.0f * M_PI) {
            	phase[i] -= 2.0f * M_PI;
        	}
        	s += sinf(phase[i]);
    	}
    	p_osc[n] = 0.02 * s;
	}
	*/
	// -------- end oscillating source --------
	
	int pbin_size = 0; // for plotting animation correctly
	
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
		while(n < Nt){
		// Launch Euler kernel
			
		kernel_Euler<<<BP, TP>>>(Nx, Ny, d_p0, d_u0, d_v0, d_p1, d_u1, d_v1, rho, c, h, dt, 2*lim_ID, 2*lim_ID + 1, d_S, N_PML, S_PML, E_PML, W_PML);
			
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
			fwrite(p1, sizeof(float), N, fptr);
        	fflush(fptr);  // Ensure data is written to disk
        	pbin_size += 1;
        }
		
		n += 1; // update time-step
	}
	
	// Copy final results back to host (for computing TVs)
	
	cudaMemcpy(p1, d_p1, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(u1, d_u1, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(v1, d_v1, N*sizeof(float), cudaMemcpyDeviceToHost);
		
	TV_p_ = TV(p1, Nx, Ny, h);
	TV_u_ = TV(u1, Nx, Ny, h);
	TV_v_ = TV(v1, Nx, Ny, h);
		
	TV_1 = TV_p_ + TV_u_ + TV_v_;
	printf("\n\nTV_0 = %f\n", TV_0);
	printf("\nTV_1 = %f\n", TV_1);
		
	is_TVD(TV_0, TV_1);

    // Free memory
    delete[] p0, delete[] u0, delete[] v0;
    delete[] p1, delete[] u1, delete[] v1;
    cudaFree(d_p1); cudaFree(d_u1); cudaFree(d_v1);
    cudaFree(d_p2); cudaFree(d_u2); cudaFree(d_v2);
    	
    }
    // WENO3 + SSP-RK3
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
			/*
			// Copy pressure to device (for oscillating source)
			p0[flat(src_x, src_y, Nx)] += p_osc[n]; // if += then it is a soft source
			cudaMemcpy(d_p0, p0, N*sizeof(float), cudaMemcpyHostToDevice);
			*/
				
			// Launch WENO3 kernel + SSP-RK3
				
			// (0)
			kernel_WENO3_step0<<<BP, TP>>>(Nx, Ny, d_p0, d_u0, d_v0, d_p1, d_u1, d_v1, rho, c, h, dt, c_rl, eps, dd, d_S, N_PML, S_PML, E_PML, W_PML);	
			kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u1, d_v1, d_p1, Nx, Ny, h, d_S, d_oid, rho, c, n, dt, d_u0, d_v0, d_p0, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 0.0f, 1.0f, 1.0f, phi, phi_old, phi_old, psi_1, psi_1_old, psi_1_old, psi_2, psi_2_old, psi_2_old); // right arrays are the same as middle (can be passed twice as dummy function arguments)
				
			// (1)
			kernel_WENO3_step1<<<BP, TP>>>(Nx, Ny, d_p0, d_u0, d_v0, d_p1, d_u1, d_v1, d_p2, d_u2, d_v2, rho, c, h, dt, c_rl, eps, dd, d_S, N_PML, S_PML, E_PML, W_PML);	
			kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u2, d_v2, d_p2, Nx, Ny, h, d_S, d_oid, rho, c, n, dt, d_u1, d_v1, d_p1, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 3.0f, 1.0f, 4.0f, phi_nu, phi_old, phi, psi_1_nu, psi_1_old, psi_1, psi_2_nu, psi_2_old, psi_2);
				
			// (2)
			kernel_WENO3_step2<<<BP, TP>>>(Nx, Ny, d_p0, d_u0, d_v0, d_p1, d_u1, d_v1, d_p2, d_u2, d_v2, rho, c, h, dt, c_rl, eps, dd, d_S, N_PML, S_PML, E_PML, W_PML);
			kernel_apply_BC_immersed_sdf<<<BP, TP>>>(d_u1, d_v1, d_p1, Nx, Ny, h, d_S, d_oid, rho, c, n, dt, d_u2, d_v2, d_p2, tol, d_Rinf, d_Ak, d_zeta, d_Bl, d_Cl, d_alpha, d_beta, K_m, L_m, d_bmap, 1.0f, 2.0f, 3.0f, phi, phi_old, phi_nu, psi_1, psi_1_old, psi_1_nu, psi_2, psi_2_old, psi_2_nu);

			// exchange state arrays
			cudaMemcpy(d_p0, d_p1, N*sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_u0, d_u1, N*sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_v0, d_v1, N*sizeof(float), cudaMemcpyDeviceToDevice);
				
			// exchange ADE state arrays
			cudaMemcpy(phi_old, phi, nB * K_m * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(psi_1_old, psi_1, nB * L_m * sizeof(float), cudaMemcpyDeviceToDevice);
			cudaMemcpy(psi_2_old, psi_2, nB * L_m * sizeof(float), cudaMemcpyDeviceToDevice);
				
			// sync
			cudaDeviceSynchronize();
				
			print_progress(n, Nt, 50);
		
			// Copy result back to host
			cudaMemcpy(p1, d_p1, N*sizeof(float), cudaMemcpyDeviceToHost);
				
			// write to file for animation (full pressure of size Nx * Nx * pbin)
			if (n % anim_step == 0){
				fwrite(p1, sizeof(float), N, fptr);
        		fflush(fptr);  // Ensure data is written to disk
        		pbin_size += 1;
        	}
        		
        	// write to file for single receiver
			fwrite(&p1[flat(rec_x, rec_y, Nx)], sizeof(float), 1, f_rec);
        	fflush(f_rec);  // Ensure data is written to disk
        		
			n += 1; // update time-step
		}
		
		// Copy final results back to host (for computing TVs)
		cudaMemcpy(p1, d_p1, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(u1, d_u1, N*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(v1, d_v1, N*sizeof(float), cudaMemcpyDeviceToHost);
		
		TV_p_ = TV(p1, Nx, Ny, h);
		TV_u_ = TV(u1, Nx, Ny, h);
		TV_v_ = TV(v1, Nx, Ny, h);
		
		TV_1 = TV_p_ + TV_u_ + TV_v_;
		printf("\n\nTV_0 = %f\n", TV_0);
		printf("\nTV_1 = %f\n", TV_1);
		
		is_TVD(TV_0, TV_1);
	
    	// Free host memory
    	delete[] p0, delete[] u0, delete[] v0;
    	delete[] p1, delete[] u1, delete[] v1;
    	delete[] p2, delete[] u2, delete[] v2;
    	
    	// Free device memory
    	cudaFree(d_p0); cudaFree(d_u0); cudaFree(d_v0);
    	cudaFree(d_p1); cudaFree(d_u1); cudaFree(d_v1);
    	cudaFree(d_p2); cudaFree(d_u2); cudaFree(d_v2);
    	cudaFree(c_rl); cudaFree(dd);
    	
    }
    
    toc = clock();
    
    elapsed_time = ((double)(toc - tic)/60.0)/CLOCKS_PER_SEC; // in minutes
    printf("\n[OK] Simulation time: %f min \n\n", elapsed_time); 
    
    fclose(f_rec);
    printf("[OK] Wrote to file for single receiver\n\n");
    
    FILE *header; // file for data output
	header = fopen("output/header.txt", "w");
	if (header != NULL){
    	fprintf(header, "%d %d %d %f %f %d %d %f %s %f %d\n", Nx, Ny, Nt, Lx, Ly, pbin_size, anim_step, dt, argv[1], h, (int)(tol/h));
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
    
    fclose(fptr);
    fclose(fp);
    
    delete[] S;
    delete[] oid;
    //delete[] R;
    
    cudaFree(d_S);
    cudaFree(d_oid);
    //cudaFree(d_R);
    
    printf("\n[OK] Freed arrays\n");  
    
    // -----------------------
    
    printf("\n[OK] Finished.\n\n"); 
    
    return 0;
}
