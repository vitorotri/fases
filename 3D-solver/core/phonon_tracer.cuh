// routine for phonon tracing

#include <cmath>

// phonon (quasi-particle) struct of arrays
typedef struct phonon{
	uint32_t *id; // particle index / ID
	float *x, *y, *z; // position [m]
	float *u, *v, *w; // particle Lagrangean velocities [m/s]
	float *p; // sound pressure computed from intensity
	// RGB colors
	/*
	uint8_t *R;
	uint8_t *G;
	uint8_t *B;
	*/
} phonon;

// squared velocity
__host__ __device__ u2(float u, float v, float w){
	return u*u + v*v + w*w;
}

// phonon's instantaneous energy
__host__ __device__ en(float rho0, float u2, float c, float p){
	return 0.5f * (rho0*u2 + p*p/(rho0*c));
}

// Should generate similar to a Fibonacci Sphere, more uniform
// Kernel to init particle velocities (omnidirectional)
__global__ init_V(int N_ptc, phonon ptcs, float c){
	int n = blockIdx.x*blockDim.x + threadIdx.x;	
	if (n < N_ptc){
		
		float i = n + 0.5;
		float cz = 1.0 - 2.0*i/N_ptc;
		float sz = sqrtf(1.0 - cz*cz);
		float theta = M_PI * (1.0 + sqrtf(5.0)) * i
		
		ptcs.u[n] = c*sz*cos(theta);
		ptcs.v[n] = c*sz*sin(theta);
		ptcs.w[n] = c*cz;
	}
}

// N_ptc is the number of particles per source
// N is total the number of total particles = N_ptc * N_sources (flattened 2D array)
// dt is obtained from the FVM method, non-adaptive
__global__ void trace(float h, int Nx, int Ny, float *S, uint8_t *oid, float dt, int N_ptc, float *src_arr, float *rec_arr, phonon ptcs){	
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	if (n < N_ptc){
	
		// phonon indexes
		int i, j, k;
		
		i = (int)(ptcs.x[n] / h);
		j = (int)(ptcs.y[n] / h);
		k = (int)(ptcs.z[n] / h);
		
		// gradients from S[] for detecting normals   
    	float gx = (S[flat3(i+1,j,k,Nx,Ny)] - S[flat3(i-1,j,k,Nx,Ny)]) * (0.5f / h);
    	float gy = (S[flat3(i,j+1,k,Nx,Ny)] - S[flat3(i,j-1,k,Nx,Ny)]) * (0.5f / h);
    	float gz = (S[flat3(i,j,k+1,Nx,Ny)] - S[flat3(i,j,k-1,Nx,Ny)]) * (0.5f / h);
    	float gn = sqrtf(gx*gx + gy*gy + gz*gz);
    	
    	gn = fmaxf(gn, 1e-12f); //safety
		
    	float nx = -gx/gn;
    	float ny = -gy/gn;
    	float nz = -gz/gn;
		
		// phonon Lagrangean velocities
		float vn = ptcs.u[n]*nx + ptcs.v[n]*ny + ptcs.w[n]*nz;
		
		ptcs.u[n] += -2.0f*vn*nx;
		ptcs.v[n] += -2.0f*vn*ny;
		ptcs.w[n] += -2.0f*vn*nz;
		
		// phonon positions, non-adaptive
		// x_i = x0_i + v_i * dt
		ptcs.x[n] += ptcs.u[n]*dt;
		ptcs.y[n] += ptcs.v[n]*dt;
		ptcs.z[n] += ptcs.w[n]*dt;
		
		delete[] ptcs;
	}
}
