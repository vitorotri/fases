// routine for phonon tracing

#include <cmath>

// phonon (quasi-particle) struct of arrays
typedef struct phonons{
	uint32_t *id; // particle index / ID
	float *x, *y, *z; // position [m]
	float *u, *v, *w; // particle velocities [m/s]
	float *p; // sound pressure computed from intensity (if absorption of material < 0.5 -> invert phase) [Pa]
	// RGB colors
	/*
	uint8_t *R;
	uint8_t *G;
	uint8_t *B;
	*/
} phonons;

// squared velocity
__host__ __device__ u2(float u, float v, float w){
	return u*u + v*v + w*w;
}

// phonon's instantaneous energy
__host__ __device__ en(float rho0, float u2, float c, float p){
	return 0.5f * (rho0*u2 + p*p/(rho0*c));
}

// N is the number of particles
__global__ void trace(float h, int Nx, int Ny, float *S, float dt, int N){	
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (n < N){
	
		// phonon indexes
		int i, j, k;
		
		i = (int)(ptc.x[n] / h);
		j = (int)(ptc.y[n] / h);
		k = (int)(ptc.z[n] / h);
		
		// gradients from S[] for detecting normals   
    	float gx = (S[flat3(i+1,j,k,Nx,Ny)] - S[flat3(i-1,j,k,Nx,Ny)]) * (0.5f / h);
    	float gy = (S[flat3(i,j+1,k,Nx,Ny)] - S[flat3(i,j-1,k,Nx,Ny)]) * (0.5f / h);
    	float gz = (S[flat3(i,j,k+1,Nx,Ny)] - S[flat3(i,j,k-1,Nx,Ny)]) * (0.5f / h);
    	float gn = sqrtf(gx*gx + gy*gy + gz*gz);
    	
    	gn = fmaxf(gn, 1e-12f); //safety
		
    	float nx = -gx/gn;
    	float ny = -gy/gn;
    	float nz = -gz/gn;
		
		// phonon velocities
		float vn = ptc.u[n]*nx + ptc.v[n]*ny + ptc.w[n]*nz;
		
		ptc.u[n] += -2.0f*vn*nx;
		ptc.v[n] += -2.0f*vn*ny;
		ptc.w[n] += -2.0f*vn*nz;
		
		// phonon positions
		// x_i = x0_i + v_i * dt
		ptc.x[n] += ptc.u[n]*dt;
		ptc.y[n] += ptc.v[n]*dt;
		ptc.z[n] += ptc.w[n]*dt;
		
		delete[] ptc;
	}
}
