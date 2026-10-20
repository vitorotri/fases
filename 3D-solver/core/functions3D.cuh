#include <cmath>
#include <cstring>
#include <random>
#include "parsing3D.h"
#include "filterABC.h"

#define P_0 101325 // [Pa]
#define T_0 293.15 // [K]

// Split-PML layer thickness in cells (per face). Used by BOTH sPML (below) and the
// px/py/pz tile allocation in main3D.cu — keep them in sync via this one constant.
#ifndef PML_CELLS
#define PML_CELLS 20
#endif

// Computes relaxation frequency for Oxygen in air
__host__ float frO(float p_a, float h){
	return p_a*(24.0 + 4.04*1e4*h*(0.02 + h)/(0.391 + h))/P_0;
}

// Computes relaxation frequency for Nitrogen in air
__host__ float frN(float p_a, float h, float T){
	float A, B;
	A = p_a*std::sqrt(T_0/T)/P_0;
	B = 9.0 + 280.0*h*std::exp(-4.17*(std::pow(T/T_0,-1.0/3.0) - 1.0));
	return A*B;
}

// Computes the coefficient of attenuation of air given, frequency, temperature, directly multiplied by 0.115
// and converted to [1/m] units
__host__ float alpha_air(float p_a, float h, float T, float f){
	float A, B, C, D;
	A = 1.84*1e-11*P_0*std::sqrt(T/T_0)/p_a;
	B = std::pow(T/T_0,-5.0/2.0);
	C = 0.01275*std::exp(-2239.1/T)*(1.0/(frO(p_a,h) + (f*f/frO(p_a,h))));
	D = 0.10680*std::exp(-3352.0/T)*(1.0/(frN(p_a,h,T) + (f*f/frN(p_a,h,T))));
	return 0.115*8.686*f*f*(A + B*(C+D));
}

// Barometric formula
__host__ float p_atm(float alt, float T){
	float M_air = 0.02897; // [Kg]/[mol]
	float g = 9.8067; // [m]/[s^2]
	float R = 8.314; // [J]/[mol K]
	return P_0*std::exp(-M_air*g*alt/(R*T));
}

__host__ __device__ inline size_t flat3(int i, int j, int k, int Nx, int Ny){
    return i + Nx*(j + k*Ny);
}

// Godunov and Roe fluxes (for WENO, no simple substitutions)

// inputs on p,u,v,w are the reconstructed values
// pP and pM denote, for example, p_plus and p_minus
__host__ __device__ float F_p_Godunov(float pM, float pP, float uM, float uP, float rho, float c){
    return 0.5f*(rho*c*c*(uP + uM) - c*(pP - pM));
}

__host__ __device__ float F_u_Godunov(float pM, float pP, float uM, float uP, float rho, float c){
    return 0.5f*((pP + pM)/rho - c*(uP - uM));
}

__host__ __device__ float G_p_Godunov(float pM, float pP, float vM, float vP, float rho, float c){
    return 0.5f*(rho*c*c*(vP + vM) - c*(pP - pM));
}

__host__ __device__ float G_v_Godunov(float pM, float pP, float vM, float vP, float rho, float c){
    return 0.5f*((pP + pM)/rho - c*(vP - vM));
}

__host__ __device__ float H_p_Godunov(float pM, float pP, float wM, float wP, float rho, float c){
    return 0.5f*(rho*c*c*(wP + wM) - c*(pP - pM));
}

__host__ __device__ float H_w_Godunov(float pM, float pP, float wM, float wP, float rho, float c){
    return 0.5f*((pP + pM)/rho - c*(wP - wM));
}

__host__ float SPL(float p){
	return 20.0*log10f(fabsf(p)/(0.00002));
}

// Total variations

// compute on regular variables
// Total Variation of variable Q
__host__ __device__ float TV(float *__restrict__ Q, int Nx, int Ny, int Nz, float h){
	float retval = 0.0;
	for (int i = 0; i < Nx-1; i++){
		for (int j = 0; j < Ny-1; j++){
			for (int k = 0; k < Nz-1; k++){
				retval += fabsf(Q[flat3(i+1,j,k,Nx,Ny)] - Q[flat3(i,j,k,Nx,Ny)])
						+ fabsf(Q[flat3(i,j+1,k,Nx,Ny)] - Q[flat3(i,j,k,Nx,Ny)])
						+ fabsf(Q[flat3(i,j,k+1,Nx,Ny)] - Q[flat3(i,j,k,Nx,Ny)]);
			}
		}
	}
	return h*h*retval;
}

// check if scheme is TVD
__host__ void is_TVD(float TV_0, float TV_1){
	if (TV_1 <= TV_0){
		printf("\n[OK] Scheme is numerically TVD\n\n");
	}
	else printf("\n[!] Scheme is not TVD\n\n");
}

// TV norm is not the best to estimate energy injection into the system if source is a sinusoid or similar with E0 = 0.0
__host__ float Energy(float *p, float *u, float *v, float *w, int Nx, int Ny, int Nz, float h, float rho, float c){

	float sum = 0.0;
	for (int i = 0; i < Nx; i++){
		for (int j = 0; j < Ny; j++){
			for (int k = 0; k < Nz; k++){
				
				int idx = flat3(i,j,k,Nx,Ny);
				
				float pi = p[idx];
				float ui = u[idx], vi = v[idx], wi = w[idx];
				
				sum += (pi*pi) / (rho * c*c);
				sum += rho * (ui*ui + vi*vi + wi*wi);
			}
		}
	}
	return 0.5 * sum;
}

// check if scheme is conservative
__host__ void is_conservative(float E0, float E1){
	if (E1 <= E0){
		printf("\n[OK] Scheme is conservative\n\n");
	}
	else printf("\n[!] Scheme is not conservative\n\n");
}

__device__ __forceinline__
float cubic_interp_1d(float fm1, float f0, float f1, float f2, float s){
    // s in [0,1): position between grid points i (0) and i+1
    float s2 = s * s;

    float w_m1 = -s * (s - 2.0f) * (s - 1.0f) / 6.0f;
    float w_0  =  (s - 2.0f) * (s - 1.0f) * (s + 1.0f) / 2.0f;
    float w_1  =  s * (-s2 + s + 2.0f) / 2.0f;
    float w_2  =  s * (s2 - 1.0f) / 6.0f;

    return w_m1 * fm1 + w_0 * f0 + w_1 * f1 + w_2 * f2;
}

__device__ __forceinline__
float tricubic_interp(const float *__restrict__ f, int Nx, int Ny, int Nz, float Dx, float Dy, float Dz, float h){
    // Convert physical position to grid coordinates
    
    float gx = Dx / h - 0.5;
    float gy = Dy / h - 0.5;
    float gz = Dz / h - 0.5;
	
	/*
	float gx = Dx / h;
    float gy = Dy / h;
    float gz = Dz / h;
	*/
    int i = (int)floorf(gx);
	int j = (int)floorf(gy);
	int k = (int)floorf(gz);
	
	// clamp so that i-1 >= 0 and i+2 < Nx  -> i in [1, Nx-3]
	i = max(1, min(i, Nx - 3));
	j = max(1, min(j, Ny - 3));
	k = max(1, min(k, Nz - 3));
	
    float sx = gx - (float)i;   // in [0,1)
    float sy = gy - (float)j;   // in [0,1)
    float sz = gz - (float)k;   // in [0,1)

    // neighbor indices in each direction: base-1, base, base+1, base+2
    int ixm1 = i - 1, ix0 = i, ixp1 = i + 1, ixp2 = i + 2;
    int iym1 = j - 1, iy0 = j, iyp1 = j + 1, iyp2 = j + 2;
    int izm1 = k - 1, iz0 = k, izp1 = k + 1, izp2 = k + 2;

    // Step 1: For each z-slab (4) and y-row (4), interpolate along x -> 16 values
    // Then Step 2: For each z-slab (4), interpolate those 4 values along y -> 4 values
    // Then Step 3: Interpolate the 4 values along z -> final

    float hy[4];  // result after y-interp for each z-slab

    // unroll z = {k-1,k,k+1,k+2}
    const int zlist[4] = { izm1, iz0, izp1, izp2 };
    const int ylist[4] = { iym1, iy0, iyp1, iyp2 };
    const int xlist[4] = { ixm1, ix0, ixp1, ixp2 };

    for (int zz = 0; zz < 4; ++zz){
        int kz = zlist[zz];

        // Interpolate in x for each of 4 y-rows at this kz
        float gxrow[4];

        #pragma unroll
        for (int yy = 0; yy < 4; ++yy)
        {
            int jy = ylist[yy];

            float fm1 = f[ flat3(xlist[0], jy, kz, Nx, Ny) ];
            float f0  = f[ flat3(xlist[1], jy, kz, Nx, Ny) ];
            float f1  = f[ flat3(xlist[2], jy, kz, Nx, Ny) ];
            float f2  = f[ flat3(xlist[3], jy, kz, Nx, Ny) ];

            gxrow[yy] = cubic_interp_1d(fm1, f0, f1, f2, sx);
        }

        // Interpolate the 4 intermediate values along y
        hy[zz] = cubic_interp_1d(gxrow[0], gxrow[1], gxrow[2], gxrow[3], sy);
    }

    // Interpolate the 4 values along z
    return cubic_interp_1d(hy[0], hy[1], hy[2], hy[3], sz);
}

__host__ __device__ inline float R_to_Zb(float R, float Z0){
    // clamp away from ±1
    const float eps = 1e-6f;
    if (R >  1.f - eps) R =  1.f - eps;
    if (R < -1.f + eps) R = -1.f + eps;
    return Z0 * (1.0f + R) / (1.0f - R);
}

// Raised-cosine bump: full width = 2*w (Smits: 0.04 m => w=0.02 m)
__host__ __device__ float raised_cosine(float x, float x0, float w){
    float xi = (x - x0) / w;
    if (fabsf(xi) > 1.0f) return 0.0f;
    return 0.5f * (1.0f + cosf((float)M_PI * xi));
}

// case above but correctly ranges only inside the fluid
// tol need to be 4*h (?)
__device__ float nu_phi(float r){
	if (r <= 4.0f && r >= 0.0f){
		return 0.5f*(1.0f + std::cos(0.25f*M_PI*r));
	}
	return 0.0f;
}

__device__ inline float brinkman_chi(float s, float h){
    if (s < 0.0f) return 1.0f;      // solid
    if (s > h)   return 0.0f;       // fluid (outside penalization band)
    // In the layer 0 <= s <= h, smoothly ramp from 1 -> 0
    return (h - s) / h;
}

__device__ float peskin_chi(float s){
	if (s >= 0.0f && s < 1.0f){
		return (3.0f - 2.0f*s + sqrtf(1.0f + 4.0f*s - 4.0f*s*s))/8.0f;
	}
	if (s >= 1.0f && s < 2.0f){
		return (5.0f - 2.0f*s - sqrtf(-7.0f + 12.0f*s - 4.0f*s*s))/8.0f;
	}
	return 0.0f;
}

// correctly considers cases where s < h, so it does not sum itself with old U[Cij] values
__device__ inline float chi_layered(float s, float h, float tol){
    if (s >= tol) return 0.0f;
	if (s <= h)	return 1.0f;
	
    float r = (s - h) / (tol -  h);
    float smooth = r*r*(3.0f - 2.0f*r);
    
    return 1.0f - smooth;
}

__host__ bool check_energy_coeffs(float R, float T){
	float aux = R*R + T*T;
	if (!(aux <= 1.0f)){
		printf("\n\n[!] WARNING: Reflection and Transmission must satisfy |R|**2 + |T|**2 <= 1\nTotal Variation may grow...\n");
		return 1;
	}
	return 0;
}

__global__ void kernel_periodic_trick_z(float *S, uint8_t *oid, int Nx, int Ny, int Nz, float h){
                          
    int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
	if ((i < Nx) && (j < Ny)){
    
    	// lower ghosts: 0,1,2, ...
        for (int k = 0; k < 6; k++){
          	oid[flat3(i, j, k, Nx, Ny)] = 0xFF;
          	S[flat3(i, j, k, Nx, Ny)] = (1+k) * h; // should be smaller, but we want to trick WENO kernels
		}

        // upper ghosts: ..., Nz-3,Nz-2,Nz-1
        for (int k = Nz-7; k < Nz; k++){
        	oid[flat3(i, j, k, Nx, Ny)] = 0xFF;
        	S[flat3(i, j, k, Nx, Ny)] = (Nz - 1+k) * h; // should be smaller, but we want to trick WENO kernels
        }
	
	}
}

__global__ void kernel_periodic_trick_x(float *__restrict__ S, uint8_t *__restrict__ oid,
                                        int Nx, int Ny, int Nz, float h)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= Ny || k >= Nz) return;

    for (int i = 0; i < 8; ++i) {
        oid[flat3(i, j, k, Nx, Ny)] = 0xFF;
        S[flat3(i, j, k, Nx, Ny)] = (i + 1) * h;
    }

    for (int t = 0; t < 8; ++t) {
        int i = Nx - 8 + t;
        oid[flat3(i, j, k, Nx, Ny)] = 0xFF;
        S[flat3(i, j, k, Nx, Ny)] = (8 - t) * h;
    }
}

__global__ void kernel_apply_periodic_x(float *__restrict__ p, float *__restrict__ u, float *__restrict__ v, float *__restrict__ w, int Nx, int Ny, int Nz){
    int j = blockIdx.x * blockDim.x + threadIdx.x + 3;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 3;

    if (j >= Ny - 3 || k >= Nz - 3) return;

    // left ghosts 0,1,2 <- Nx-6,Nx-5,Nx-4
    p[flat3(0, j, k, Nx, Ny)] = p[flat3(Nx-6, j, k, Nx, Ny)];
    u[flat3(0, j, k, Nx, Ny)] = u[flat3(Nx-6, j, k, Nx, Ny)];
    v[flat3(0, j, k, Nx, Ny)] = v[flat3(Nx-6, j, k, Nx, Ny)];
    w[flat3(0, j, k, Nx, Ny)] = w[flat3(Nx-6, j, k, Nx, Ny)];

    p[flat3(1, j, k, Nx, Ny)] = p[flat3(Nx-5, j, k, Nx, Ny)];
    u[flat3(1, j, k, Nx, Ny)] = u[flat3(Nx-5, j, k, Nx, Ny)];
    v[flat3(1, j, k, Nx, Ny)] = v[flat3(Nx-5, j, k, Nx, Ny)];
    w[flat3(1, j, k, Nx, Ny)] = w[flat3(Nx-5, j, k, Nx, Ny)];

    p[flat3(2, j, k, Nx, Ny)] = p[flat3(Nx-4, j, k, Nx, Ny)];
    u[flat3(2, j, k, Nx, Ny)] = u[flat3(Nx-4, j, k, Nx, Ny)];
    v[flat3(2, j, k, Nx, Ny)] = v[flat3(Nx-4, j, k, Nx, Ny)];
    w[flat3(2, j, k, Nx, Ny)] = w[flat3(Nx-4, j, k, Nx, Ny)];

    // right ghosts Nx-3,Nx-2,Nx-1 <- 3,4,5
    p[flat3(Nx-3, j, k, Nx, Ny)] = p[flat3(3, j, k, Nx, Ny)];
    u[flat3(Nx-3, j, k, Nx, Ny)] = u[flat3(3, j, k, Nx, Ny)];
    v[flat3(Nx-3, j, k, Nx, Ny)] = v[flat3(3, j, k, Nx, Ny)];
    w[flat3(Nx-3, j, k, Nx, Ny)] = w[flat3(3, j, k, Nx, Ny)];

    p[flat3(Nx-2, j, k, Nx, Ny)] = p[flat3(4, j, k, Nx, Ny)];
    u[flat3(Nx-2, j, k, Nx, Ny)] = u[flat3(4, j, k, Nx, Ny)];
    v[flat3(Nx-2, j, k, Nx, Ny)] = v[flat3(4, j, k, Nx, Ny)];
    w[flat3(Nx-2, j, k, Nx, Ny)] = w[flat3(4, j, k, Nx, Ny)];

    p[flat3(Nx-1, j, k, Nx, Ny)] = p[flat3(5, j, k, Nx, Ny)];
    u[flat3(Nx-1, j, k, Nx, Ny)] = u[flat3(5, j, k, Nx, Ny)];
    v[flat3(Nx-1, j, k, Nx, Ny)] = v[flat3(5, j, k, Nx, Ny)];
    w[flat3(Nx-1, j, k, Nx, Ny)] = w[flat3(5, j, k, Nx, Ny)];
}

__global__ void kernel_apply_periodic_z(float *__restrict__ p, float *__restrict__ u, float *__restrict__ v, float *__restrict__ w, int Nx, int Ny, int Nz){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 3;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 3;

    if (i >= Nx-3 || j >= Ny-3) return;

    // lower ghosts 0,1,2 <- Nz-6,Nz-5,Nz-4
    p[flat3(i,j,0,Nx,Ny)] = p[flat3(i,j,Nz-6,Nx,Ny)];
    u[flat3(i,j,0,Nx,Ny)] = u[flat3(i,j,Nz-6,Nx,Ny)];
    v[flat3(i,j,0,Nx,Ny)] = v[flat3(i,j,Nz-6,Nx,Ny)];
    w[flat3(i,j,0,Nx,Ny)] = w[flat3(i,j,Nz-6,Nx,Ny)];

    p[flat3(i,j,1,Nx,Ny)] = p[flat3(i,j,Nz-5,Nx,Ny)];
    u[flat3(i,j,1,Nx,Ny)] = u[flat3(i,j,Nz-5,Nx,Ny)];
    v[flat3(i,j,1,Nx,Ny)] = v[flat3(i,j,Nz-5,Nx,Ny)];
    w[flat3(i,j,1,Nx,Ny)] = w[flat3(i,j,Nz-5,Nx,Ny)];

    p[flat3(i,j,2,Nx,Ny)] = p[flat3(i,j,Nz-4,Nx,Ny)];
    u[flat3(i,j,2,Nx,Ny)] = u[flat3(i,j,Nz-4,Nx,Ny)];
    v[flat3(i,j,2,Nx,Ny)] = v[flat3(i,j,Nz-4,Nx,Ny)];
    w[flat3(i,j,2,Nx,Ny)] = w[flat3(i,j,Nz-4,Nx,Ny)];

    // upper ghosts Nz-3,Nz-2,Nz-1 <- 3,4,5
    p[flat3(i,j,Nz-3,Nx,Ny)] = p[flat3(i,j,3,Nx,Ny)];
    u[flat3(i,j,Nz-3,Nx,Ny)] = u[flat3(i,j,3,Nx,Ny)];
    v[flat3(i,j,Nz-3,Nx,Ny)] = v[flat3(i,j,3,Nx,Ny)];
    w[flat3(i,j,Nz-3,Nx,Ny)] = w[flat3(i,j,3,Nx,Ny)];

    p[flat3(i,j,Nz-2,Nx,Ny)] = p[flat3(i,j,4,Nx,Ny)];
    u[flat3(i,j,Nz-2,Nx,Ny)] = u[flat3(i,j,4,Nx,Ny)];
    v[flat3(i,j,Nz-2,Nx,Ny)] = v[flat3(i,j,4,Nx,Ny)];
    w[flat3(i,j,Nz-2,Nx,Ny)] = w[flat3(i,j,4,Nx,Ny)];

    p[flat3(i,j,Nz-1,Nx,Ny)] = p[flat3(i,j,5,Nx,Ny)];
    u[flat3(i,j,Nz-1,Nx,Ny)] = u[flat3(i,j,5,Nx,Ny)];
    v[flat3(i,j,Nz-1,Nx,Ny)] = v[flat3(i,j,5,Nx,Ny)];
    w[flat3(i,j,Nz-1,Nx,Ny)] = w[flat3(i,j,5,Nx,Ny)];
}

// Trilinear sample at a physical point (same cell-centred grid convention as tricubic_interp).
// A convex combination of the 8 surrounding cells: it can never exceed what it reads. The
// tricubic's negative lobes (-1/16) can, and on an OBLIQUE wall -- where the probe point falls
// between cells and its stencil covers the wall's own blended band cells -- that is a loop
// interior -> WENO -> band -> probe -> interior with gain > 1 that 1 % absorption does not damp
// (CR4 trace (32,42,74): clean plane, 10 parallel readers, x10 per 170 steps). On axis-aligned
// walls the probe lands on cell centres and both interpolants are the exact copy.
__device__ __forceinline__
float trilinear_interp(const float *__restrict__ f, int Nx, int Ny, int Nz, float Dx, float Dy, float Dz, float h){
    float gx = Dx / h - 0.5f, gy = Dy / h - 0.5f, gz = Dz / h - 0.5f;
    int i = (int)floorf(gx), j = (int)floorf(gy), k = (int)floorf(gz);
    i = max(0, min(i, Nx - 2)); j = max(0, min(j, Ny - 2)); k = max(0, min(k, Nz - 2));
    const float sx = gx - (float)i, sy = gy - (float)j, sz = gz - (float)k;
    const float c00 = f[flat3(i,j,k,Nx,Ny)]*(1-sx) + f[flat3(i+1,j,k,Nx,Ny)]*sx;
    const float c10 = f[flat3(i,j+1,k,Nx,Ny)]*(1-sx) + f[flat3(i+1,j+1,k,Nx,Ny)]*sx;
    const float c01 = f[flat3(i,j,k+1,Nx,Ny)]*(1-sx) + f[flat3(i+1,j,k+1,Nx,Ny)]*sx;
    const float c11 = f[flat3(i,j+1,k+1,Nx,Ny)]*(1-sx) + f[flat3(i+1,j+1,k+1,Nx,Ny)]*sx;
    const float c0 = c00*(1-sy) + c10*sy, c1 = c01*(1-sy) + c11*sy;
    return c0*(1-sz) + c1*sz;
}

// non-implicit, solved with regular SSP-RK3, but coefficients alpha, beta and zeta are restricted in terms
// of time-step dt in the fitting script
__device__ float ADE_solver(int b_ID, uint8_t m_ID, float w_in, float dt, float *__restrict__ A, float *__restrict__ zeta, float *__restrict__ B, float *__restrict__ C, float *__restrict__ alpha, float *__restrict__ beta, float *__restrict__ R_inf, int K, int L, float a_SSP, float b_SSP, float c_SSP, float *__restrict__ phi_left, float *__restrict__ phi_middle, float *__restrict__ phi_right, float *__restrict__ psi_1_left, float *__restrict__ psi_1_middle, float *__restrict__ psi_1_right, float *__restrict__ psi_2_left, float *__restrict__ psi_2_middle, float *__restrict__ psi_2_right){
    float sum_A  = 0.0f;
    float sum_BC = 0.0f;

    for (int k = 0; k < K; k++){
        int m = m_ID*K + k;	// flattened material coeff idx
        int s = b_ID*K + k; // flattened compact state idx

		// -------------------------------------------------
		
		float L_phi = w_in - zeta[m]*phi_right[s];
		
		phi_left[s] = a_SSP*phi_middle[s] + b_SSP*(phi_right[s] + dt*L_phi);
		phi_left[s] /= c_SSP;
		sum_A += A[m] * phi_left[s];
    }

    for (int l = 0; l < L; l++){
        int m = m_ID*L + l;	// flattened material coeff idx
        int s = b_ID*L + l;	// flattened compact state idx
        
        float L_psi_1 = w_in - (alpha[m]*psi_1_right[s] + beta[m]*psi_2_right[s]);
        float L_psi_2 = beta[m]*psi_1_right[s] - alpha[m]*psi_2_right[s];

		psi_1_left[s] = a_SSP*psi_1_middle[s] + b_SSP*(psi_1_right[s] + dt*L_psi_1);
		psi_1_left[s] /= c_SSP;
		psi_2_left[s] = a_SSP*psi_2_middle[s] + b_SSP*(psi_2_right[s] + dt*L_psi_2);
		psi_2_left[s] /= c_SSP;

        sum_BC += B[m]*psi_1_left[s] + C[m]*psi_2_left[s];
    }

    return R_inf[m_ID]*w_in + sum_A + sum_BC;
}

// clean version
__global__ void kernel_apply_BC_immersed_sdf(float *__restrict__ u, float *__restrict__ v, float *__restrict__ w, float *__restrict__ p, int Nx, int Ny, int Nz, float h, float *S, uint8_t *__restrict__ oid, float rho, float c, int n, float dt, float *__restrict__ u_old, float *__restrict__ v_old, float *__restrict__ w_old, float *__restrict__ p_old, float tol, float *__restrict__ Rinf, float *__restrict__ Ak, float *__restrict__ zeta, float *__restrict__ Bl, float *__restrict__ Cl, float *__restrict__ alpha, float *__restrict__ beta, int K_m, int L_m, int *bmap, float a_SSP, float b_SSP, float c_SSP, float *__restrict__ phi_left, float *__restrict__ phi_middle, float *__restrict__ phi_right, float *__restrict__ psi_1_left, float *__restrict__ psi_1_middle, float *__restrict__ psi_1_right, float *__restrict__ psi_2_left, float *__restrict__ psi_2_middle, float *__restrict__ psi_2_right){

	int i = blockIdx.x*blockDim.x + threadIdx.x + 3;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
	int k = blockIdx.z*blockDim.z + threadIdx.z + 3;
    if ((i < Nx-3) && (j < Ny-3) && (k < Nz-3)){
		
    	int Cijk = flat3(i,j,k,Nx,Ny);
    	
    	float s = S[Cijk]; // signed distance from cell center to boundary
    	
    	// only act on immersed cells (fluid-side band from your OID map), provided
    	// those values are only on positive s, which is the case here
    	// oid[] ranges from 1 to nMat, 255 elsewhere
		const uint8_t o_ID = oid[Cijk]; // object's material ID
		int b_ID = bmap[Cijk]; // boundary map for phi and psi (ADE)
		if (o_ID == 0 || o_ID == 0xFF || b_ID < 0) return;
		
		//if (s >= h) return; // no blending test
		
		// gradients from S[] for detecting normals   
    	float gx = (S[flat3(i+1,j,k,Nx,Ny)] - S[flat3(i-1,j,k,Nx,Ny)]) * (0.5f / h);
    	float gy = (S[flat3(i,j+1,k,Nx,Ny)] - S[flat3(i,j-1,k,Nx,Ny)]) * (0.5f / h);
    	float gz = (S[flat3(i,j,k+1,Nx,Ny)] - S[flat3(i,j,k-1,Nx,Ny)]) * (0.5f / h);
    	float gn = sqrtf(gx*gx + gy*gy + gz*gz);
    	gn = fmaxf(gn, 1e-12f); //safety
	
    	float nx = -gx/gn;
    	float ny = -gy/gn;
    	float nz = -gz/gn;
		
    	// cell center
    	float Cx = (i + 0.5f) * h;
    	float Cy = (j + 0.5f) * h;
    	float Cz = (k + 0.5f) * h;
    	
    	// footpoint on boundary: W = C - s * n_hat  (since S>0 fluid, S<0 solid)
    	float Wx = Cx + s * nx;
    	float Wy = Cy + s * ny;
    	float Wz = Cz + s * nz;
    	
    	// interpolation point in the fluid along n, at probe_len from the wall (see its comment)
    	const float l = s + 2.0f*h;
    	float Dx = Wx - l * nx;
    	float Dy = Wy - l * ny;
    	float Dz = Wz - l * nz;
		
    	float pD = trilinear_interp(p_old, Nx, Ny, Nz, Dx, Dy, Dz, h);
    	float uD = trilinear_interp(u_old, Nx, Ny, Nz, Dx, Dy, Dz, h);
    	float vD = trilinear_interp(v_old, Nx, Ny, Nz, Dx, Dy, Dz, h);
    	float wD = trilinear_interp(w_old, Nx, Ny, Nz, Dx, Dy, Dz, h);

        const float un =  uD*nx + vD*ny + wD*nz;
        // tangential vector (free-slip)
		const float utx = uD - un*nx;
		const float uty = vD - un*ny;
		const float utz = wD - un*nz;

        const float Z0 = rho * c;

        // ------------------- incoming characteristic from fluid LOCAL side (towards obstacle) -------
        const float w_out_local = pD + Z0 * un;
        
        int m_ID = o_ID - 1; // material ID
        
        const float w_in_local = ADE_solver(
    		b_ID, m_ID, w_out_local, dt,
    		Ak, zeta,
    		Bl, Cl, alpha, beta,
    		Rinf,
    		K_m, L_m,
    		a_SSP, b_SSP, c_SSP,
    		phi_left, phi_middle, phi_right,
    		psi_1_left, psi_1_middle, psi_1_right,
    		psi_2_left, psi_2_middle, psi_2_right
		);
		
        // boundary state on THIS side (same as before when T_obj = 0)
        const float p_b  = 0.5f * (w_in_local + w_out_local);
        const float un_b = (w_out_local - w_in_local) / (2.0f * Z0);

        // back to Cartesian
        const float u_b = un_b*nx + utx;
		const float v_b = un_b*ny + uty;
		const float w_b = un_b*nz + utz;  	
    	
    	const float chi = chi_layered(s, h, tol);
		
		// immersed boundary smoothing / blending with value computed with WENO (s > h) (Brinkman penalization)
		// as well as full values (s <= h => chi = 1) / (s >= tol => chi = 0)
		u[Cijk] = (1.0f - chi) * u[Cijk] + chi * u_b;
    	v[Cijk] = (1.0f - chi) * v[Cijk] + chi * v_b;
    	w[Cijk] = (1.0f - chi) * w[Cijk] + chi * w_b;
    	p[Cijk] = (1.0f - chi) * p[Cijk] + chi * p_b;
    	
    	/*
    	// no blending
		u[Cijk] = u_b;
    	v[Cijk] = v_b;
    	w[Cijk] = w_b;
    	p[Cijk] = p_b;
    	*/
    }
}

//
// sPML Layer - "simple" unsplit PML according to:
// 
// https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/AcousticsTimeDomain.html
// 
// as well as Montanaro, Knöpfel, Heutschi and Pieren (2022) - "Parallelized FDTD simulations to improve
// engineering models of urban road traffic noise"
//
//
// Memory-lean split PML. Total pressure p stays a full-grid field; each axis sub-field
// px/py/pz lives ONLY in its own boundary band (px_size = 2*PML_CELLS*Ny*Nz etc, allocated
// in main3D.cu). Bérenger split, reformulated so no px+py+pz reconstruction is needed:
//
//     dp_i/dt = L_i - sig_i * p_i          (L_i = i-direction flux divergence, passed in)
//     dp/dt   = L - sum_i sig_i * p_i       (p already got +dt*L in the kernel)
//
// A sub-field is touched ONLY inside its own enabled face-band, so there is no cross-axis
// access and no out-of-bounds. Velocity: each component damped by its own-axis conductivity.
//
__host__ __device__
void sPML(float *__restrict__ p, float *__restrict__ px, float *__restrict__ py, float *__restrict__ pz,
          float Lx, float Ly, float Lz,
          float *__restrict__ u, float *__restrict__ v, float *__restrict__ w,
          int i, int j, int k, int Nx, int Ny, int Nz, float h, float dt, float c,
          float U_PML, float D_PML, float F_PML, float B_PML, float R_PML, float L_PML, float SSP_w){

    const int   d   = PML_CELLS;
    const int   ng  = PARSE3D_GHOST;    // band on UPDATED cells only, i in [ng, ng+d)
    const int   Cijk = flat3(i, j, k, Nx, Ny);
    const float R0 = 5e-8, lnR0 = logf(R0), m = 2.0;

    // depth anchored at interior edge ng: peak sigma on outermost updated cell, ghosts excluded.
    const float xm = fminf(1.f, fmaxf(0.f, ((float)(ng + d) - (float)i) / (float)d));
    const float xp = fminf(1.f, fmaxf(0.f, ((float)i - ((float)(Nx-1-ng) - (float)d)) / (float)d));
    const float ym = fminf(1.f, fmaxf(0.f, ((float)(ng + d) - (float)j) / (float)d));
    const float yp = fminf(1.f, fmaxf(0.f, ((float)j - ((float)(Ny-1-ng) - (float)d)) / (float)d));
    const float zm = fminf(1.f, fmaxf(0.f, ((float)(ng + d) - (float)k) / (float)d));
    const float zp = fminf(1.f, fmaxf(0.f, ((float)k - ((float)(Nz-1-ng) - (float)d)) / (float)d));

    const float pref = -0.5f * c * (m + 1.0) * lnR0 / ((float)d * h);

    // per-axis conductivity (face-flagged; 0 outside the band via the clamp)
    const float sig_x = L_PML*pref*powf(xm,m) + R_PML*pref*powf(xp,m);
    const float sig_y = B_PML*pref*powf(ym,m) + F_PML*pref*powf(yp,m);
    const float sig_z = D_PML*pref*powf(zm,m) + U_PML*pref*powf(zp,m);

    const float dte = SSP_w * dt;

    // Only enter a band whose face is enabled -> tile index is always in range.
    const bool bx = (L_PML > 0.f && i >= ng && i < ng + d) || (R_PML > 0.f && i >= Nx-ng-d && i <= Nx-1-ng);
    const bool by = (B_PML > 0.f && j >= ng && j < ng + d) || (F_PML > 0.f && j >= Ny-ng-d && j <= Ny-1-ng);
    const bool bz = (D_PML > 0.f && k >= ng && k < ng + d) || (U_PML > 0.f && k >= Nz-ng-d && k <= Nz-1-ng);

	// split
	/*
    float corr = 0.0f;   // pressure correction = sum_i sig_i * p_i
    if (bx){
        const int ix = (i < d) ? i : i - (Nx - 2*d);
        const int X  = ix + (2*d)*(j + Ny*k);       // px tile: 2*d wide in x, full y,z
        px[X] += dte*(Lx - sig_x*px[X]);
        corr  += sig_x*px[X];
    }
    if (by){
        const int jy = (j < d) ? j : j - (Ny - 2*d);
        const int Y  = i + Nx*(jy + (2*d)*k);        // py tile: 2*d wide in y
        py[Y] += dte*(Ly - sig_y*py[Y]);
        corr  += sig_y*py[Y];
    }
    if (bz){
        const int kz = (k < d) ? k : k - (Nz - 2*d);
        const int Z  = i + Nx*(j + Ny*kz);           // pz tile: 2*d wide in z
        pz[Z] += dte*(Lz - sig_z*pz[Z]);
        corr  += sig_z*pz[Z];
    }
    p[Cijk] -= dte*corr;
	*/
    u[Cijk] -= dte*sig_x*u[Cijk];
    v[Cijk] -= dte*sig_y*v[Cijk];
    w[Cijk] -= dte*sig_z*w[Cijk];

    // unsplit version
    p[Cijk] -= dte*(sig_x + sig_y + sig_z)*p[Cijk];
}


//
// Unsplit recursive convolutional PML (cPML). Stretch a directional divergence D -> D/kappa + psi.
// Split into two ops so the FIELD can be stretched in every RK stage (RK-consistent) while the
// psi memory advances only ONCE per step:
//   cpml_corr : read-only correction (D_pml - D) using the current (frozen) psi.
//   cpml_adv  : advance psi by exact recursive convolution (call once per step).
// alpha=0 => plain PML; alpha>0 => CFS.
//
__device__ __forceinline__
float cpml_corr(float D, float kappa, float psi){
    return (1.0f/kappa - 1.0f)*D + psi;       // D_pml - D
}
__device__ __forceinline__
void cpml_adv(float D, float kappa, float sigma, float alpha, float dt, float &psi){
    const float b  = expf(-(alpha + sigma/kappa)*dt);
    const float ca = (sigma > 0.f) ? (sigma/(kappa*(sigma + kappa*alpha)))*(b - 1.f) : 0.f;
    psi = b*psi + ca*D;
}

//
// cPML layer. Same tiling/placement as sPML (once per step, in kernel_WENO_stage2). Six psi
// band-tiles: p<-x,y,z and u<-x, v<-y, w<-z. Needs the per-axis flux divergences (Lp*, Lu/Lv/Lw).
// kappa_max/alpha_max are the CFS knobs (default alpha_max=0 => reproduces a plain sigma-PML).
//
// SSP_w = the stage's dt-weight in the SSP-RK3 combination (step0 1.0, step1 0.25, step2 2/3):
//   the correction rides the same weight as the flux, so the FIELD is RK-consistent.
// advance = true only on the final stage (step2): psi advances once per step at full dt,
//   using that stage's divergence, AFTER the correction has read the frozen psi.
__device__
void cPML(float *__restrict__ p, float *__restrict__ u, float *__restrict__ v, float *__restrict__ w,
          float *__restrict__ psi_px, float *__restrict__ psi_py, float *__restrict__ psi_pz,
          float *__restrict__ psi_ux, float *__restrict__ psi_vy, float *__restrict__ psi_wz,
          float Lpx, float Lpy, float Lpz, float Lux, float Lvy, float Lwz,
          int i, int j, int k, int Nx, int Ny, int Nz, float h, float dt, float c,
          float U_PML, float D_PML, float F_PML, float B_PML, float R_PML, float L_PML,
          float SSP_w, bool advance){

    const int   d    = PML_CELLS;
    const int   ng   = PARSE3D_GHOST;   // absorbing band lives on UPDATED cells, i in [ng, ng+d)
    const int   Cijk = flat3(i, j, k, Nx, Ny);
    const float R0 = 1e-7, lnR0 = logf(1.0/R0), m = 2.0;
    const float kappa_max = 5.0; // knob: [1,15] -> causes waves to be 'squared' fixing grazing incidence
    const float alpha_max = M_PI * 20.0; // knob: >0 = CFS shift [rad/s ~ pi*f_min]; 0 = plain PML

    // depth anchored at the interior edge ng (not the grid edge): peak sigma sits on the
    // outermost UPDATED cell (i=ng), 0 at the inner edge (i=ng+d). Ghost cells excluded.
    const float xm = fminf(1.f, fmaxf(0.f, ((float)(ng + d) - (float)i) / (float)d));
    const float xp = fminf(1.f, fmaxf(0.f, ((float)i - ((float)(Nx-1-ng) - (float)d)) / (float)d));
    const float ym = fminf(1.f, fmaxf(0.f, ((float)(ng + d) - (float)j) / (float)d));
    const float yp = fminf(1.f, fmaxf(0.f, ((float)j - ((float)(Ny-1-ng) - (float)d)) / (float)d));
    const float zm = fminf(1.f, fmaxf(0.f, ((float)(ng + d) - (float)k) / (float)d));
    const float zp = fminf(1.f, fmaxf(0.f, ((float)k - ((float)(Nz-1-ng) - (float)d)) / (float)d));

    const float pref = 0.5 * c * (m + 1.0) * lnR0 / ((float)d * h);
    const float sig_x = L_PML*pref*powf(xm,m) + R_PML*pref*powf(xp,m);
    const float sig_y = B_PML*pref*powf(ym,m) + F_PML*pref*powf(yp,m);
    const float sig_z = D_PML*pref*powf(zm,m) + U_PML*pref*powf(zp,m);

    const float dep_x = L_PML*xm + R_PML*xp, dep_y = B_PML*ym + F_PML*yp, dep_z = D_PML*zm + U_PML*zp;
    const float kap_x = 1.f + (kappa_max-1.f)*powf(dep_x,m);
    const float kap_y = 1.f + (kappa_max-1.f)*powf(dep_y,m);
    const float kap_z = 1.f + (kappa_max-1.f)*powf(dep_z,m);
    // alpha graded alpha_max (inner) -> alpha_min (outer), NEVER to 0: alpha=0 at the wall
    // reseeds the plain-PML late-time instability exactly where evanescent energy piles up.
    const float alpha_min = 0.1f * alpha_max;   // CFS floor, keeps stabilization everywhere
    const float alp_x = alpha_min + (alpha_max - alpha_min)*(1.f - dep_x);
    const float alp_y = alpha_min + (alpha_max - alpha_min)*(1.f - dep_y);
    const float alp_z = alpha_min + (alpha_max - alpha_min)*(1.f - dep_z);

    const bool bx = (L_PML > 0.f && i >= ng && i < ng + d) || (R_PML > 0.f && i >= Nx-ng-d && i <= Nx-1-ng);
    const bool by = (B_PML > 0.f && j >= ng && j < ng + d) || (F_PML > 0.f && j >= Ny-ng-d && j <= Ny-1-ng);
    const bool bz = (D_PML > 0.f && k >= ng && k < ng + d) || (U_PML > 0.f && k >= Nz-ng-d && k <= Nz-1-ng);

    const float wdt = SSP_w * dt;   // correction weight = flux weight in this RK stage
    if (bx){
        const int ix = (i < ng + d) ? (i - ng) : (i - (Nx-ng-d) + d);   // -> [0, 2d)
        const int X  = ix + (2*d)*(j + Ny*k);
        p[Cijk] += wdt*cpml_corr(Lpx, kap_x, psi_px[X]);   // read frozen psi
        u[Cijk] += wdt*cpml_corr(Lux, kap_x, psi_ux[X]);
        if (advance){                                       // then advance once per step
            cpml_adv(Lpx, kap_x, sig_x, alp_x, dt, psi_px[X]);
            cpml_adv(Lux, kap_x, sig_x, alp_x, dt, psi_ux[X]);
        }
    }
    if (by){
        const int jy = (j < ng + d) ? (j - ng) : (j - (Ny-ng-d) + d);
        const int Y  = i + Nx*(jy + (2*d)*k);
        p[Cijk] += wdt*cpml_corr(Lpy, kap_y, psi_py[Y]);
        v[Cijk] += wdt*cpml_corr(Lvy, kap_y, psi_vy[Y]);
        if (advance){
            cpml_adv(Lpy, kap_y, sig_y, alp_y, dt, psi_py[Y]);
            cpml_adv(Lvy, kap_y, sig_y, alp_y, dt, psi_vy[Y]);
        }
    }
    if (bz){
        const int kz = (k < ng + d) ? (k - ng) : (k - (Nz-ng-d) + d);
        const int Z  = i + Nx*(j + Ny*kz);
        p[Cijk] += wdt*cpml_corr(Lpz, kap_z, psi_pz[Z]);
        w[Cijk] += wdt*cpml_corr(Lwz, kap_z, psi_wz[Z]);
        if (advance){
            cpml_adv(Lpz, kap_z, sig_z, alp_z, dt, psi_pz[Z]);
            cpml_adv(Lwz, kap_z, sig_z, alp_z, dt, psi_wz[Z]);
        }
    }
}


// --- WENO5 functions ---

// reconstruct in x, for k = 3
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_{i-1/2} instead of V_{i+1/2})
__device__ float construct_Vi_12(float *__restrict__ u, int i, int j, int k, int Nx, int Ny, float *__restrict__ c_rl, int r, int idx){
	float retval = 0.0;
	for (int l = 0; l <= 2; l++){
		//if (r+idx > 3) printf("[!] Violation");
		retval += c_rl[flat(r+idx,l,12)]*u[flat3(i-r+l,j,k,Nx,Ny)];
	}
	return retval;
}

// reconstruct in y, for k = 3
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_{j-1/2} instead of V_{j+1/2})
__device__ float construct_Vj_12(float *__restrict__ u, int i, int j, int k, int Nx, int Ny, float *__restrict__ c_rl, int r, int idx){
	float retval = 0.0;
	for (int l = 0; l <= 2; l++){
		retval += c_rl[flat(r+idx,l,12)]*u[flat3(i,j-r+l,k,Nx,Ny)];
	}
	return retval;
}

// reconstruct in z, for k = 3
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_{k-1/2} instead of V_{k+1/2})
__device__ float construct_Vk_12(float *__restrict__ u, int i, int j, int k, int Nx, int Ny, float *__restrict__ c_rl, int r, int idx){
	float retval = 0.0;
	for (int l = 0; l <= 2; l++){
		retval += c_rl[flat(r+idx,l,12)]*u[flat3(i,j,k-r+l,Nx,Ny)];
	}
	return retval;
}

// compute smoothness indicators in x
__device__ float compute_beta_x(float *__restrict__ u, int r, int i, int j, int k, int Nx, int Ny, float *__restrict__ beta_rl){
	float b1, b2;
	
	b1 = u[flat3(i-r,j,k,Nx,Ny)] - 2*u[flat3(i-r+1,j,k,Nx,Ny)] + u[flat3(i-r+2,j,k,Nx,Ny)];
	
	b2 = beta_rl[flat(r,0,9)]*u[flat3(i-r,j,k,Nx,Ny)] + beta_rl[flat(r,1,9)]*u[flat3(i-r+1,j,k,Nx,Ny)]
			+ beta_rl[flat(r,2,9)]*u[flat3(i-r+2,j,k,Nx,Ny)];
	
	return (13.0*b1*b1/12.0) + (0.25*b2*b2);
}

// compute smoothness indicators in y
__device__ float compute_beta_y(float *__restrict__ u, int r, int i, int j, int k, int Nx, int Ny, float *__restrict__ beta_rl){
	float b1, b2;
	
	b1 = u[flat3(i,j-r,k,Nx,Ny)] - 2*u[flat3(i,j-r+1,k,Nx,Ny)] + u[flat3(i,j-r+2,k,Nx,Ny)];
	
	b2 = beta_rl[flat(r,0,9)]*u[flat3(i,j-r,k,Nx,Ny)] + beta_rl[flat(r,1,9)]*u[flat3(i,j-r+1,k,Nx,Ny)]
			+ beta_rl[flat(r,2,9)]*u[flat3(i,j-r+2,k,Nx,Ny)];
	
	return (13.0*b1*b1/12.0) + (0.25*b2*b2);
}

// compute smoothness indicators in z
__device__ float compute_beta_z(float *__restrict__ u, int r, int i, int j, int k, int Nx, int Ny, float *__restrict__ beta_rl){
	float b1, b2;
	
	b1 = u[flat3(i,j,k-r,Nx,Ny)] - 2*u[flat3(i,j,k-r+1,Nx,Ny)] + u[flat3(i,j,k-r+2,Nx,Ny)];
	
	b2 = beta_rl[flat(r,0,9)]*u[flat3(i,j,k-r,Nx,Ny)] + beta_rl[flat(r,1,9)]*u[flat3(i,j,k-r+1,Nx,Ny)]
			+ beta_rl[flat(r,2,9)]*u[flat3(i,j,k-r+2,Nx,Ny)];
	
	return (13.0*b1*b1/12.0) + (0.25*b2*b2);
}
/*
// compute alpha[r]
// if S = 0, compute alpha, if 1, compute alpha~
__device__ float compute_alpha(float *__restrict__ dd, float *__restrict__ beta, float eps, int r, int SS){
	float aux = beta[r] + eps;
	int rr = SS*3 - SS*1 - r;
	return dd[(int)abs(rr)]/(aux*aux);
}
*/

// WENO-Z weights: Borges, Carmona, Costa & Don (2008), eqs. (25) and (28)
// tau5 = |beta_0 - beta_2| is symmetric, so the SS mirroring does not affect it
__device__ float compute_alpha(float *__restrict__ dd, float *__restrict__ beta, float eps, int r, int SS){
	float tau5  = fabsf(beta[0] - beta[2]);
	float ratio = tau5 / (beta[r] + eps); // float32 overflow guard
	int rr = SS*3 - SS*1 - r;
	return dd[(int)abs(rr)] * (1.0f + ratio); // q = 1 (less dissipative)
}

/*
// UP5 (5th linear Upwind scheme - highly dissipative)
__device__ float compute_alpha(float *__restrict__ dd, int r, int SS){
	int rr = SS*3 - SS*1 - r;
	return dd[(int)abs(rr)];
}
*/

// compute omega[r] - w here is distinct from velocity in z!
__device__ float compute_w(float *alpha, int r){
	float sum;
	sum = alpha[0] + alpha[1] + alpha[2];
	return alpha[r]/sum;
}

// WENO algorithm in x
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_i- instead of V_i+)
__device__ float WENO_x(float *__restrict__ V, float *__restrict__ dd, float *__restrict__ c_rl, float *__restrict__ beta_rl, int i, int j, int k, int Nx, int Ny, int idx, float eps){
	
	// 1) Contruct point values V_{i+1/2}, V_{i-1/2}
	float Vi_12[3];
	for (int r = 0; r < 3; r++){
		Vi_12[r] = construct_Vi_12(V, i, j, k, Nx, Ny, c_rl, r, idx);
	}
	
	// 2) Compute coefficients d_r: computed on array dd
	
	// 3) Compute smoothness indicators beta[r] (comment for UP5)
	//WENO-5/Z
	float beta[3];
	for (int r = 0; r < 3; r++){
		beta[r] = compute_beta_x(V, r, i, j, k, Nx, Ny, beta_rl);
	}
	
	// 4) Define omega[r]
	float alpha[3], w[3];
	for (int r = 0; r < 3; r++){
		//WENO-5/Z
		alpha[r] = compute_alpha(dd, beta, eps, r, abs(idx-1));
		// UP5
		//alpha[r] = compute_alpha(dd, r, abs(idx-1));
	}
	for (int r = 0; r < 3; r++){
		w[r] = compute_w(alpha, r);
	}
	
	// 5) Compute V_i+ / V_i-, depending on the value of idx
	float V_i = 0.0;
	for (int r = 0; r < 3; r++){
		V_i += w[r]*Vi_12[r];
	}
	
	return V_i;
}

// WENO algorithm in y
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_j- instead of V_j+)
__device__ float WENO_y(float *__restrict__ V, float *__restrict__ dd, float *__restrict__ c_rl, float *__restrict__ beta_rl, int i, int j, int k, int Nx, int Ny, int idx, float eps){
	
	// 1) Contruct point values V_{i+1/2}, V_{i-1/2}
	float Vj_12[3];
	for (int r = 0; r < 3; r++){
		Vj_12[r] = construct_Vj_12(V, i, j, k, Nx, Ny, c_rl, r, idx);
	}
	
	// 2) Compute coefficients d_r: computed on array dd
	
	// 3) Compute smoothness indicators beta[r] (comment for UP5)
	//WENO-5/Z
	float beta[3];
	for (int r = 0; r < 3; r++){
		beta[r] = compute_beta_y(V, r, i, j, k, Nx, Ny, beta_rl);
	}
	
	// 4) Define omega[r]
	float alpha[3], w[3];
	for (int r = 0; r < 3; r++){
		alpha[r] = compute_alpha(dd, beta, eps, r, abs(idx-1));
		// UP5
		//alpha[r] = compute_alpha(dd, r, abs(idx-1));
	}
	for (int r = 0; r < 3; r++){
		w[r] = compute_w(alpha, r);
	}
	
	// 5) Compute V_j+ / V_j-, depending on the value of idx
	float V_j = 0.0;
	for (int r = 0; r < 3; r++){
		V_j += w[r]*Vj_12[r];
	}
	
	return V_j;
}

// WENO algorithm in z
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_k- instead of V_k+)
__device__ float WENO_z(float *__restrict__ V, float *__restrict__ dd, float *__restrict__ c_rl, float *__restrict__ beta_rl, int i, int j, int k, int Nx, int Ny, int idx, float eps){
	
	// 1) Contruct point values V_{k+1/2}, V_{k-1/2}
	float Vk_12[3];
	for (int r = 0; r < 3; r++){
		Vk_12[r] = construct_Vk_12(V, i, j, k, Nx, Ny, c_rl, r, idx);
	}
	
	// 2) Compute coefficients d_r: computed on array dd
	
	// 3) Compute smoothness indicators beta[r] (comment for UP5)
	//WENO-5/Z
	float beta[3];
	for (int r = 0; r < 3; r++){
		beta[r] = compute_beta_z(V, r, i, j, k, Nx, Ny, beta_rl);
	}
	
	// 4) Define omega[r]
	float alpha[3], w[3]; // != velocity in z!
	for (int r = 0; r < 3; r++){
		alpha[r] = compute_alpha(dd, beta, eps, r, abs(idx-1));
		// UP5
		//alpha[r] = compute_alpha(dd, r, abs(idx-1));
	}
	for (int r = 0; r < 3; r++){
		w[r] = compute_w(alpha, r);
	}
	
	// 5) Compute V_k+ / V_k-, depending on the value of idx
	float V_k = 0.0;
	for (int r = 0; r < 3; r++){
		V_k += w[r]*Vk_12[r];
	}
	
	return V_k;
}

// Compute final 4 V's for pluging WENO reconstructions in x flux
__device__ void compute_Vs_x(float *Q_V, float *__restrict__ V, float *__restrict__ dd, float *__restrict__ c_rl, float *__restrict__ beta_rl, int i, int j, int k, int Nx, int Ny, float eps){
	Q_V[0] = WENO_x(V, dd, c_rl, beta_rl, i, j, k, Nx, Ny, 1, eps);
	Q_V[1] = WENO_x(V, dd, c_rl, beta_rl, i+1, j, k, Nx, Ny, 0, eps);
	Q_V[2] = WENO_x(V, dd, c_rl, beta_rl, i-1, j, k, Nx, Ny, 1, eps);
	Q_V[3] = WENO_x(V, dd, c_rl, beta_rl, i, j, k, Nx, Ny, 0, eps);
}

// Compute final 4 V's for pluging WENO reconstructions in y flux
__device__ void compute_Vs_y(float *Q_V, float *__restrict__ V, float *__restrict__ dd, float *__restrict__ c_rl, float *__restrict__ beta_rl, int i, int j, int k, int Nx, int Ny, float eps){
	Q_V[0] = WENO_y(V, dd, c_rl, beta_rl, i, j, k, Nx, Ny, 1, eps);
	Q_V[1] = WENO_y(V, dd, c_rl, beta_rl, i, j+1, k, Nx, Ny, 0, eps);
	Q_V[2] = WENO_y(V, dd, c_rl, beta_rl, i, j-1, k, Nx, Ny, 1, eps);
	Q_V[3] = WENO_y(V, dd, c_rl, beta_rl, i, j, k, Nx, Ny, 0, eps);
}

// Compute final 4 V's for pluging WENO reconstructions in z flux
__device__ void compute_Vs_z(float *Q_V, float *__restrict__ V, float *__restrict__ dd, float *__restrict__ c_rl, float *__restrict__ beta_rl, int i, int j, int k, int Nx, int Ny, float eps){
	Q_V[0] = WENO_z(V, dd, c_rl, beta_rl, i, j, k, Nx, Ny, 1, eps);
	Q_V[1] = WENO_z(V, dd, c_rl, beta_rl, i, j, k+1, Nx, Ny, 0, eps);
	Q_V[2] = WENO_z(V, dd, c_rl, beta_rl, i, j, k-1, Nx, Ny, 1, eps);
	Q_V[3] = WENO_z(V, dd, c_rl, beta_rl, i, j, k, Nx, Ny, 0, eps);
}

__global__ void kernel_test_BCs_no_parsing(int Nx, int Ny, int Nz, float *p, float *__restrict__ u, float *__restrict__ v, float *__restrict__ w){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 3;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
	int k = blockIdx.z*blockDim.z + threadIdx.z + 3;
   
   	float R = 0.95;
   
    if ((i < Nx-3) && (j < Ny-3) && (k < Nz-3)){
    	
    	// wall at x-
    	p[flat3(3,j,k,Nx,Ny)] = R * p[flat3(4,j,k,Nx,Ny)];
    	u[flat3(3,j,k,Nx,Ny)] = -R * u[flat3(4,j,k,Nx,Ny)];
    	
    	// wall at x+
    	p[flat3(Nx-4,j,k,Nx,Ny)] = R * p[flat3(Nx-5,j,k,Nx,Ny)];
    	u[flat3(Nx-4,j,k,Nx,Ny)] = -R * u[flat3(Nx-5,j,k,Nx,Ny)];
    	
    	// wall at y-
    	p[flat3(i,3,k,Nx,Ny)] = R * p[flat3(i,4,k,Nx,Ny)];
    	v[flat3(i,3,k,Nx,Ny)] = -R * v[flat3(i,4,k,Nx,Ny)];
    	
    	// wall at y+
    	p[flat3(i,Ny-4,k,Nx,Ny)] = R * p[flat3(i,Ny-5,k,Nx,Ny)];
    	v[flat3(i,Ny-4,k,Nx,Ny)] = -R * v[flat3(i,Ny-5,k,Nx,Ny)];
    	
    	// wall at z-
    	p[flat3(i,j,3,Nx,Ny)] = R * p[flat3(i,j,4,Nx,Ny)];
    	w[flat3(i,j,3,Nx,Ny)] = -R * w[flat3(i,j,4,Nx,Ny)];
    	
    	// wall at z+
    	p[flat3(i,j,Nz-4,Nx,Ny)] = R * p[flat3(i,j,Nz-5,Nx,Ny)];
    	w[flat3(i,j,Nz-4,Nx,Ny)] = -R * w[flat3(i,j,Nz-5,Nx,Ny)];
    }
}

__global__ void kernel_inject_src_SSP(float *p, float *u_s, int *sidx, float *sval, float SSP3w, float c, float h, float dt, float rho0, float A_s, int Nx, int Ny, int N_src, int nn, int src_opt, float *src_freq){

	int s = blockIdx.x*blockDim.x + threadIdx.x; // sources
	if (s >= N_src) return;
	float s_amp = (src_opt == 2)
             ? sinf(2.0f*M_PI*src_freq[s]*nn*(0.5f*dt)) // sinusoid: per-source freq, t = nn*dt_half
             : u_s[nn];                                 // PCS/Ricker: shared precomputed waveform
	p[sidx[s]] += SSP3w * dt * c*c * sval[s] * s_amp * rho0 * A_s / (h*h*h);
}

// kernel to show boundary artifacts, R is a global reflection coefficient
__global__ void kernel_apply_BC_vox(float *q, float *S, int Nx, int Ny, int Nz, float R){
    int i = blockIdx.x*blockDim.x + threadIdx.x + 3;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
	int k = blockIdx.y*blockDim.y + threadIdx.y + 3;
    if ((i < Nx-3) && (j < Ny-3) && (k < Nz-3)){
    	
		int Cijk = flat3(i,j,k,Nx,Ny);
		
		float nodesum = 0.0, sum = 0.0, eps = 1e-7, F;
		
		//R *= -1.0f; // phase inversion, if desired
		
    	// Compute contributions from neighbors for each direction, ensuring parallel paradigm
    	
    	nodesum += R * S[flat3(i+1, j, k, Nx, Ny)] * q[flat3(i+1, j, k, Nx, Ny)];
    	nodesum += R * S[flat3(i+1, j, k, Nx, Ny)] * q[flat3(i+1, j, k, Nx, Ny)];
    	nodesum += R * S[flat3(i, j+1, k, Nx, Ny)] * q[flat3(i, j+1, k, Nx, Ny)];
    	nodesum += R * S[flat3(i, j-1, k, Nx, Ny)] * q[flat3(i, j-1, k, Nx, Ny)];
    	nodesum += R * S[flat3(i, j, k+1, Nx, Ny)] * q[flat3(i, j, k+1, Nx, Ny)];
    	nodesum += R * S[flat3(i, j, k-1, Nx, Ny)] * q[flat3(i, j, k-1, Nx, Ny)];
    	
    	// for inner nodes, nodesum is already 0.0   
	
    	// Sum of active neighbors (based on S)
    	sum = S[flat3(i+1, j, k, Nx, Ny)]+ S[flat3(i, j+1, k, Nx, Ny)] + S[flat3(i, j, k+1, Nx, Ny)] +
          	  S[flat3(i-1, j, k, Nx, Ny)]+ S[flat3(i, j-1, k, Nx, Ny)] + S[flat3(i, j, k-1, Nx, Ny)];
	
    	// Determine node type
    	F = 1.0 - S[Cijk]; // Boundary node factor (If F = 1.0, it is a boundary)
	
    	// Apply boundary condition
    	q[Cijk] = S[Cijk]*q[Cijk] + F*nodesum/(sum + eps);
    }
}

// WENO5 SSP-RK3 step 0
__global__ void kernel_WENO_stage0(int Nx, int Ny, int Nz, float *__restrict__ p0, float *__restrict__ u0, float *__restrict__ v0, float *__restrict__ w0, float *__restrict__ p1, float *__restrict__ u1, float *__restrict__ v1, float *__restrict__ w1, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *beta_rl, float *__restrict__ S, float U_PML, float D_PML, float R_PML, float L_PML, float F_PML, float B_PML, float *px, float *py, float *pz, int pml_mode, float *psi_px, float *psi_py, float *psi_pz, float *psi_ux, float *psi_vy, float *psi_wz){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 3;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
	int k = blockIdx.z*blockDim.z + threadIdx.z + 3;
    if ((i < Nx-3) && (j < Ny-3) && (k < Nz-3)){
    	
    	// central index
		int Cijk = flat3(i,j,k,Nx,Ny);
		
		// restrict
		if (S[Cijk] < h) return;
    	
    	// flux values
		float F_pP, F_pM, F_uP, F_uM;
		float G_pP, G_pM, G_vP, G_vM;
		float H_pP, H_pM, H_wP, H_wM;		
		
		// final reconstructed WENO values V+, V-
		float p_V[4], u_V[4], v_V[4], w_V[4];
		
		// (1)
		
		// compute WENO values V_i+, V_i-
		compute_Vs_x(p_V, p0, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_x(u_V, u0, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux F_p+
		F_pP = F_p_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c);
		// compute flux F_p-
		F_pM = F_p_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c);
		// compute flux F_u+
		F_uP = F_u_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c);
		// compute flux F_u-
		F_uM = F_u_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c);
		
		// compute WENO values V_j+, V_j- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_y(p_V, p0, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_y(v_V, v0, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux G_p+
		G_pP = G_p_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c);
		// compute flux G_p-
		G_pM = G_p_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c);
		// compute flux G_v+
		G_vP = G_v_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c);
		// compute flux G_v-
		G_vM = G_v_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c);
		
		// compute WENO values V_k+, V_k- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_z(p_V, p0, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_z(w_V, w0, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux H_p+
		H_pP = H_p_Godunov(p_V[0], p_V[1], w_V[0], w_V[1], rho, c);
		// compute flux H_p-
		H_pM = H_p_Godunov(p_V[2], p_V[3], w_V[2], w_V[3], rho, c);
		// compute flux H_w+
		H_wP = H_w_Godunov(p_V[0], p_V[1], w_V[0], w_V[1], rho, c);
		// compute flux H_w-
		H_wM = H_w_Godunov(p_V[2], p_V[3], w_V[2], w_V[3], rho, c);
		
		// stage updates
		u1[Cijk] = u0[Cijk] + dt*(-F_uP + F_uM)/h;
		v1[Cijk] = v0[Cijk] + dt*(-G_vP + G_vM)/h;
		w1[Cijk] = w0[Cijk] + dt*(-H_wP + H_wM)/h;
		p1[Cijk] = p0[Cijk] + dt*(-F_pP + F_pM - G_pP + G_pM - H_pP + H_pM)/h;
	
		// other BCs
		// split PML (mode 0) runs once per step in step2. cPML (mode 1) is RK-consistent:
		// stretch the field this stage (weight 1.0), do NOT advance psi yet.
		if (pml_mode == 1){
			cPML(p1, u1, v1, w1, psi_px, psi_py, psi_pz, psi_ux, psi_vy, psi_wz,
			     (-F_pP + F_pM)/h, (-G_pP + G_pM)/h, (-H_pP + H_pM)/h,
			     (-F_uP + F_uM)/h, (-G_vP + G_vM)/h, (-H_wP + H_wM)/h,
			     i, j, k, Nx, Ny, Nz, h, dt, c, U_PML, D_PML, F_PML, B_PML, R_PML, L_PML, 1.0f, false);
		}
	}
}

// WENO5 SSP-RK3 step 1
__global__ void kernel_WENO_stage1(int Nx, int Ny, int Nz, float *__restrict__ p0, float *__restrict__ u0, float *__restrict__ v0, float *__restrict__ w0, float *__restrict__ p1, float *__restrict__ u1, float *__restrict__ v1, float *__restrict__ w1, float *__restrict__ p2, float *__restrict__ u2, float *__restrict__ v2, float *__restrict__ w2, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *beta_rl, float *__restrict__ S, float U_PML, float D_PML, float R_PML, float L_PML, float F_PML, float B_PML, float *px, float *py, float *pz, int pml_mode, float *psi_px, float *psi_py, float *psi_pz, float *psi_ux, float *psi_vy, float *psi_wz){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 3;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
	int k = blockIdx.z*blockDim.z + threadIdx.z + 3;
    if ((i < Nx-3) && (j < Ny-3) && (k < Nz-3)){
    	
    	// central index
		int Cijk = flat3(i,j,k,Nx,Ny);
		
		// restrict
		if (S[Cijk] < h) return;
    	
    	// flux values
		float F_pP, F_pM, F_uP, F_uM;
		float G_pP, G_pM, G_vP, G_vM;
		float H_pP, H_pM, H_wP, H_wM;		
		
		// final reconstructed WENO values V+, V-
		float p_V[4], u_V[4], v_V[4], w_V[4];
		
		// (2)
		
		// compute WENO values V_i+, V_i-
		compute_Vs_x(p_V, p1, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_x(u_V, u1, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux F_p+
		F_pP = F_p_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c);
		// compute flux F_p-
		F_pM = F_p_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c);
		// compute flux F_u+
		F_uP = F_u_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c);
		// compute flux F_u-
		F_uM = F_u_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c);
		
		// compute WENO values V_j+, V_j- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_y(p_V, p1, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_y(v_V, v1, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux G_p+
		G_pP = G_p_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c);
		// compute flux G_p-
		G_pM = G_p_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c);
		// compute flux G_v+
		G_vP = G_v_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c);
		// compute flux G_v-
		G_vM = G_v_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c);
		
		// compute WENO values V_k+, V_k- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_z(p_V, p1, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_z(w_V, w1, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux H_p+
		H_pP = H_p_Godunov(p_V[0], p_V[1], w_V[0], w_V[1], rho, c);
		// compute flux H_p-
		H_pM = H_p_Godunov(p_V[2], p_V[3], w_V[2], w_V[3], rho, c);
		// compute flux H_w+
		H_wP = H_w_Godunov(p_V[0], p_V[1], w_V[0], w_V[1], rho, c);
		// compute flux H_w-
		H_wM = H_w_Godunov(p_V[2], p_V[3], w_V[2], w_V[3], rho, c);
		
		// stage updates
		u2[Cijk] = 0.75f*u0[Cijk] + 0.25f*(u1[Cijk] + dt*((-F_uP + F_uM)/h));
		v2[Cijk] = 0.75f*v0[Cijk] + 0.25f*(v1[Cijk] + dt*((-G_vP + G_vM)/h));
		w2[Cijk] = 0.75f*w0[Cijk] + 0.25f*(w1[Cijk] + dt*((-H_wP + H_wM)/h));
		p2[Cijk] = 0.75f*p0[Cijk] + 0.25f*(p1[Cijk] + dt*((-F_pP + F_pM - G_pP + G_pM - H_pP + H_pM)/h));
		
		// other BCs
		// cPML (mode 1): stretch the field this stage (weight 0.25), do NOT advance psi yet.
		if (pml_mode == 1){
			cPML(p2, u2, v2, w2, psi_px, psi_py, psi_pz, psi_ux, psi_vy, psi_wz,
			     (-F_pP + F_pM)/h, (-G_pP + G_pM)/h, (-H_pP + H_pM)/h,
			     (-F_uP + F_uM)/h, (-G_vP + G_vM)/h, (-H_wP + H_wM)/h,
			     i, j, k, Nx, Ny, Nz, h, dt, c, U_PML, D_PML, F_PML, B_PML, R_PML, L_PML, 0.25f, false);
		}
	}
}

// WENO5 SSP-RK3 step 2
__global__ void kernel_WENO_stage2(int Nx, int Ny, int Nz, float *__restrict__ p0, float *__restrict__ u0, float *__restrict__ v0, float *__restrict__ w0, float *__restrict__ p1, float *__restrict__ u1, float *__restrict__ v1, float *__restrict__ w1, float *__restrict__ p2, float *__restrict__ u2, float *__restrict__ v2, float *__restrict__ w2, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *beta_rl, float *__restrict__ S, float U_PML, float D_PML, float R_PML, float L_PML, float F_PML, float B_PML, float *px, float *py, float *pz, int pml_mode, float *psi_px, float *psi_py, float *psi_pz, float *psi_ux, float *psi_vy, float *psi_wz){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 3;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 3;
	int k = blockIdx.z*blockDim.z + threadIdx.z + 3;
    if ((i < Nx-3) && (j < Ny-3) && (k < Nz-3)){
    	
    	// central index
		int Cijk = flat3(i,j,k,Nx,Ny);
		
		// restrict
		if (S[Cijk] < h) return;
    	
    	// flux values
		float F_pP, F_pM, F_uP, F_uM;
		float G_pP, G_pM, G_vP, G_vM;
		float H_pP, H_pM, H_wP, H_wM;		
		
		// final reconstructed WENO values V+, V-
		float p_V[4], u_V[4], v_V[4], w_V[4];
		
		// (2)
		
		// compute WENO values V_i+, V_i-
		compute_Vs_x(p_V, p2, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_x(u_V, u2, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux F_p+
		F_pP = F_p_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c);
		// compute flux F_p-
		F_pM = F_p_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c);
		// compute flux F_u+
		F_uP = F_u_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c);
		// compute flux F_u-
		F_uM = F_u_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c);
		
		// compute WENO values V_j+, V_j- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_y(p_V, p2, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_y(v_V, v2, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux G_p+
		G_pP = G_p_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c);
		// compute flux G_p-
		G_pM = G_p_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c);
		// compute flux G_v+
		G_vP = G_v_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c);
		// compute flux G_v-
		G_vM = G_v_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c);
		
		// compute WENO values V_k+, V_k- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_z(p_V, p2, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		compute_Vs_z(w_V, w2, dd, c_rl, beta_rl, i, j, k, Nx, Ny, eps);
		
		// compute flux H_p+
		H_pP = H_p_Godunov(p_V[0], p_V[1], w_V[0], w_V[1], rho, c);
		// compute flux H_p-
		H_pM = H_p_Godunov(p_V[2], p_V[3], w_V[2], w_V[3], rho, c);
		// compute flux H_w+
		H_wP = H_w_Godunov(p_V[0], p_V[1], w_V[0], w_V[1], rho, c);
		// compute flux H_w-
		H_wM = H_w_Godunov(p_V[2], p_V[3], w_V[2], w_V[3], rho, c);
		
		// stage update
		// trick to avoid yet another array for each field U is to use U1 as final u here (outp), since not used
		u1[Cijk] = (u0[Cijk] + 2.0f*(u2[Cijk] + dt*((-F_uP + F_uM)/h)))/3.0f;
		v1[Cijk] = (v0[Cijk] + 2.0f*(v2[Cijk] + dt*((-G_vP + G_vM)/h)))/3.0f;
		w1[Cijk] = (w0[Cijk] + 2.0f*(w2[Cijk] + dt*((-H_wP + H_wM)/h)))/3.0f;
		p1[Cijk] = (p0[Cijk] + 2.0f*(p2[Cijk] + dt*((-F_pP + F_pM - G_pP + G_pM - H_pP + H_pM)/h)))/3.0f;
		
		// other BCs
		
		// PML: mode 0 = split sPML (Bérenger, once/step). mode 1 = RK-consistent cPML:
		// final stage (weight 2/3), and advance psi ONCE here (advance=true, full dt).
		if (pml_mode == 1){
			cPML(p1, u1, v1, w1, psi_px, psi_py, psi_pz, psi_ux, psi_vy, psi_wz,
			     (-F_pP + F_pM)/h, (-G_pP + G_pM)/h, (-H_pP + H_pM)/h,
			     (-F_uP + F_uM)/h, (-G_vP + G_vM)/h, (-H_wP + H_wM)/h,
			     i, j, k, Nx, Ny, Nz, h, dt, c, U_PML, D_PML, F_PML, B_PML, R_PML, L_PML, 2.0f/3.0f, true);
		}
	}
}

// Heuristics absorption of air for lower frequencies.
// Since the absorption of air is proportional to a*f^2,
// then the energy-weighted mean coefficient is
// 
// 		<m> = a * <f^2> =
// 		a * (1/f_cutoff) * int_0^{f_cutoff} f^2 df = 
// 		a * f_cutoff^2 / sqrt(3) ~ 0.58 * f_cutoff
// 	
// Must be appllied to p and u, otherwise the
// Riemann invariant W = p + rho c u is  not conserved.
// To apply to each component of u, just multiply
// them by the same factor as p (vector multiplication
// by a scalar).
__global__ void kernel_air(float *p, float *u, float *v, float *w, float f_cutoff){
}

// --------------------------------------- VOX file loading --------------------------------------------

int load_vox(const char *path, float **S_out, int *Nx_out, int *Ny_out, float *Lx, float *Ly, float h){
    FILE *fp = fopen(path, "r");
    if (!fp) { perror("open"); return 0; }

    int x, y, lastx=-1, lasty=-1;

    // Read through to get the last line
    while (fscanf(fp, "%d %d", &x, &y) == 2){
        lastx = x;
        lasty = y;
    }

    if (lastx < 0 || lasty < 0) {
        fclose(fp);
        fprintf(stderr, "Invalid or empty file\n");
        return 0;
    }

    int Nx = lastx + 1;
    int Ny = lasty + 1;

    // Allocate and set all to 1.0 (free space)
    float *S = (float*)malloc((size_t)Nx * Ny * sizeof(float));
    if (!S) { perror("malloc"); fclose(fp); return 0; }

    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i)
            S[flat(i, j, Nx)] = 1.0f;

    // Rewind and mark obstacles = 0.0
    rewind(fp);
    while (fscanf(fp, "%d %d", &x, &y) == 2) {
        if (x >= 0 && x < Nx && y >= 0 && y < Ny)
            S[flat(x, y, Nx)] = 0.0f;
    }

    fclose(fp);
    *S_out = S;
    *Nx_out = Nx;
    *Ny_out = Ny;
    *Lx = h*Nx;
    *Ly = h*Ny;
    return 1;
}

/* --------------------------- string check ---------------------------*/

bool ends_with_dotobj(const char *str) {
    size_t len = strlen(str);
    return (len >= 4 && strcmp(str + len - 4, ".obj") == 0);
}


char* replace_xyz_with_vox(const char* filename) {
    const char* dot = strrchr(filename, '.');
    size_t base_len = dot ? (size_t)(dot - filename) : strlen(filename);
    
    char* new_filename = (char*)malloc(base_len + 5); // ".vox" + null terminator
    if (!new_filename) return NULL;

    strncpy(new_filename, filename, base_len);
    new_filename[base_len] = '\0';
    strcat(new_filename, ".vox");

    return new_filename;
}

/* --------------------- progress bar ------------------------------*/

void print_progress(int n, int Nt, int bar_width) {
    // update every 10% (change 10 to whatever granularity desired)
    int step = Nt / 10;// if (step < 1) step = 1;
    int mask   = ((n % step) == 0) | (n == Nt);   // print only at 10% ticks and at the end
    int filled = (int)((long long)bar_width * n / Nt);
    static const char FILL[] =
        "===================================================================================================="; // 100 '='
    char buf[192];
    int nnn = snprintf(buf, sizeof buf, "\r[%-*.*s] %3d%%",
                       bar_width, filled, FILL,
                       (int)(100.0 * n / Nt + 0.5));
    printf("%.*s", mask * nnn, buf);  // branchless print
    fflush(stdout);
}

void print_progress_nobar(int n, int Nt) {
    // Print JUST the integer percentage, one per line, at each 5% tick (and at the very end),
    // so the GUI can parse "NN%" and drive its progress bar. No bar, no '\r' -> clean log lines.
    int step = Nt / 20; if (step < 1) step = 1;   // guard tiny Nt (avoid n % 0)
    if ((n % step) == 0 || n == Nt) {
        printf("%d%%\n", (int)(100.0 * n / Nt + 0.5));
        fflush(stdout);
    }
}
