#include <cmath>
#include <cstring>
#include "parsing3D.h"
#include "filterABC.h"

__host__ __device__ inline int flat3D(int i, int j, int k, int Nx, int Ny){
    return (k*Ny + j)*Nx + i;
}

// no Euler option, WENO3 only (better for 3D fields)

// Godunov and Roe fluxes (for WENO, no simple substitutions)

// inputs on p,u,v,w are the reconstructed values
// pP and pM denote, for example, p_plus and p_minus
__host__ __device__ float F_p_Godunov(float pM, float pP, float uM, float uP, float rho, float c, int j){
    return 0.5f*(rho*c*c*(uP + uM) - c*(pP - pM));
}

__host__ __device__ float F_u_Godunov(float pM, float pP, float uM, float uP, float rho, float c, int j){
    return 0.5f*((pP + pM)/rho - c*(uP - uM));
}

__host__ __device__ float G_p_Godunov(float pM, float pP, float vM, float vP, float rho, float c, int j){
    return 0.5f*(rho*c*c*(vP + vM) - c*(pP - pM));
}

__host__ __device__ float G_v_Godunov(float pM, float pP, float vM, float vP, float rho, float c, int j){
    return 0.5f*((pP + pM)/rho - c*(vP - vM));
}

__host__ __device__ float H_p_Godunov(float pM, float pP, float wM, float wP, float rho, float c, int j){
    return 0.5f*(rho*c*c*(wP + wM) - c*(pP - pM));
}

__host__ __device__ float H_w_Godunov(float pM, float pP, float wM, float wP, float rho, float c, int j){
    return 0.5f*((pP + pM)/rho - c*(wP - wM));
}

__host__ float SPL(float p){
	return 20.0*log10f(fabsf(p)/(0.00002));
}

// Total variations

// compute on regular variables
// Total Variation of variable Q
__host__ __device__ float TV(float *Q, int Nx, int Ny, float h){
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
	return h*retval;
}

// check if scheme is TVD
__host__ void is_TVD(float TV_0, float TV_1){
	if (TV_1 <= TV_0){
		printf("\n[OK] Scheme is numerically TVD\n\n");
	}
	else printf("\n[!] Scheme is not TVD\n\n");
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
float tricubic_interp(const float *f, int Nx, int Ny, float Dx, float Dy, float Dz, float h){
    // Convert physical position to grid coordinates
    float gx = Dx / h;
    float gy = Dy / h;
    float gz = Dz / h;

    int i = (int)floorf(gx);
    int j = (int)floorf(gy);
    int k = (int)floorf(gz);

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

    for (int zz = 0; zz < 4; ++zz)
    {
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

// artificial diffusion
__host__ __device__ float artfd(float *q, int i, int j, int Nx, float h, float mu){
	float aux_x, aux_y;
	
	aux_x = q[flat(i+1,j,Nx)] + q[flat(i-1,j,Nx)] - 2.0f*q[flat(i,j,Nx)];
	aux_y = q[flat(i,j+1,Nx)] + q[flat(i,j-1,Nx)] - 2.0f*q[flat(i,j,Nx)];
	
	return mu*(aux_x + aux_y)/(h*h);
}

// 4-point raised cosine distribution (as seen in Bilbao "Modelling impedance ... one-dimensional case", 2023)
// multiplied by a factor of 2 to give correct weighting in the kernel below (?)
__device__ float phi(float r){
	if (r <= 2.0f && r >= 0.0f){
		return 0.5f*(1.0f + cos(0.5f*M_PI*r));
	}
	return 0.0f;
}

// case above but correctly ranges only inside the fluid
// tol need to be 4*h (?)
__device__ float nu_phi(float r){
	if (r <= 4.0f && r >= 0.0f){
		return 0.5f*(1.0f + cos(0.25f*M_PI*r));
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
	if (s >= 2.0f) return 0.0f;
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

// non-implicit, solved with regular SSP-RK3, but coefficients alpha, beta and zeta are restricted in terms
// of time-step dt in the fitting script
__device__ float ADE_solver(int b_ID, uint8_t m_ID, float w_in, float dt, float *A, float *zeta, float *B, float *C, float *alpha, float *beta, float *R_inf, int K, int L, float a_SSP, float b_SSP, float c_SSP, float *phi_left, float *phi_middle, float *phi_right, float *psi_1_left, float *psi_1_middle, float *psi_1_right, float *psi_2_left, float *psi_2_middle, float *psi_2_right){
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
__global__ void kernel_apply_BC_immersed_sdf(float *u, float *v, float *w, float *p, int Nx, int Ny, int Nz, float h, float *S, uint8_t *oid, float rho, float c, int n, float dt, float *u_old, float *v_old, float *w_old, float *p_old, float tol, float *Rinf, float *Ak, float *zeta, float *Bl, float *Cl, float *alpha, float *beta, int K_m, int L_m, int *bmap, float a_SSP, float b_SSP, float c_SSP, float *phi_left, float *phi_middle, float *phi_right, float *psi_1_left, float *psi_1_middle, float *psi_1_right, float *psi_2_left, float *psi_2_middle, float *psi_2_right){

	int i = blockIdx.x*blockDim.x + threadIdx.x + 2;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 2;
	int k = blockIdx.z*blockDim.z + threadIdx.z + 2;
    if ((i < Nx-2) && (j < Ny-2) && (k < Nz-2)){
		
    	int Cijk = flat3(i,j,k,Nx,Ny);
    	
    	float s = S[Cijk]; // signed distance from cell center to boundary
    	
    	// only act on immersed cells (fluid-side band from your OID map), provided
    	// those values are only on positive s, which is the case here
    	// oid[] ranges from 1 to nMat, 255 elsewhere
		const uint8_t o_ID = oid[Cij]; // object's material ID
		if (o_ID == 0xFF) return;
		
		int b_ID = bmap[Cij];
		//if (b_ID < 0) return; // safety; should match oid test
		
		// gradients from S[] for detecting normals   
    	float gx = (S[flat3(i+1,j,k,Nx,Ny)] - S[flat3(i-1,j,k,Nx,Ny)]) * (0.5f / h);
    	float gy = (S[flat3(i,j+1,k,Nx,Ny)] - S[flat3(i,j-1,k,Nx,Ny)]) * (0.5f / h);
    	float gz = (S[flat3(i,j,k+1,Nx,Ny)] - S[flat3(i,j,k-1,Nx,Ny)]) * (0.5f / h);
    	float gn = sqrtf(gx*gx + gy*gy + gz*gz);
    	gn = fmaxf(gn, 1e-12f); //safety
	
    	float nx = gx/gn;
    	float ny = gy/gn;
    	float nz = gz/gn;
	
    	// cell center
    	float Cx = i * h;
    	float Cy = j * h;
    	float Cz = k * h
		
    	// footpoint on boundary: W = C - s * n_hat  (since S>0 fluid, S<0 solid) 
    	float Wx = Cx - s * nx;
    	float Wy = Cy - s * ny;
    	float Wz = Cz - s * nz;
	
    	// interpolation point slightly inside the fluid along n
    	//float l = s + h;
    	// avoid bicubic interpolation using zero-valued cells, so reflections are not injected with the WENO3
    	// stencil ==> l = (s + 2h)
    	float l = s + 2.0f * h; 
    	float Dx = Wx + l * nx;
    	float Dy = Wy + l * ny;
    	float Dz = Wz + l * nz;
		
    	float pD = tricubic_interp(p_old, Nx, Ny, Dx, Dy, Dz, h);
    	float uD = tricubic_interp(u_old, Nx, Ny, Dx, Dy, Dz, h);
    	float vD = tricubic_interp(v_old, Nx, Ny, Dx, Dy, Dz, h);
    	float wD = tricubic_interp(w_old, Nx, Ny, Dx, Dy, Dz, h);

        const float un =  uD*nx + vD*ny + wD*nz;
        // tangential vector (free-slip)
		const float utx = uD - un*nx;
		const float uty = vD - un*ny;
		const float utz = wD - un*nz;

        const float Z0 = rho * c;

        // ------------------- incoming characteristic from fluid LOCAL side (towards obstacle) -------
        const float w_in_local = pD - Z0 * un;
        
        int m_ID = o_ID - 1; // material ID
        
        const float w_out_local = ADE_solver(
    		b_ID, m_ID, w_in_local, dt,
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

        const float ut_b = ut; // still free-slip (or add tangential model)

        // back to Cartesian
        const float u_b = un_b*nx + utx;
		const float v_b = un_b*ny + uty;
		const float w_b = un_b*nz + utz;  	
    	
    	const float chi = chi_layered(s, h, tol);
		/*
		// immersed boundary smoothing
		// uses "new" values cause WENO3 was already called at this time-step
		u[Cijk] = (1.0f - chi) * u[Cijk] + chi * u_b;
    	v[Cijk] = (1.0f - chi) * v[Cijk] + chi * v_b;
    	w[Cijk] = (1.0f - chi) * w[Cijk] + chi * w_b;
    	p[Cijk] = (1.0f - chi) * p[Cijk] + chi * p_b;
    	*/
    	// immersed boundary smoothing / blending (Brinkman penalization): tuned to different resolutions
    	float kappa = 0.5f; // once tuned for a test case, no more need to change (for a given tol and chi)
    	float eta = kappa * h / c;
    	float lam = 1.0f - expf(-chi * dt / eta); // dimensionless
		lam = fminf(lam, 1.0f); // clamp for explicit stability + no overshoot
		
		u[Cij] = (1.0f - lam) * u[Cij] + lam * u_b;
    	v[Cij] = (1.0f - lam) * v[Cij] + lam * v_b;
    	p[Cij] = (1.0f - lam) * p[Cij] + lam * p_b;
    	
    	    
    }
}

// sPML Layer - "simple" PML according to:
// 
// https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/AcousticsTimeDomain.html
// 
// 
__host__ __device__ void sPML(float *q, int i, int j, k, int Nx, int Ny, int Nz, float h, float N_PML, float S_PML, float E_PML, float W_PML, float F_PML, float B_PML){
    float s_max = 0.018f; // Maximum absorption factor
    int max_idx = (int)(0.15f*min(Nx, min(Ny, Nz))); // halfwidth
    float half   = (float)max_idx;

	// ramps in [0,1] (to exclude a side, multiply by 0.0f)
	float rx = E_PML*fmaxf(0.0f, fminf((i - (Nx - half)) / half, 1.0f)); // right boundary ramp
	float lx = W_PML*fmaxf(0.0f, fminf((half - i) / half, 1.0f)); // left boundary ramp
	float uy = N_PML*fmaxf(0.0f, fminf((j - (Ny - half)) / half, 1.0f)); // upper boundary ramp
	float ly = S_PML*fmaxf(0.0f, fminf((half - j) / half, 1.0f)); // lower boundary ramp
	float fz = F_PML*fmaxf(0.0f, fminf((k - (Nz - half)) / half, 1.0f)); // front boundary ramp
	float bz = B_PML*fmaxf(0.0f, fminf((half - k) / half, 1.0f)); // back boundary map
	
	// cubic (faster than powf)
	float rx3 = rx*rx*rx, lx3 = lx*lx*lx, ly3 = ly*ly*ly, uy3 = uy*uy*uy, fz3 = fz*fz*fz, sz3 = bz*bz*bz;
	
	// per-axis combined strengths (L/R, U/D and F/B are mutually exclusive)
	float sx = s_max * (rx3 + lx3);
	float sy = s_max * (uy3 + ly3);
	float sz = s_max * (fz3 + bz3);
	
	// exact corner composition
	//float s = 1.0f - (1.0f - sx) * (1.0f - sy);
	float s = 1.0f - (1.0f - sx) * (1.0f - sy) * (1.0f - sz);
	
	// (optional) keep strictly < 1.0 to avoid killing the value numerically
	s = fminf(s, 0.999f);
	
	// apply once
	int Cijk = flatk(i, j, k, Nx, Ny);
	q[Cijk] -= s * q[Cijk];
    
}

// --- WENO3 functions ---

// reconstruct in x, for k = 2
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_{i-1/2} instead of V_{i+1/2})
__device__ float construct_Vi_12(float *u, int i, int j, int Nx, float *c_rl, int r, int idx){
	float retval = 0.0;
	for (int l = 0; l <= 1; l++){
		retval += c_rl[flat(r+idx,l,3)]*u[flat(i-r+l,j,Nx)];
	}
	return retval;
}

// reconstruct in y, for k = 2
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_{j-1/2} instead of V_{j+1/2})
__device__ float construct_Vj_12(float *u, int i, int j, int Nx, float *c_rl, int r, int idx){
	float retval = 0.0;
	for (int l = 0; l <= 1; l++){
		retval += c_rl[flat(r+idx,l,3)]*u[flat(i,j-r+l,Nx)];
	}
	return retval;
}

// compute smoothness indicators in x
__device__ float compute_beta_x(float *u, int r, int i, int j, int Nx){
	float b;
	
	b = u[flat(i-r+1,j,Nx)] - u[flat(i-r,j,Nx)];
	
	return b*b;
}

// compute smoothness indicators in y
__device__ float compute_beta_y(float *u, int r, int i, int j, int Nx){
	float b;
	
	b = u[flat(i,j-r+1,Nx)] - u[flat(i,j-r,Nx)];
	
	return b*b;
}

// compute alpha[r]
// if auxiliar S = 0, compute alpha, if 1, compute alpha~
__device__ float compute_alpha(float *dd, float *beta, float eps, int r, int S){
	//if (beta[r] != 0.0) printf("beta = %.17f\n", beta[r]);
	float aux = beta[r] + eps;
	int rr = S - r;
	return dd[(int)abs(rr)]/(aux*aux);
}

// compute omega[r]
__device__ float compute_w(float *alpha, int r){
	float sum;
	sum = alpha[0] + alpha[1];
	return alpha[r]/sum;
}

// WENO algorithm in x
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_i- instead of V_i+)
__device__ float WENO_x(float *V, float *dd, float *c_rl, int i, int j, int Nx, int idx, float eps){
	
	// 1) Contruct point values V_{i+1/2}, V_{i-1/2}
	float Vi_12[2];
	for (int r = 0; r < 2; r++){
		Vi_12[r] = construct_Vi_12(V, i, j, Nx, c_rl, r, idx);
	}
	
	// 2) Compute coefficients d_r: computed on array dd
	
	// 3) Compute smoothness indicators beta[r]
	float beta[2];
	for (int r = 0; r < 2; r++){
		beta[r] = compute_beta_x(V, r, i, j, Nx);
	}
	
	// 4) Define omega[r]
	float alpha[2], w[2];
	for (int r = 0; r < 2; r++){
		alpha[r] = compute_alpha(dd, beta, eps, r, abs(idx-1));
	}
	for (int r = 0; r < 2; r++){
		w[r] = compute_w(alpha, r);
	}

	// 5) Compute V_i+ / V_i-, depending on the value of idx
	float V_i = 0.0;
	for (int r = 0; r < 2; r++){
		V_i += w[r]*Vi_12[r];
	}
	
	return V_i;
}

// WENO algorithm in y
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_j- instead of V_j+)
__device__ float WENO_y(float *V, float *dd, float *c_rl, int i, int j, int Nx, int idx, float eps){
	
	// 1) Contruct point values V_{i+1/2}, V_{i-1/2}
	float Vj_12[2];
	for (int r = 0; r < 2; r++){
		Vj_12[r] = construct_Vj_12(V, i, j, Nx, c_rl, r, idx);
	}
	
	// 2) Compute coefficients d_r: computed on array dd
	
	// 3) Compute smoothness indicators beta[r]
	float beta[2];
	for (int r = 0; r < 2; r++){
		beta[r] = compute_beta_y(V, r, i, j, Nx);
	}
	
	// 4) Define omega[r]
	float alpha[2], w[2];
	for (int r = 0; r < 2; r++){
		alpha[r] = compute_alpha(dd, beta, eps, r, abs(idx-1));
	}
	for (int r = 0; r < 2; r++){
		w[r] = compute_w(alpha, r);
	}
	
	// 5) Compute V_j+ / V_j-, depending on the value of idx
	float V_j = 0.0;
	for (int r = 0; r < 2; r++){
		V_j += w[r]*Vj_12[r];
	}
	
	return V_j;
}

// Compute final 4 V's for pluging WENO reconstructions in x flux
__device__ void compute_Vs_x(float *Q_V, float *V, float *dd, float *c_rl, int i, int j, int Nx, float eps){
	Q_V[0] = WENO_x(V, dd, c_rl, i, j, Nx, 1, eps);
	Q_V[1] = WENO_x(V, dd, c_rl, i+1, j, Nx, 0, eps);
	Q_V[2] = WENO_x(V, dd, c_rl, i-1, j, Nx, 1, eps);
	Q_V[3] = WENO_x(V, dd, c_rl, i, j, Nx, 0, eps);
}

// Compute final 4 V's for pluging WENO reconstructions in y flux
__device__ void compute_Vs_y(float *Q_V, float *V, float *dd, float *c_rl, int i, int j, int Nx, float eps){
	Q_V[0] = WENO_y(V, dd, c_rl, i, j, Nx, 1, eps);
	Q_V[1] = WENO_y(V, dd, c_rl, i, j+1, Nx, 0, eps);
	Q_V[2] = WENO_y(V, dd, c_rl, i, j-1, Nx, 1, eps);
	Q_V[3] = WENO_y(V, dd, c_rl, i, j, Nx, 0, eps);
}

// WENO3 SSP-RK3 step 0
__global__ void kernel_WENO3_step0(int Nx, int Ny, float *p0, float *u0, float *v0, float *p1, float *u1, float *v1, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *S, float N_PML, float S_PML, float E_PML, float W_PML){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 2;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 2;
    if ((i < Nx-2) && (j < Ny-2)){
    	
    	// central index
		int Cij = flat(i,j,Nx);
		
		// restrict
		//if (S[Cij] < 0.0f) return;
		if (S[Cij] < h) return;
    	
    	// flux values
		float F_pP, F_pM, F_uP, F_uM;
		float G_pP, G_pM, G_vP, G_vM;		
		
		// final reconstructed WENO values V+, V-
		float p_V[4], u_V[4], v_V[4];
		
		// (1)
		
		// compute WENO values V_i+, V_i-
		compute_Vs_x(p_V, p0, dd, c_rl, i, j, Nx, eps);
		compute_Vs_x(u_V, u0, dd, c_rl, i, j, Nx, eps);
		compute_Vs_x(v_V, v0, dd, c_rl, i, j, Nx, eps);
		
		// compute flux F_p+
		F_pP = F_p_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_p-
		F_pM = F_p_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		// compute flux F_u+
		F_uP = F_u_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_u-
		F_uM = F_u_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		
		// compute WENO values V_j+, V_j- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_y(p_V, p0, dd, c_rl, i, j, Nx, eps);
		compute_Vs_y(u_V, u0, dd, c_rl, i, j, Nx, eps);
		compute_Vs_y(v_V, v0, dd, c_rl, i, j, Nx, eps);
		
		// compute flux G_p+
		G_pP = G_p_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_p-
		G_pM = G_p_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		// compute flux G_v+
		G_vP = G_v_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_v-
		G_vM = G_v_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		
		u1[Cij] = u0[Cij] + dt*((-F_uP + F_uM)/h);
		v1[Cij] = v0[Cij] + dt*((-G_vP + G_vM)/h);
		p1[Cij] = p0[Cij] + dt*((-F_pP + F_pM - G_pP + G_pM)/h);

		
		sPML(u1, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		sPML(v1, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		sPML(p1, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		
	}
}

// WENO3 SSP-RK3 step 1
__global__ void kernel_WENO3_step1(int Nx, int Ny, float *p0, float *u0, float *v0, float *p1, float *u1, float *v1, float *p2, float *u2, float *v2, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *S, float N_PML, float S_PML, float E_PML, float W_PML){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 2;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 2;
    if ((i < Nx-2) && (j < Ny-2)){
    	
    	// central index
		int Cij = flat(i,j,Nx);
		
		// restrict
		//if (S[Cij] < 0.0f) return;
		if (S[Cij] < h) return;
    	
    	// flux values
		float F_pP, F_pM, F_uP, F_uM;
		float G_pP, G_pM, G_vP, G_vM;		
		
		// final reconstructed WENO values V+, V-
		float p_V[4], u_V[4], v_V[4];
		
		// (2)
		
		// compute WENO values V_i+, V_i-
		compute_Vs_x(p_V, p1, dd, c_rl, i, j, Nx, eps);
		compute_Vs_x(u_V, u1, dd, c_rl, i, j, Nx, eps);
		compute_Vs_x(v_V, v1, dd, c_rl, i, j, Nx, eps);
		
		// compute flux F_p+
		F_pP = F_p_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_p-
		F_pM = F_p_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		// compute flux F_u+
		F_uP = F_u_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_u-
		F_uM = F_u_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		
		// compute WENO values V_j+, V_j- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_y(p_V, p1, dd, c_rl, i, j, Nx, eps);
		compute_Vs_y(u_V, u1, dd, c_rl, i, j, Nx, eps);
		compute_Vs_y(v_V, v1, dd, c_rl, i, j, Nx, eps);
		
		// compute flux G_p+
		G_pP = G_p_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_p-
		G_pM = G_p_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		// compute flux G_v+
		G_vP = G_v_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_v-
		G_vM = G_v_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		
		u2[Cij] = 0.75f*u0[Cij] + 0.25f*(u1[Cij] + dt*((-F_uP + F_uM)/h));
		v2[Cij] = 0.75f*v0[Cij] + 0.25f*(v1[Cij] + dt*((-G_vP + G_vM)/h));
		p2[Cij] = 0.75f*p0[Cij] + 0.25f*(p1[Cij] + dt*((-F_pP + F_pM - G_pP + G_pM)/h));

		
		sPML(u2, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		sPML(v2, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		sPML(p2, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		
	}
}

// WENO3 SSP-RK3 step 2
__global__ void kernel_WENO3_step2(int Nx, int Ny, float *p0, float *u0, float *v0, float *p1, float *u1, float *v1, float *p2, float *u2, float *v2, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *S, float N_PML, float S_PML, float E_PML, float W_PML){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 2;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 2;
    if ((i < Nx-2) && (j < Ny-2)){
		
		// central index
		int Cij = flat(i,j,Nx);
		
		// restrict
		//if (S[Cij] < 0.0f) return;
		if (S[Cij] < h) return;
		
		// flux values
		float F_pP, F_pM, F_uP, F_uM;
		float G_pP, G_pM, G_vP, G_vM;		
		
		// final reconstructed WENO values V+, V-
		float p_V[4], u_V[4], v_V[4];
		
		// (2)
		
		// compute WENO values V_i+, V_i-
		compute_Vs_x(p_V, p2, dd, c_rl, i, j, Nx, eps);
		compute_Vs_x(u_V, u2, dd, c_rl, i, j, Nx, eps);
		compute_Vs_x(v_V, v2, dd, c_rl, i, j, Nx, eps);
		
		// compute flux F_p+
		F_pP = F_p_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_p-
		F_pM = F_p_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		// compute flux F_u+
		F_uP = F_u_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_u-
		F_uM = F_u_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		
		// compute WENO values V_j+, V_j- (re-use of arrays p_V, u_V and v_V)
		compute_Vs_y(p_V, p2, dd, c_rl, i, j, Nx, eps);
		compute_Vs_y(u_V, u2, dd, c_rl, i, j, Nx, eps);
		compute_Vs_y(v_V, v2, dd, c_rl, i, j, Nx, eps);
		
		// compute flux G_p+
		G_pP = G_p_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_p-
		G_pM = G_p_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		// compute flux G_v+
		G_vP = G_v_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_v-
		G_vM = G_v_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		
		// trick to avoid yet another array for each field U is to use U1 input as final u here, since not used
		u1[Cij] = (u0[Cij] + 2.0f*(u2[Cij] + dt*((-F_uP + F_uM)/h)))/3.0f;
		v1[Cij] = (v0[Cij] + 2.0f*(v2[Cij] + dt*((-G_vP + G_vM)/h)))/3.0f;
		p1[Cij] = (p0[Cij] + 2.0f*(p2[Cij] + dt*((-F_pP + F_pM - G_pP + G_pM)/h)))/3.0f;

		
		sPML(u1, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		sPML(v1, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		sPML(p1, i, j, Nx, Ny, h, N_PML, S_PML, E_PML, W_PML);
		
	}
}

__global__ void kernel_zero_boundaries(int Nx, int Ny, float *p, float *u, float *v, float *S){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
    if ((i < Nx) && (j < Ny)){
    	int Cij = flat(i,j,Nx);
    	if (S[Cij] >= 0.0f) return;
    	p[Cij] = 0.0f;
    	u[Cij] = 0.0f;
    	v[Cij] = 0.0f;
    }
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
