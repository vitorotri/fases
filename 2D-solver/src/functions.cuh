#include <cmath>
#include <cstring>
#include "../include/parsing.h"
#include "../include/filterABC.h"

// flattening function
/*
__host__ __device__ int flat(int i, int j, int Nx){
    return (j*Nx) + i;
}
*/
__host__ __device__ float no_slope(float *u, int i, int j, int Nx){
	return 0.0;
}

// regular slopes

__host__ __device__ float forward_x(float *u, int i, int j, int Nx){
	return u[flat(i+1,j,Nx)] - u[flat(i,j,Nx)];
}

__host__ __device__ float backward_x(float *u, int i, int j, int Nx){
	return u[flat(i,j,Nx)] - u[flat(i-1,j,Nx)];
}

__host__ __device__ float central_x(float *u, int i, int j, int Nx){
	return 0.5*(u[flat(i+1,j,Nx)] - u[flat(i-1,j,Nx)]);
}

__host__ __device__ float forward_y(float *u, int i, int j, int Nx){
	return u[flat(i,j+1,Nx)] - u[flat(i,j,Nx)];
}

__host__ __device__ float backward_y(float *u, int i, int j, int Nx){
	return u[flat(i,j,Nx)] - u[flat(i,j-1,Nx)];
}

__host__ __device__ float central_y(float *u, int i, int j, int Nx){
	return 0.5*(u[flat(i,j+1,Nx)] - u[flat(i,j-1,Nx)]);
}

// slope limiter functions

__host__ __device__ float minmod(float down, float up){
	if ((up > 0) && (down > 0)){
		return fmin(up,down);
	}
	else if ((up < 0) && (down < 0)){
		return -fmin(fabsf(up),fabsf(down));
	}
	else return 0.0;
}

__host__ __device__ float minmod_x(float *u, int i, int j, int Nx){
	float up, down;
	up = backward_x(u,i,j,Nx);
	down = forward_x(u,i,j,Nx);
	if ((up > 0) && (down > 0)){
		return fmin(up,down);
	}
	else if ((up < 0) && (down < 0)){
		return -fmin(fabsf(up),fabsf(down));
	}
	else return 0.0;
}

__host__ __device__ float minmod_y(float *u, int i, int j, int Nx){
	float up, down;
	up = backward_y(u,i,j,Nx);
	down = forward_y(u,i,j,Nx);
	if ((up > 0) && (down > 0)){
		return fmin(up,down);
	}
	else if ((up < 0) && (down < 0)){
		return -fmin(fabsf(up),fabsf(down));
	}
	else return 0.0;
}

// adjusted superbees to counteract its aggresive nature

__host__ __device__ float superbee_x(float *u, int i, int j, int Nx) {
    float down = forward_x(u, i, j, Nx);
    float up = backward_x(u, i, j, Nx);
    
    if (down * up <= 0.0f) {
        return 0.0f;
    }
    
    float s1 = minmod(down, 2.0f * up);
    float s2 = minmod(2.0f * down, up);
    float limiter = (fabsf(s1) > fabsf(s2)) ? s1 : s2;

    // Adjust to limit overshoots by bounding the result
    return fminf(fabsf(limiter), 1.5f) * ((limiter > 0.0f) ? 1.0f : -1.0f);
}

__host__ __device__ float superbee_y(float *u, int i, int j, int Nx) {
    float down = forward_y(u, i, j, Nx);
    float up = backward_y(u, i, j, Nx);
    
    if (down * up <= 0.0f) {
        return 0.0f;
    }
    
    float s1 = minmod(down, 2.0f * up);
    float s2 = minmod(2.0f * down, up);
    float limiter = (fabsf(s1) > fabsf(s2)) ? s1 : s2;

    // Adjust to limit overshoots by bounding the result
    return fminf(fabsf(limiter), 1.5f) * ((limiter > 0.0f) ? 1.0f : -1.0f);
}


// auxiliar function for MC limiter (below)
__host__ __device__ float minmod3(float a, float b, float c){
	float min = a;
	if (b < min) {
        min = b;
    }
    if (c < min) {
        min = c;
    }
    return min;
}

__host__ __device__ float MC_x(float *u, int i, int j, int Nx){
	float up, down, centr;
	up = backward_x(u,i,j,Nx);
	down = forward_x(u,i,j,Nx);
	centr = central_x(u,i,j,Nx);
    if (centr == 0.0 || (down*up <= 0.0)) {
        return 0.0;
    }
    if (centr > 0.0) {
        return minmod3(2.0*down, centr, 2.0*up);
    } else {
        return -minmod3(2.0*fabsf(down), fabsf(centr), 2.0*fabsf(up));
    }
}

__host__ __device__ float MC_y(float *u, int i, int j, int Nx){
	float up, down, centr;
	up = backward_y(u,i,j,Nx);
	down = forward_y(u,i,j,Nx);
	centr = central_y(u,i,j,Nx);
    if (centr == 0.0 || (down * up <= 0.0)) {
        return 0.0;
    }
    if (centr > 0.0) {
        return minmod3(2.0*down, centr, 2.0*up);
    } else {
        return -minmod3(2.0*fabsf(down), fabsf(centr), 2.0*fabsf(up));
    }
}

__host__ __device__ float vanLeer_x(float *u, int i, int j, int Nx) {
    float down = forward_x(u, i, j, Nx);
    float up = backward_x(u, i, j, Nx);
    
    if (down * up <= 0.0f) {
        return 0.0f;
    }
    
    return (2.0 * down * up) / (down + up);
}

__host__ __device__ float vanLeer_y(float *u, int i, int j, int Nx) {
    float down = forward_y(u, i, j, Nx);
    float up = backward_y(u, i, j, Nx);
    
    if (down * up <= 0.0f) {
        return 0.0f;
    }
    
    return (2.0 * down * up) / (down + up);
}

__host__ __device__ float vanAlbada_x(float *u, int i, int j, int Nx) {
    float down = forward_x(u, i, j, Nx);
    float up = backward_x(u, i, j, Nx);
    
    if (down * up <= 0.0f) {
        return 0.0f;
    }
    
    return (down * up * (down + up)) / (down * down + up * up);
}

__host__ __device__ float vanAlbada_y(float *u, int i, int j, int Nx) {
    float down = forward_y(u, i, j, Nx);
    float up = backward_y(u, i, j, Nx);
    
    if (down * up <= 0.0f) {
        return 0.0f;
    }
    
    return (down * up * (down + up)) / (down * down + up * up);
}

// Godunov and Roe fluxes (for WENO, no simple substitutions)

// inputs on p,u,v are the reconstructed values
// pP and pM denote, for example, p_plus and p_minus
__host__ __device__ float F_p_Godunov(float pM, float pP, float uM, float uP, float rho, float c, int j){
	return 0.5*(rho*c*c*(uP + uM) - c*(pP - pM));
}

__host__ __device__ float F_u_Godunov(float pM, float pP, float uM, float uP, float rho, float c, int j){
	return 0.5*((pP + pM)/rho - c*(uP - uM));
}

__host__ __device__ float G_p_Godunov(float pM, float pP, float vM, float vP, float rho, float c, int j){
	return 0.5*(rho*c*c*vP + rho*c*c*vM - (c*pP - c*pM));
}

__host__ __device__ float G_v_Godunov(float pM, float pP, float vM, float vP, float rho, float c, int j){
	return 0.5*(pM/rho + pP/rho - (c*vP - c*vM));
}

// limiter type
typedef float (*limiter)(float*, int, int, int);
// array of pointers for limiters
__device__ limiter limiter_functions[] = {no_slope, no_slope, central_x, central_y, minmod_x, minmod_y, superbee_x, superbee_y, MC_x, MC_y, vanLeer_x, vanLeer_y, vanAlbada_x, vanAlbada_y};

// max_idx is a number associated with max stencil number: 2 for regular 3-point stencil
__host__ __device__ void enclosing_BC(float *p, float *u, float *v, int i, int j, int Nx, int Ny, int max_idx){
	
	// For reflective BCs, make velocities negative (or zero) in the right-hand side
	
	// Incorrect BCs can enforce spurious oscillations
	
    // Left boundary
    p[flat(max_idx-1,j,Nx)] = p[flat(max_idx,j,Nx)];
    u[flat(max_idx-1,j,Nx)] = u[flat(max_idx,j,Nx)];
    // for WENO5 (see LeVeque, page 137)
    p[flat(max_idx-2,j,Nx)] = p[flat(max_idx+1,j,Nx)];
    u[flat(max_idx-2,j,Nx)] = u[flat(max_idx+1,j,Nx)];

    // Right boundary
    p[flat(Nx-max_idx,j,Nx)] = p[flat(Nx-(max_idx+1),j,Nx)];
    u[flat(Nx-max_idx,j,Nx)] = u[flat(Nx-(max_idx+1),j,Nx)];
    // for WENO5
   	p[flat(Nx-(max_idx-1),j,Nx)] = p[flat(Nx-(max_idx+1)-1,j,Nx)];
    u[flat(Nx-(max_idx-1),j,Nx)] = u[flat(Nx-(max_idx+1)-1,j,Nx)];
			
    // Top boundary
    /*
    p[flat(i,Ny-max_idx,Nx)] = p[flat(i,Ny-(max_idx+1),Nx)];
    v[flat(i,Ny-max_idx,Nx)] = v[flat(i,Ny-(max_idx+1),Nx)];
    // for WENO5
    p[flat(i,Ny-(max_idx-1),Nx)] = p[flat(i,Ny-(max_idx+1)-1,Nx)];
    v[flat(i,Ny-(max_idx-1),Nx)] = v[flat(i,Ny-(max_idx+1)-1,Nx)];
    */
    // for plane wave traveling on x
    p[flat(i,Ny-max_idx,Nx)] = p[flat(i,Ny-(max_idx+1),Nx)];
    v[flat(i,Ny-max_idx,Nx)] = 0.0;
        
    // Bottom boundary
    /*
    p[flat(i,max_idx-1,Nx)] = p[flat(i,max_idx,Nx)];
    v[flat(i,max_idx-1,Nx)] = v[flat(i,max_idx,Nx)];
    // for WENO5
    p[flat(i,max_idx-2,Nx)] = p[flat(i,max_idx+1,Nx)];
    v[flat(i,max_idx-2,Nx)] = v[flat(i,max_idx+1,Nx)];
    */
    // for plane wave traveling on x
    p[flat(i,max_idx-1,Nx)] = p[flat(i,max_idx,Nx)];
    v[flat(i,max_idx-1,Nx)] = 0.0;
        
}

// Periodic BC in x (absorsive in j = 0 and absorsive in j = Ny-1)
__host__ __device__ void periodic_BC(float *p, float *u, float *v, int i, int j, int Nx, int Ny, int max_idx){
	
    // Left boundary
    p[flat(max_idx-1,j,Nx)] = p[flat(Nx-(max_idx+1),j,Nx)];
    u[flat(max_idx-1,j,Nx)] = u[flat(Nx-(max_idx+1),j,Nx)];

    // Right boundary
    p[flat(Nx-max_idx,j,Nx)] = p[flat(max_idx,j,Nx)];
    u[flat(Nx-max_idx,j,Nx)] = u[flat(max_idx,j,Nx)];
    /*
	// Top boundary
    p[flat(i,Ny-max_idx,Nx)] = p[flat(i,Ny-(max_idx+1),Nx)];
    v[flat(i,Ny-max_idx,Nx)] = v[flat(i,Ny-(max_idx+1),Nx)];
    // for WENO5
    p[flat(i,Ny-(max_idx-1),Nx)] = p[flat(i,Ny-(max_idx+1),Nx)];
    v[flat(i,Ny-(max_idx-1),Nx)] = v[flat(i,Ny-(max_idx+1),Nx)];
        
    // Bottom boundary
    p[flat(i,max_idx-1,Nx)] = p[flat(i,max_idx,Nx)];
    v[flat(i,max_idx-1,Nx)] = v[flat(i,max_idx,Nx)];
    // for WENO5
    p[flat(i,max_idx-2,Nx)] = p[flat(i,max_idx,Nx)];
    v[flat(i,max_idx-2,Nx)] = v[flat(i,max_idx,Nx)];
    */
    
    // Absorsive
    
    // Top boundary
    p[flat(i,Ny-max_idx,Nx)] = p[flat(i,Ny-(max_idx+1),Nx)];
    v[flat(i,Ny-max_idx,Nx)] = v[flat(i,Ny-(max_idx+1),Nx)];
    // Bottom boundary
    p[flat(i,max_idx-1,Nx)] = p[flat(i,max_idx,Nx)];
    v[flat(i,max_idx-1,Nx)] = v[flat(i,max_idx,Nx)];
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
			retval += fabsf(Q[flat(i+1,j,Nx)] - Q[flat(i,j,Nx)]) + fabsf(Q[flat(i,j+1,Nx)] - Q[flat(i,j,Nx)]);
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

__global__ void exchange(int Nx, int Ny, float *p, float *p_old, float *p_new){
	int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
	if ((i < Nx-1) && (j < Ny-1)){
		p_old[flat(i,j,Nx)] = p[flat(i,j,Nx)];
		p[flat(i,j,Nx)] = p_new[flat(i,j,Nx)];
	}
}

// reconstruct slope limited values in x
__device__ void compute_Us_x(float *U, float *u, int i, int j, int Nx, int limiter_IDx){
	U[0] = u[flat(i,j,Nx)] + 0.5*limiter_functions[limiter_IDx](u, i, j, Nx);
	U[1] = u[flat(i+1,j,Nx)] - 0.5*limiter_functions[limiter_IDx](u, i+1, j, Nx);
	U[2] = u[flat(i-1,j,Nx)] + 0.5*limiter_functions[limiter_IDx](u, i-1, j, Nx);
	U[3] = u[flat(i,j,Nx)] - 0.5*limiter_functions[limiter_IDx](u, i, j, Nx);
}

// reconstruct slope limited values in y
__device__ void compute_Us_y(float *U, float *u, int i, int j, int Nx, int limiter_IDy){
	U[0] = u[flat(i,j,Nx)] + 0.5*limiter_functions[limiter_IDy](u, i, j, Nx);
	U[1] = u[flat(i,j+1,Nx)] - 0.5*limiter_functions[limiter_IDy](u, i, j+1, Nx);
	U[2] = u[flat(i,j-1,Nx)] + 0.5*limiter_functions[limiter_IDy](u, i, j-1, Nx);
	U[3] = u[flat(i,j,Nx)] - 0.5*limiter_functions[limiter_IDy](u, i, j, Nx);
}

// delta approximate distribution
// maybe also suited for delta pressure
__host__ void delta(float *p, float a, int pos_x, int pos_y, int Nx, int Ny, float amp){
	// zero-centered normal distribution
	float r2;
	for (int i = 0; i < Nx; i++){
		for (int j = 0; j < Ny; j++){
           	r2 = (i - pos_x)*(i - pos_x) + (j - pos_y)*(j - pos_y);
			p[flat(i,j,Nx)] = amp*exp(-r2/(a*a))/(fabsf(a)*sqrt(M_PI));
		}
	}
}

// hybrid BC
// S should be integer, but to avoid wrong conversions, using floats
// alpha should be an array when proper point-clouds are imported, representing, maybe, the absorption coeff
// for pressure, alpha is different (distinctit from WENO alphas!)
// since workd for PNG, single R, no oid[] was processed
__global__ void kernel_apply_BC_vox(float *q, float *S, int Nx, int Ny, float R){
    int i = blockIdx.x*blockDim.x + threadIdx.x + 2;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 2;
    if ((i < Nx-2) && (j < Ny-2)){
    	
		int Cij = flat(i,j,Nx);
		
		float nodesum = 0.0, sum = 0.0, eps = 1e-7, F;
		
		//R *= -1.0f; // phase inversion, if desired
		
    	// Compute contributions from neighbors for each direction, ensuring parallel paradigm
    	
    	nodesum += R * S[flat(i+1, j, Nx)] * q[flat(i+1, j, Nx)];
    	nodesum += R * S[flat(i-1, j, Nx)] * q[flat(i-1, j, Nx)];
    	nodesum += R * S[flat(i, j+1, Nx)] * q[flat(i, j+1, Nx)];
    	nodesum += R * S[flat(i, j-1, Nx)] * q[flat(i, j-1, Nx)];
    	
    	// for inner nodes, nodesum is already 0.0   
	
    	// Sum of active neighbors (based on S)
    	sum = S[flat(i+1, j, Nx)] + S[flat(i-1, j, Nx)] +
          	  S[flat(i, j+1, Nx)] + S[flat(i, j-1, Nx)];
	
    	// Determine node type
    	F = 1.0 - S[Cij]; // Boundary node factor (If F = 1.0, it is a boundary)
	
    	// Apply boundary conditions
    	q[Cij] = S[Cij]*q[Cij] + F*nodesum/(sum + eps);
    }
}

__device__ float bilinear_interp(float *f, int Nx, float Dx, float Dy, float h){
	int i1 = (int)floorf(Dx/h);
	int i2 = i1 + 1;
	int j1 = (int)floorf(Dy/h);
	int j2 = j1 + 1;

	float x1 = i1 * h, x2 = i2 * h;
	float y1 = j1 * h, y2 = j2 * h;

	float f11 = f[flat(i1, j1, Nx)];
	float f21 = f[flat(i2, j1, Nx)];
	float f12 = f[flat(i1, j2, Nx)];
	float f22 = f[flat(i2, j2, Nx)];

	float wx = (Dx - x1) / (x2 - x1);
	float wy = (Dy - y1) / (y2 - y1);

	return (1 - wx)*(1 - wy)*f11 + wx*(1 - wy)*f21 + (1 - wx)*wy*f12 + wx*wy*f22;
}

__device__ __forceinline__
float cubic_interp_1d(float fm1, float f0, float f1, float f2, float s)
{
    // s in [0,1): position between grid points i (0) and i+1
    float s2 = s * s;

    float w_m1 = -s * (s - 2.0f) * (s - 1.0f) / 6.0f;
    float w_0  =  (s - 2.0f) * (s - 1.0f) * (s + 1.0f) / 2.0f;
    float w_1  =  s * (-s2 + s + 2.0f) / 2.0f;
    float w_2  =  s * (s2 - 1.0f) / 6.0f;

    return w_m1 * fm1 + w_0 * f0 + w_1 * f1 + w_2 * f2;
}


__device__ inline float bicubic_interp(float *f, int Nx, float Dx, float Dy, float h)
{
    // Convert physical position to grid coordinates
    float gx = Dx / h;                // grid coord in x
    float gy = Dy / h;                // grid coord in y

    int i = (int)floorf(gx);          // base index in x
    int j = (int)floorf(gy);          // base index in y

    float sx = gx - (float)i;         // local coord in [0,1)
    float sy = gy - (float)j;         // local coord in [0,1)

    // 4 neighbors in x: i-1, i, i+1, i+2
    int im1 = i - 1;
    int i0  = i;
    int ip1 = i + 1;
    int ip2 = i + 2;

    // 4 neighbors in y: j-1, j, j+1, j+2
    int jm1 = j - 1;
    int j0  = j;
    int jp1 = j + 1;
    int jp2 = j + 2;

    // Interpolate in x for each of the 4 rows
    float g_m1 = cubic_interp_1d(
        f[flat(im1, jm1, Nx)], f[flat(i0, jm1, Nx)],
        f[flat(ip1, jm1, Nx)], f[flat(ip2, jm1, Nx)], sx);

    float g_0 = cubic_interp_1d(
        f[flat(im1, j0, Nx)], f[flat(i0, j0, Nx)],
        f[flat(ip1, j0, Nx)], f[flat(ip2, j0, Nx)], sx);

    float g_1 = cubic_interp_1d(
        f[flat(im1, jp1, Nx)], f[flat(i0, jp1, Nx)],
        f[flat(ip1, jp1, Nx)], f[flat(ip2, jp1, Nx)], sx);

    float g_2 = cubic_interp_1d(
        f[flat(im1, jp2, Nx)], f[flat(i0, jp2, Nx)],
        f[flat(ip1, jp2, Nx)], f[flat(ip2, jp2, Nx)], sx);

    // Finally, interpolate in y using the 4 intermediate values
    float value = cubic_interp_1d(g_m1, g_0, g_1, g_2, sy);

    return value;
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
__device__ inline float chi_layered(float s, float h, float W){
	// not considering cases where s < 0.0, provided sdf kernel computes only where o_IDs are s > 0.0
    if (s >= W)	return 0.0f;
	if (s <= h)	return 1.0f;
    float r = (s - h) / (W - h);
    float smooth = 3.0f*r*r - 2.0f*r*r*r;
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

/*
// implicit solver for the ADE (Auxiliary Differential Equations)
// implicitness come from the analytical solving of the equations, still need SSP-RK3 if main solver uses that
// substituted in the SSP stages is the implicit Euler method at the place of the standard U[n] + dt*L(U[n]),
// so the 'right' value is this operator, 'middle' is the U[n] at stages (2) and (n+1), and 'left' the receiving
// arrays 
__device__ float ADE_solver(int b_ID, uint8_t m_ID, float w_in, float dt, float *A, float *zeta, float *B, float *C, float *alpha, float *beta, float *R_inf, int K, int L, float a_SSP, float b_SSP, float c_SSP, float *phi_left, float *phi_middle, float *phi_right, float *psi_1_left, float *psi_1_middle, float *psi_1_right, float *psi_2_left, float *psi_2_middle, float *psi_2_right){
    float sum_A  = 0.0f;
    float sum_BC = 0.0f;

    for (int k = 0; k < K; k++){
        int m = m_ID*K + k;	// flattened material coeff idx
        int s = b_ID*K + k; // flattened compact state idx

		// -------------------------------------------------

		float phi_BE = (phi_right[s] + dt*w_in) / (1.0f + zeta[m]*dt); // backwards (implicit) Euler solver
		
		phi_left[s] = a_SSP*phi_middle[s] + b_SSP*phi_BE;
		phi_left[s] /= c_SSP;
		sum_A += A[m] * phi_left[s];

    }

    for (int l = 0; l < L; l++){
        int m = m_ID*L + l;	// flattened material coeff idx
        int s = b_ID*L + l;	// flattened compact state idx

        float g = 1.0f + alpha[m]*dt;
        float d = beta[m]*dt;
        float D = g*g + d*d;

        float r1 = psi_1_right[s] + dt*w_in;
        float r2 = psi_2_right[s];
        
        float psi_1_BE = (g*r1 - d*r2)/D; // backwards (implicit) Euler solver
        float psi_2_BE = (d*r1 + g*r2)/D; // backwards (implicit) Euler solver

		psi_1_left[s] = a_SSP*psi_1_middle[s] + b_SSP*psi_1_BE;
		psi_1_left[s] /= c_SSP;
		psi_2_left[s] = a_SSP*psi_2_middle[s] + b_SSP*psi_2_BE;
		psi_2_left[s] /= c_SSP;

        sum_BC += B[m]*psi_1_left[s] + C[m]*psi_2_left[s];
    }

    return R_inf[m_ID]*w_in + sum_A + sum_BC;
}
*/

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
		
		phi_left[s] = a_SSP*phi_middle[s] + b_SSP*(phi_middle[s] + dt*L_phi);
		phi_left[s] /= c_SSP;
		sum_A += A[m] * phi_left[s];

    }

    for (int l = 0; l < L; l++){
        int m = m_ID*L + l;	// flattened material coeff idx
        int s = b_ID*L + l;	// flattened compact state idx
        
        float L_psi_1 = w_in - (alpha[m]*psi_1_right[s] + beta[m]*psi_2_right[s]);
        float L_psi_2 = beta[m]*psi_1_right[s] - alpha[m]*psi_2_right[s];

		psi_1_left[s] = a_SSP*psi_1_middle[s] + b_SSP*(psi_1_middle[s] + dt*L_psi_1);
		psi_1_left[s] /= c_SSP;
		psi_2_left[s] = a_SSP*psi_2_middle[s] + b_SSP*(psi_2_middle[s] + dt*L_psi_2);
		psi_2_left[s] /= c_SSP;

        sum_BC += B[m]*psi_1_left[s] + C[m]*psi_2_left[s];
    }

    return R_inf[m_ID]*w_in + sum_A + sum_BC;
}

// clean version
__global__ void kernel_apply_BC_immersed_sdf(float *u, float *v, float *p, int Nx, int Ny, float h, float *S, float *R, uint8_t *oid, float rho, float c, int n, float dt, float *u_old, float *v_old, float *p_old, float tol, float *T, float *th, float *Rinf, float *Ak, float *zeta, float *Bl, float *Cl, float *alpha, float *beta, int K_m, int L_m, int *bmap, float a_SSP, float b_SSP, float c_SSP, float *phi_left, float *phi_middle, float *phi_right, float *psi_1_left, float *psi_1_middle, float *psi_1_right, float *psi_2_left, float *psi_2_middle, float *psi_2_right){

	int i = blockIdx.x*blockDim.x + threadIdx.x + 2; // bicubic interp limits to + 3 instead of +2
	int j = blockIdx.y*blockDim.y + threadIdx.y + 2;
    if ((i < Nx-2) && (j < Ny-2)){
		
    	int Cij = flat(i,j,Nx);
    	
    	float s = S[Cij]; // signed distance from cell center to boundary
    	
    	// only act on immersed cells (fluid-side band from your OID map), provided
    	// those values are only on positive s, which is the case here
    	// oid[] ranges from 1 to nMat, 255 elsewhere
		const uint8_t o_ID = oid[Cij]; // object's material ID
		if (o_ID == 0xFF) return;
		
		int b_ID = bmap[Cij];
		//if (b_ID < 0) return; // safety; should match oid test
		
		// gradients from S[] for detecting normals   
    	float gx = (S[flat(i+1,j,Nx)] - S[flat(i-1,j,Nx)]) * (0.5f / h);
    	float gy = (S[flat(i,j+1,Nx)] - S[flat(i,j-1,Nx)]) * (0.5f / h);
    	float gn = sqrtf(gx*gx + gy*gy);
    	gn = fmaxf(gn, 1e-12f); //safety
	
    	float nx = gx/gn;
    	float ny = gy/gn;
	
    	// cell center
    	float Cx = i * h;
    	float Cy = j * h;
		
    	// footpoint on boundary: W = C - s * n_hat  (since S>0 fluid, S<0 solid) 
    	float Wx = Cx - s * nx;
    	float Wy = Cy - s * ny;
	
    	// interpolation point slightly inside the fluid along n
    	float l = s + h;
    	float Dx = Wx + l * nx;
    	float Dy = Wy + l * ny;
		
    	float pD = bicubic_interp(p_old, Nx, Dx, Dy, h);
    	float uD = bicubic_interp(u_old, Nx, Dx, Dy, h);
    	float vD = bicubic_interp(v_old, Nx, Dx, Dy, h);

		//float R_obj = R[o_ID];
    	
    	// --------------- barrier approach (inside domain, transmission) -------------------
    	
    	// NEW: per-object thickness and transmission
        float th_obj = th[o_ID]; // 0 for walls, >0 for partitions
        float T_obj  = T[o_ID]; // 0 for walls, >0 for partitions

        // --- local side normal/tangential ---
        const float un =  uD*nx + vD*ny;
        const float ut = -uD*ny + vD*nx;

        const float Z0 = rho * c;
        //float Zb = R_to_Zb(R_obj, Z0);

        // ------------------- incoming characteristic from fluid LOCAL side (towards obstacle) -------
        const float w_in_local = pD - Z0 * un;
		
        // ---------- other side sample (always computed) ----------
        // For partitions: sample across thickness + offset into opposite fluid.
        // For walls: th_obj == 0, T_obj == 0 ⇒ this term is harmless and
        // its contribution is killed by T_obj.
        float Ex = Wx - (th_obj + l) * nx;	// cross partition and into opposite fluid
        float Ey = Wy - (th_obj + l) * ny;
        // interpolation safety clamping for outer boundaries limits
        Ex = fminf(fmaxf(Ex, 2.0f*h), (Nx-3)*h);
		Ey = fminf(fmaxf(Ey, 2.0f*h), (Ny-3)*h);

        float pE = bilinear_interp(p_old, Nx, Ex, Ey, h);
        float uE = bilinear_interp(u_old, Nx, Ex, Ey, h);
        float vE = bilinear_interp(v_old, Nx, Ex, Ey, h);

        // normal velocity on opposite side relative to +n direction
        const float un_opp = -(uE*nx + vE*ny);

        // choose a characteristic definition for the opposite side.
        // The exact sign convention you'll tune, but it does NOT matter
        // for non-partitions because T_obj == 0 there.
        const float w_in_opp = pE - Z0 * un_opp;

        // ---------- unified scattering (no extra branch) ----------
        //
        // Generalized 1-port view for THIS side:
        //   w_out_local = R_obj * w_in_local + T_obj * w_in_opp
        //
        // For walls: T_obj = 0  ⇒ w_out_local = R_obj * w_in_local.
        //
        //
        
        //const float w_out_local = R_obj * w_in_local + T_obj * w_in_opp;
        
        // considering only reflection for now
        // if using transmission, also implement implicit computation of w_in_opp similar to w_in_local
        
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
        const float u_b = un_b*nx - ut_b*ny;
        const float v_b = un_b*ny + ut_b*nx;    	
    	
    	const float chi = chi_layered(s, h, tol);
		/*
		u[Cij] = u_b;
    	v[Cij] = v_b;
    	p[Cij] = p_b;
    	*/
		
		u[Cij] = (1.0f - chi) * u[Cij] + chi * u_b;
    	v[Cij] = (1.0f - chi) * v[Cij] + chi * v_b;
    	p[Cij] = (1.0f - chi) * p[Cij] + chi * p_b;
    	
    	    
    }
}

// sPML Layer - "simple" PML according to:
// 
// https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/AcousticsTimeDomain.html
// 
// 
__host__ __device__ void sPML(float *q, int i, int j, int Nx, int Ny, float h){
    float s_max = 0.018f; // Maximum absorption factor
    int max_idx = (int)(0.15f*(Nx < Ny ? Nx : Ny)); // halfwidth
    float half   = (float)max_idx;

	// ramps in [0,1] (to exclude a side, multiply by 0.0f)
	float rx = fmaxf(0.0f, fminf((i - (Nx - half)) / half, 1.0f)); // right boundary ramp
	float lx = fmaxf(0.0f, fminf((half - i) / half, 1.0f)); // left boundary ramp
	float ly = fmaxf(0.0f, fminf((half - j) / half, 1.0f)); // lower boundary ramp
	float uy = fmaxf(0.0f, fminf((j - (Ny - half)) / half, 1.0f)); // upper boundary ramp)
	
	// cubic (faster than powf)
	float rx3 = rx*rx*rx, lx3 = lx*lx*lx, ly3 = ly*ly*ly, uy3 = uy*uy*uy;
	
	// per-axis combined strengths (L/R and U/D are mutually exclusive)
	float sx = s_max * (rx3 + lx3);
	float sy = s_max * (uy3 + ly3);
	
	// exact corner composition
	//float s = 1.0f - (1.0f - sx) * (1.0f - sy);
	float s = sx + sy - sx * sy;
	
	// (optional) keep strictly < 1.0 to avoid killing the value numerically
	s = fminf(s, 0.999f);
	
	// apply once
	int Cij = flat(i, j, Nx);
	q[Cij] -= s * q[Cij];
    
}

// Euler integration kernel
__global__ void kernel_Euler(int Nx, int Ny, float *p0, float *u0, float *v0, float *p1, float *u1, float *v1, float rho, float c, float h, float dt, int limiter_IDx, int limiter_IDy, float *S){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    if ((i < Nx-1) && (j < Ny-1)){
		// flux values
		float F_pP, F_pM, F_uP, F_uM;
		float G_pP, G_pM, G_vP, G_vM;		
		
		// final reconstructed slope limiter values uR, uL
		float p_V[4], u_V[4], v_V[4];
		
		// central index
		int Cij = flat(i,j,Nx);
		
		// restrict
		if (S[Cij] < 0.0f) return;
		
		// compute slope limited values u_iR, u_iL
		compute_Us_x(p_V, p0, i, j, Nx, limiter_IDx);
		compute_Us_x(u_V, u0, i, j, Nx, limiter_IDx);
		compute_Us_x(v_V, v0, i, j, Nx, limiter_IDx);
		
		// compute flux F_p+
		F_pP = F_p_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_p-
		F_pM = F_p_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		// compute flux F_u+
		F_uP = F_u_Godunov(p_V[0], p_V[1], u_V[0], u_V[1], rho, c, j);
		// compute flux F_u-
		F_uM = F_u_Godunov(p_V[2], p_V[3], u_V[2], u_V[3], rho, c, j);
		
		// compute slope limited values u_jR, u_jL (re-use of arrays p_V, u_V and v_V)
		compute_Us_y(p_V, p0, i, j, Nx, limiter_IDy);
		compute_Us_y(u_V, u0, i, j, Nx, limiter_IDy);
		compute_Us_y(v_V, v0, i, j, Nx, limiter_IDy);
		
		// compute flux G_p+
		G_pP = G_p_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_p-
		G_pM = G_p_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		// compute flux G_v+
		G_vP = G_v_Godunov(p_V[0], p_V[1], v_V[0], v_V[1], rho, c, j);
		// compute flux G_v-
		G_vM = G_v_Godunov(p_V[2], p_V[3], v_V[2], v_V[3], rho, c, j-1);
		
		u1[Cij] += dt*((-F_uP + F_uM)/h);
		v1[Cij] += dt*((-G_vP + G_vM)/h);
		p1[Cij] += dt*((-F_pP + F_pM - G_pP + G_pM)/h);
		/*
		sPML(u1, i, j, Nx, Ny, h);
		sPML(v1, i, j, Nx, Ny, h);
		sPML(p1, i, j, Nx, Ny, h);
		*/
	}
}

// --- WENO3 functions ---

// reconstruct in x, for k = 2
// idx = 0 or idx = 1 (default is 1, if 0, will compute V_{i-1/2} instead of V_{i+1/2})
__device__ float construct_Vi_12(float *u, int i, int j, int Nx, float *c_rl, int r, int idx){
	float retval = 0.0;
	for (int l = 0; l <= 1; l++){
		//if (r+idx > 3) printf("[!] Violation");
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
__global__ void kernel_WENO3_step0(int Nx, int Ny, float *p0, float *u0, float *v0, float *p1, float *u1, float *v1, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *S){
	
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

		/*
		sPML(u1, i, j, Nx, Ny, h);
		sPML(v1, i, j, Nx, Ny, h);
		sPML(p1, i, j, Nx, Ny, h);
		*/
	}
}

// WENO3 SSP-RK3 step 1
__global__ void kernel_WENO3_step1(int Nx, int Ny, float *p0, float *u0, float *v0, float *p1, float *u1, float *v1, float *p2, float *u2, float *v2, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *S){
	
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

		/*
		sPML(u2, i, j, Nx, Ny, h);
		sPML(v2, i, j, Nx, Ny, h);
		sPML(p2, i, j, Nx, Ny, h);
		*/
	}
}

// WENO3 SSP-RK3 step 2
__global__ void kernel_WENO3_step2(int Nx, int Ny, float *p0, float *u0, float *v0, float *p1, float *u1, float *v1, float *p2, float *u2, float *v2, float rho, float c, float h, float dt, float *c_rl, float eps, float *dd, float *S){
	
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

		/*
		sPML(u1, i, j, Nx, Ny, h);
		sPML(v1, i, j, Nx, Ny, h);
		sPML(p1, i, j, Nx, Ny, h);
		*/
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
