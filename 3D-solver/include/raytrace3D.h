#pragma once
/*
raytrace3D.h
------------
Backward (receiver -> source) specular acoustic ray tracer, built to CONTINUE the
ISM (ism3D.h) into the late field: it registers only paths of order >= order_min
(= the ISM max_order), so early specular reflections stay the ISM's job and the
tracer fills the reverberant tail that image enumeration is too expensive to reach.

Adapted from the author's backward light ray tracer (raytracer.c): same vec3-array
style, same "shoot rays, reflect, accumulate" shape. Two changes for acoustics:
  * surfaces are PLANES (mesh triangles), not spheres -- closest hit via a flattened
    median-split AABB BVH (built once, reused every band/ray; O(log n) per bounce so
    imported furnished rooms stay tractable). Sources keep the ray-SPHERE test from
    the light tracer, reused as finite-radius DETECTORS (reciprocity: emit at the
    receiver, detect at the source).
  * transparency/refraction dropped; each bounce loses energy by the hit facet's
    absorption alpha (per octave band), resolved through the triangle's group_id.

Emission: each receiver is a finite sphere; n_rays directions on a Fibonacci sphere
leave its surface. Detection: on every segment, a source sphere crossed BEFORE the
next wall is an exact specular arrival (no shadow ray needed for specular).

Output: an ENERGY echogram, energy[Nt*Nr] (same flat(n,r,Nt)=r*Nt+n layout as p_ism),
accumulated only for samples n in (t_cross[r], Nt). Energy, not pressure: the caller
turns it into the pressure tail (e.g. random-sign sqrt), matching the ISM convention.
`gain` is the calibration knob (detector solid angle / ray count are not first-
principles here -- the physical world needs tuning a minimal model cannot see).

Frame: positions are given RAW (as in src.txt/rec.txt); rt_build stores the scene
shift/scale and transforms them to (raw+shift)*scale, identical to
Mesh3D::vertex_transformed, so receivers/sources and walls share one frame.

Requires parsing3D.h (vec3, Tri, Scene3D, ray_triangle_intersect, HD) included first.
CUDA path (one thread per ray, atomic deposits) when compiled with nvcc; OpenMP
fallback (parallel over receivers) otherwise. Define RT_FORCE_CPU to force the host
path even under nvcc.
*/

#include "parsing3D.h"
#include <vector>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <algorithm>

// parsing3D.h #undef's its HD macro at end-of-header, so define our own
#ifdef __CUDACC__
  #define RTHD __host__ __device__ __forceinline__
#else
  #define RTHD inline
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

// ============================================================
// Flattened scene + BVH (built once on the host)
// ============================================================
struct RTNode {
    vec3 lo = v3( FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 hi = v3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    int  left  = -1;   // internal: index of first child (children are left, left+1)
    int  first = 0;    // leaf: first tri in tri_order; count>0 marks a leaf
    int  count = 0;
};

struct RTScene {
    // triangle vertices already transformed to (v+shift)*scale
    std::vector<vec3> A, B, C;
    std::vector<int>  gid;      // group id per triangle -> material alpha
    std::vector<RTNode> nodes;
    std::vector<int>  tri_order;
    int   max_gid = -1;
    vec3  shift = v3(0,0,0);
    float scale = 1.f;
    vec3  xform(const vec3& raw) const { return mul3(add3(raw, shift), scale); }
};

// -------- BVH build (host) --------
inline static void rt_node_bounds(RTScene& S, RTNode& nd){
    nd.lo = v3( FLT_MAX, FLT_MAX, FLT_MAX);
    nd.hi = v3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
    for(int i=0;i<nd.count;++i){
        const int t = S.tri_order[(size_t)(nd.first+i)];
        const vec3* vs[3] = {&S.A[(size_t)t], &S.B[(size_t)t], &S.C[(size_t)t]};
        for(const vec3* p : vs){
            nd.lo.x=std::min(nd.lo.x,p->x); nd.lo.y=std::min(nd.lo.y,p->y); nd.lo.z=std::min(nd.lo.z,p->z);
            nd.hi.x=std::max(nd.hi.x,p->x); nd.hi.y=std::max(nd.hi.y,p->y); nd.hi.z=std::max(nd.hi.z,p->z);
        }
    }
}

inline static void rt_subdivide(RTScene& S, int ni, const std::vector<vec3>& cent, int leaf_cap){
    RTNode nd = S.nodes[(size_t)ni];
    if(nd.count <= leaf_cap) return;                 // small enough: stays a leaf
    const vec3 ext = sub3(nd.hi, nd.lo);
    const int axis = (ext.x>ext.y && ext.x>ext.z) ? 0 : (ext.y>ext.z ? 1 : 2);
    int* beg = &S.tri_order[(size_t)nd.first];
    int* end = beg + nd.count;
    int* mid = beg + nd.count/2;
    std::nth_element(beg, mid, end, [&](int a, int b){
        const float ca = (&cent[(size_t)a].x)[axis], cb = (&cent[(size_t)b].x)[axis];
        return ca < cb;
    });
    const int half = nd.count/2;
    const int li = (int)S.nodes.size();
    RTNode L, R;
    L.first = nd.first;        L.count = half;
    R.first = nd.first + half; R.count = nd.count - half;
    S.nodes.push_back(L); S.nodes.push_back(R);
    rt_node_bounds(S, S.nodes[(size_t)li]);
    rt_node_bounds(S, S.nodes[(size_t)(li+1)]);
    S.nodes[(size_t)ni].left  = li;                  // becomes internal
    S.nodes[(size_t)ni].count = 0;
    rt_subdivide(S, li,   cent, leaf_cap);
    rt_subdivide(S, li+1, cent, leaf_cap);
}

inline static RTScene rt_build(const Scene3D& Sc, int leaf_cap = 4){
    RTScene S; S.shift = Sc.shift; S.scale = Sc.scale;
    const int nt = (int)Sc.tris.size();
    S.A.resize((size_t)nt); S.B.resize((size_t)nt); S.C.resize((size_t)nt);
    S.gid.resize((size_t)nt); S.tri_order.resize((size_t)nt);
    std::vector<vec3> cent((size_t)nt);
    for(int t=0;t<nt;++t){
        const Tri& tr = Sc.tris[(size_t)t];
        S.A[(size_t)t]=Sc.vertex_transformed(tr.v0);
        S.B[(size_t)t]=Sc.vertex_transformed(tr.v1);
        S.C[(size_t)t]=Sc.vertex_transformed(tr.v2);
        S.gid[(size_t)t]=tr.group_id; S.max_gid=std::max(S.max_gid,tr.group_id);
        S.tri_order[(size_t)t]=t;
        cent[(size_t)t]=mul3(add3(add3(S.A[(size_t)t],S.B[(size_t)t]),S.C[(size_t)t]),1.f/3.f);
    }
    if(nt==0) return S;
    RTNode root; root.first=0; root.count=nt;
    S.nodes.push_back(root);
    rt_node_bounds(S, S.nodes[0]);
    rt_subdivide(S, 0, cent, leaf_cap);
    return S;
}

// ============================================================
// Device-friendly view (plain pointers; valid on host too)
// ============================================================
struct RTView {
    const vec3* A; const vec3* B; const vec3* C; const int* gid;
    const RTNode* nodes; const int* tri_order; int nnodes; int ntris;
    const float* alpha; int nalpha;   // [max_gid+1] absorption for the CURRENT band
    const vec3* src; const float* srcp; int nsrc; float src_r2;
};

// slab test: does ray (o,d) reach the box within [0, tmax]?
RTHD static bool rt_box_hit(const vec3& o, const vec3& invd, const vec3& lo, const vec3& hi, float tmax){
    float t0x=(lo.x-o.x)*invd.x, t1x=(hi.x-o.x)*invd.x; if(t0x>t1x){float s=t0x;t0x=t1x;t1x=s;}
    float t0y=(lo.y-o.y)*invd.y, t1y=(hi.y-o.y)*invd.y; if(t0y>t1y){float s=t0y;t0y=t1y;t1y=s;}
    float t0z=(lo.z-o.z)*invd.z, t1z=(hi.z-o.z)*invd.z; if(t0z>t1z){float s=t0z;t0z=t1z;t1z=s;}
    float tmin=fmaxf(fmaxf(t0x,t0y),t0z);
    float tmx =fminf(fminf(t1x,t1y),t1z);
    return tmx>=fmaxf(tmin,0.f) && tmin<=tmax;
}

// nearest wall hit; returns triangle index or -1, sets out_t and out_gid
RTHD static int rt_closest(const RTView& V, const vec3& o, const vec3& d, float& out_t, int& out_gid){
    const vec3 invd = v3(1.f/d.x, 1.f/d.y, 1.f/d.z);
    int best=-1; float bt=FLT_MAX;
    int stack[64]; int sp=0; stack[sp++]=0;
    while(sp>0){
        const RTNode& nd = V.nodes[stack[--sp]];
        if(!rt_box_hit(o, invd, nd.lo, nd.hi, bt)) continue;
        if(nd.count>0){
            for(int i=0;i<nd.count;++i){
                const int t=V.tri_order[nd.first+i];
                float tt;
                if(ray_triangle_intersect(o,d,V.A[t],V.B[t],V.C[t],1e-4f,tt) && tt<bt){
                    bt=tt; best=t;
                }
            }
        } else if(sp<62){
            stack[sp++]=nd.left; stack[sp++]=nd.left+1;
        }
    }
    if(best>=0){ out_t=bt; out_gid=V.gid[best]; }
    return best;
}

// ray-sphere: nearest positive t of ray (o, unit d) with sphere (center, r2), else -1
RTHD static float rt_sphere_t(const vec3& o, const vec3& d, const vec3& ctr, float r2){
    const vec3 oc = sub3(o, ctr);
    const float b = dot3(oc, d);            // a==1 (d unit)
    const float c = dot3(oc, oc) - r2;
    const float disc = b*b - c;
    if(disc < 0.f) return -1.f;
    const float s = sqrtf(disc);
    float t = -b - s;
    if(t <= 1e-4f) t = -b + s;
    return (t > 1e-4f) ? t : -1.f;
}

RTHD static vec3 rt_reflect(const vec3& inc, const vec3& n){   // inc, n unit; n outward
    return sub3(inc, mul3(n, 2.f*dot3(inc, n)));
}

RTHD static vec3 rt_normalize(const vec3& v){
    const float l = sqrtf(norm2_3(v)); return (l>1e-20f) ? mul3(v, 1.f/l) : v;
}

// Fibonacci-sphere direction i of n (unit), golden-angle spiral
RTHD static vec3 rt_fib_dir(int i, int n){
    const float z = 1.f - 2.f*(i + 0.5f)/(float)n;
    const float r = sqrtf(fmaxf(0.f, 1.f - z*z));
    const float phi = (float)i * 2.399963229728653f;   // golden angle
    return v3(r*cosf(phi), r*sinf(phi), z);
}

// cheap counter-based RNG (device-safe, no <random>): hash the state, advance LCG
RTHD static float rt_u01(unsigned& st){
    st = st*1664525u + 1013904223u;
    unsigned x = st; x ^= x>>16; x *= 0x7feb352dU; x ^= x>>15;
    return (x>>8) * (1.f/16777216.f);        // 24-bit in [0,1)
}

// cosine-weighted hemisphere direction about unit normal n (Lambertian scatter)
RTHD static vec3 rt_cos_hemi(const vec3& n, float u1, float u2){
    const float r = sqrtf(u1), phi = 6.2831853f*u2;
    const float x = r*cosf(phi), y = r*sinf(phi), z = sqrtf(fmaxf(0.f, 1.f-u1));
    const vec3 t = (fabsf(n.x) > 0.9f) ? v3(0,1,0) : v3(1,0,0);
    const vec3 b1 = rt_normalize(cross3(t, n));
    const vec3 b2 = cross3(n, b1);
    return rt_normalize(add3(add3(mul3(b1,x), mul3(b2,y)), mul3(n,z)));
}

// ============================================================
// Trace one ray, deposit into energy[] (atomic on device). Shared host/device.
// r = receiver index, o0 = receiver center, dir0 = emission direction.
// ============================================================
// atomicAdd is device-only. Select it only during the DEVICE compile pass
// (__CUDA_ARCH__ is defined then), NOT merely under nvcc: __CUDACC__ is also set
// in the host pass of a __host__ __device__ function, where atomicAdd is illegal.
#if defined(__CUDA_ARCH__)
  #define RT_ATOMIC_ADD(p,v) atomicAdd((p),(v))
#else
  #define RT_ATOMIC_ADD(p,v) (*(p) += (v))
#endif

RTHD static void rt_trace_one(const RTView& V, int r, const vec3& o0, const vec3& dir0,
                            int order_min, int order_max, float c, float fs, int Nt,
                            int t_cross, float gain_over_nrays, float* energy, unsigned seed){
    vec3 o = o0;         // emit from the receiver centre along the Fibonacci direction
    vec3 d = dir0;
    unsigned st = seed;
    float path = 0.f;      // metres travelled to the current origin
    float thr  = 1.f;      // energy throughput = product of (1-alpha) over bounces so far
    // DIFFUSE (Lambertian) tracing with a shadow ray to each source at every bounce:
    // specular + source-detection spheres almost never score a hit, so the tail was
    // empty. Diffuse scatter + source visibility test deposits on (nearly) every
    // bounce, giving the dense late field a reverberant tail needs.
    for(int order=1; order<=order_max; ++order){
        float tw; int gid;
        const int hit = rt_closest(V, o, d, tw, gid);
        if(hit<0) return;                                  // ray escaped the scene

        const vec3 p = add3(o, mul3(d, tw));               // surface hit point
        vec3 nrm = cross3(sub3(V.B[hit],V.A[hit]), sub3(V.C[hit],V.A[hit]));
        const float nl = sqrtf(norm2_3(nrm));
        if(nl>1e-20f) nrm = mul3(nrm, 1.f/nl); else return;
        if(dot3(nrm, d) > 0.f) nrm = mul3(nrm, -1.f);      // face against the ray
        const float a = (gid>=0 && gid<V.nalpha) ? V.alpha[gid] : 0.f;
        const float refl = 1.f - a;                        // energy reflectance of this facet
        const vec3 po = add3(p, mul3(nrm, 1e-4f));         // offset off the surface
        const float path_hit = path + tw;                  // metres receiver->this hit

        // --- shadow ray to each source (deposit if visible), order >= order_min ---
        if(order >= order_min){
            for(int s=0; s<V.nsrc; ++s){
                const vec3 to = sub3(V.src[s], po);
                const float L = sqrtf(norm2_3(to));
                if(L < 1e-4f) continue;
                const vec3 ds = mul3(to, 1.f/L);
                const float cs = dot3(nrm, ds);
                if(cs <= 0.f) continue;                    // source behind the facet
                float tw2; int g2;
                const int occ = rt_closest(V, po, ds, tw2, g2);
                if(occ>=0 && tw2 < L - 1e-3f) continue;    // a wall blocks the source
                const float dtot = path_hit + L;
                const int nidx = (int)(dtot/c*fs);
                if(nidx>t_cross && nidx<Nt){
                    // Lambertian deposit: throughput * this facet's reflectance *
                    // cos / pi, spherical spreading to the source. Constants folded
                    // into gain (calibration knob).
                    const float e = V.srcp[s]*V.srcp[s]*thr*refl*cs*(1.f/(float)M_PI)
                                  * gain_over_nrays / (4.f*(float)M_PI*dtot*dtot);
                    RT_ATOMIC_ADD(&energy[(size_t)r*Nt + nidx], e);
                }
            }
        }

        // --- absorb, diffuse-scatter, continue ---
        thr *= refl;
        if(thr < 1e-7f) return;                            // below -70 dB: drop the ray
        path = path_hit;
        d = rt_cos_hemi(nrm, rt_u01(st), rt_u01(st));      // Lambertian bounce
        o = po;
    }
}

// ============================================================
// CUDA kernel + launcher
// ============================================================
#if defined(__CUDACC__) && !defined(RT_FORCE_CPU)
__global__ void rt_kernel(RTView V, const vec3* rec, int Nr, int n_rays,
                          int order_min, int order_max, float c, float fs, int Nt,
                          const int* t_cross, float gain_over_nrays, float* energy){
    const long long tid = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    const long long total = (long long)Nr*n_rays;
    if(tid >= total) return;
    const int r = (int)(tid / n_rays);
    const int i = (int)(tid % n_rays);
    const vec3 dir = rt_fib_dir(i, n_rays);
    const unsigned seed = (unsigned)r*2246822519u + (unsigned)i*3266489917u + 0x9e3779b9u;
    rt_trace_one(V, r, rec[r], dir, order_min, order_max, c, fs, Nt,
                 t_cross[r], gain_over_nrays, energy, seed);
}
#endif

// ============================================================
// Public entry (per band): fills energy[Nt*Nr] for t in (t_cross[r], Nt)
// ============================================================
inline static void rt_trace_tail(
    const RTScene& S,
    const float* alpha_group, int nGroups,             // absorption for THIS band, per group
    const std::vector<float>& rec_arr,                 // raw xyz, Nr*3
    const std::vector<float>& src_arr,                 // raw xyz, Ns*3
    const std::vector<float>& src_pressure,            // Ns (val_s); {} => all 1
    float rec_radius, float src_radius,
    int n_rays, int order_min, int order_max,
    float c, float fs, int Nt,
    const std::vector<int>& t_cross,                   // Nr
    float gain,
    std::vector<float>& energy)                        // out, resized to Nt*Nr
{
    const int Nr = (int)(rec_arr.size()/3), Ns = (int)(src_arr.size()/3);
    energy.assign((size_t)Nt*Nr, 0.f);
    if(Nr==0 || Ns==0 || S.A.empty() || n_rays<=0) return;

    // transform endpoints into the mesh frame; build padded alpha [max_gid+1]
    std::vector<vec3> rec((size_t)Nr), src((size_t)Ns);
    for(int r=0;r<Nr;++r) rec[(size_t)r]=S.xform(v3(rec_arr[(size_t)(3*r)],rec_arr[(size_t)(3*r+1)],rec_arr[(size_t)(3*r+2)]));
    for(int s=0;s<Ns;++s) src[(size_t)s]=S.xform(v3(src_arr[(size_t)(3*s)],src_arr[(size_t)(3*s+1)],src_arr[(size_t)(3*s+2)]));
    std::vector<float> srcp((size_t)Ns, 1.f);
    for(int s=0;s<Ns && s<(int)src_pressure.size();++s) srcp[(size_t)s]=src_pressure[(size_t)s];
    std::vector<float> alpha((size_t)std::max(0,S.max_gid+1), 0.f);
    for(int g=0; g<(int)alpha.size() && g<nGroups; ++g) alpha[(size_t)g]=alpha_group[(size_t)g];

    std::vector<int> tcx((size_t)Nr, 0);
    for(int r=0;r<Nr && r<(int)t_cross.size();++r) tcx[(size_t)r]=t_cross[(size_t)r];

    // receiver radius only sets emission origin offset (handled as center here);
    // src_radius is the detection cross-section
    (void)rec_radius;
    const float src_r2 = src_radius*src_radius;
    const float gain_over_nrays = gain / (float)n_rays;

    RTView V;
    V.A=S.A.data(); V.B=S.B.data(); V.C=S.C.data(); V.gid=S.gid.data();
    V.nodes=S.nodes.data(); V.tri_order=S.tri_order.data();
    V.nnodes=(int)S.nodes.size(); V.ntris=(int)S.A.size();
    V.alpha=alpha.data(); V.nalpha=(int)alpha.size(); V.src=src.data(); V.srcp=srcp.data(); V.nsrc=Ns; V.src_r2=src_r2;

#if defined(__CUDACC__) && !defined(RT_FORCE_CPU)
    int ndev=0; cudaGetDeviceCount(&ndev);
    if(ndev>0){
        // upload scene + query arrays, run one thread per ray
        vec3 *dA,*dB,*dC,*dsrc,*drec; int *dgid,*dord,*dtcx; RTNode* dnodes; float *dalpha,*dsrcp,*denergy;
        const size_t nt=S.A.size(), nn=S.nodes.size(), na=alpha.size();
        cudaMalloc(&dA,nt*sizeof(vec3)); cudaMalloc(&dB,nt*sizeof(vec3)); cudaMalloc(&dC,nt*sizeof(vec3));
        cudaMalloc(&dgid,nt*sizeof(int)); cudaMalloc(&dord,S.tri_order.size()*sizeof(int));
        cudaMalloc(&dnodes,nn*sizeof(RTNode)); cudaMalloc(&dalpha,na*sizeof(float));
        cudaMalloc(&dsrc,(size_t)Ns*sizeof(vec3)); cudaMalloc(&dsrcp,(size_t)Ns*sizeof(float));
        cudaMalloc(&drec,(size_t)Nr*sizeof(vec3)); cudaMalloc(&dtcx,(size_t)Nr*sizeof(int));
        cudaMalloc(&denergy,(size_t)Nt*Nr*sizeof(float));
        cudaMemcpy(dA,S.A.data(),nt*sizeof(vec3),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,S.B.data(),nt*sizeof(vec3),cudaMemcpyHostToDevice);
        cudaMemcpy(dC,S.C.data(),nt*sizeof(vec3),cudaMemcpyHostToDevice);
        cudaMemcpy(dgid,S.gid.data(),nt*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(dord,S.tri_order.data(),S.tri_order.size()*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(dnodes,S.nodes.data(),nn*sizeof(RTNode),cudaMemcpyHostToDevice);
        cudaMemcpy(dalpha,alpha.data(),na*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(dsrc,src.data(),(size_t)Ns*sizeof(vec3),cudaMemcpyHostToDevice);
        cudaMemcpy(dsrcp,srcp.data(),(size_t)Ns*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(drec,rec.data(),(size_t)Nr*sizeof(vec3),cudaMemcpyHostToDevice);
        cudaMemcpy(dtcx,tcx.data(),(size_t)Nr*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemset(denergy,0,(size_t)Nt*Nr*sizeof(float));
        RTView DV=V; DV.A=dA;DV.B=dB;DV.C=dC;DV.gid=dgid;DV.nodes=dnodes;DV.tri_order=dord;
        DV.alpha=dalpha;DV.src=dsrc;DV.srcp=dsrcp;
        const long long total=(long long)Nr*n_rays; const int TPB=128;
        rt_kernel<<<(int)((total+TPB-1)/TPB),TPB>>>(DV,drec,Nr,n_rays,order_min,order_max,
                                                    c,fs,Nt,dtcx,gain_over_nrays,denergy);
        cudaError_t kerr = cudaGetLastError();               // surface launch failures
        if(kerr==cudaSuccess) kerr = cudaDeviceSynchronize();
        if(kerr!=cudaSuccess)
            std::fprintf(stderr,"[RT] kernel failed: %s -- tail energy will be zero\n",
                         cudaGetErrorString(kerr));
        cudaMemcpy(energy.data(),denergy,(size_t)Nt*Nr*sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dgid);cudaFree(dord);cudaFree(dnodes);
        cudaFree(dalpha);cudaFree(dsrc);cudaFree(dsrcp);cudaFree(drec);cudaFree(dtcx);cudaFree(denergy);
        return;
    }
#endif
    // host / OpenMP fallback: parallelize over receivers (no deposit races per slice)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(int r=0;r<Nr;++r)
        for(int i=0;i<n_rays;++i){
            const unsigned seed = (unsigned)r*2246822519u + (unsigned)i*3266489917u + 0x9e3779b9u;
            rt_trace_one(V, r, rec[(size_t)r], rt_fib_dir(i,n_rays),
                         order_min, order_max, c, fs, Nt, tcx[(size_t)r],
                         gain_over_nrays, energy.data(), seed);
        }
}

// ============================================================
// Host self-check (assert-based). Build with:
//   g++ -std=c++17 -x c++ -D__host__= -D__device__= -D__forceinline__=inline \
//       -DRT3D_SELFTEST raytrace3D.h -o rt_selftest && ./rt_selftest
// ============================================================
#ifdef RT3D_SELFTEST
#include <cassert>
int main(){
    // 4x4x4 m closed box centred at origin, source & receiver inside.
    // 6 quads -> 12 tris, all group 0. alpha = 0 (rigid) so energy is preserved.
    Scene3D Sc; Sc.scale=1.f; Sc.shift=v3(0,0,0);
    const float L=2.f;
    vec3 v[8]={v3(-L,-L,-L),v3(L,-L,-L),v3(L,L,-L),v3(-L,L,-L),
               v3(-L,-L, L),v3(L,-L, L),v3(L,L, L),v3(-L,L, L)};
    for(int i=0;i<8;++i) Sc.vertices.push_back(v[i]);
    int q[6][4]={{0,1,2,3},{4,5,6,7},{0,1,5,4},{2,3,7,6},{1,2,6,5},{0,3,7,4}};
    for(int f=0;f<6;++f){
        Tri t1,t2; t1.group_id=t2.group_id=0;
        t1.v0=q[f][0]; t1.v1=q[f][1]; t1.v2=q[f][2];
        t2.v0=q[f][0]; t2.v1=q[f][2]; t2.v2=q[f][3];
        Sc.tris.push_back(t1); Sc.tris.push_back(t2);
    }
    RTScene S = rt_build(Sc);
    assert(!S.nodes.empty());
    assert(S.max_gid==0);

    // sanity: a ray from the origin hits a wall at ~2 m
    RTView V; V.A=S.A.data();V.B=S.B.data();V.C=S.C.data();V.gid=S.gid.data();
    V.nodes=S.nodes.data();V.tri_order=S.tri_order.data();V.nnodes=(int)S.nodes.size();V.ntris=(int)S.A.size();
    V.alpha=nullptr;V.nalpha=0;V.src=nullptr;V.srcp=nullptr;V.nsrc=0;V.src_r2=0.f;
    float tw; int g; int hit=rt_closest(V,v3(0,0,0),v3(1,0,0),tw,g);
    assert(hit>=0); assert(std::fabs(tw-2.f)<1e-3f);

    // full trace: rigid box, source at origin-ish, receiver offset. Expect nonzero
    // late energy (a reverberant tail must exist in a closed rigid room).
    std::vector<float> alpha(1,0.f);
    std::vector<float> rec={0.5f,0.f,0.f}, src={-0.5f,0.f,0.f}, sp={1.f};
    const float c=343.f, fs=44100.f; const int Nt=(int)(0.2f*fs);
    std::vector<int> tcx(1, (int)(0.001f*fs));   // register after 1 ms
    std::vector<float> energy;
    rt_trace_tail(S, alpha.data(), 1, rec, src, sp,
                  0.1f, 0.2f, 4000, /*order_min*/1, /*order_max*/80,
                  c, fs, Nt, tcx, /*gain*/1.f, energy);
    double tot=0.0; int nz=0;
    for(float e : energy){ tot+=e; if(e>0.f) ++nz; }
    printf("[rt selftest] nodes=%zu tris=%d  nonzero samples=%d  total energy=%.6g\n",
           S.nodes.size(), (int)S.A.size(), nz, tot);
    assert(nz>0 && tot>0.0);            // the rigid room produced a tail

    // alpha=0.5 (the user's case): diffuse + shadow ray must give a DENSE tail
    std::vector<float> a5(1,0.5f); std::vector<float> e5;
    rt_trace_tail(S, a5.data(), 1, rec, src, sp, 0.1f,0.2f,4000,1,80,c,fs,Nt,tcx,1.f,e5);
    int nz5=0; double tot5=0.0; for(float e:e5){ if(e>0.f)++nz5; tot5+=e; }
    printf("[rt selftest] alpha=0.5 nonzero samples=%d total=%.6g (dense tail)\n", nz5, tot5);
    assert(nz5 > Nt/10);              // dense: a large fraction of samples filled
    assert(tot5 < tot);              // more absorption than rigid => less energy

    // absorbing walls (alpha=1) must kill the tail after the first bounce
    alpha[0]=1.f;
    rt_trace_tail(S, alpha.data(), 1, rec, src, sp, 0.1f,0.2f,4000,1,80,c,fs,Nt,tcx,1.f,energy);
    double tot2=0.0; for(float e:energy) tot2+=e;
    printf("[rt selftest] alpha=1 total energy=%.6g (expect ~0)\n", tot2);
    assert(tot2 < tot*1e-3);

    printf("[rt selftest] OK\n");
    return 0;
}
#endif
