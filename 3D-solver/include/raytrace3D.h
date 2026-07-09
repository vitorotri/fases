#pragma once
/*
 * Copyright (c) 2025 Vito Romanelli Tricanico
/*
raytrace3D.h
------------
Acoustical ray-tracer.

Rays leave the source on a Fibonacci-sphere pattern (near-uniform angular spread).
This is the DIFFUSE LATE TAIL that continues the ISM past its max reflection order.

Each ray that reaches a receiver contributes the ENERGY of an ISM-style arrival --
the square of
    p_ray[b] = ps * PROD_bounces sqrt(1 - alpha_b) * air_b / (4*pi*d_run)
where d_run is the total path length the ray actually travelled to the receiver,
sqrt(1-alpha_b) is the per-band normal-incidence pressure reflection coefficient
(from alpha_gb), air_b = exp(-m_air_b * d_run), and ps is that source's pressure.
Energies are accumulated (incoherent, random-phase addition -- the correct diffuse
model), NOT pressures (which would add coherently and over-inflate the dense tail).

p_rt[(b*Nr + r)*Nt + sample] ends up holding a per-band PRESSURE amplitude: the
detection sphere is crossed by ~n_path = Nrays*R^2/(4 d^2) rays per path, so at the
end rt_trace_late divides the accumulated energy by n_path (d = c*sample/fs is exact
per bin) and takes the square root. Nrays-INDEPENDENT; one arrival reproduces the ISM
image amplitude ps*PROD sqrt(1-alpha)/(4 pi d), K overlapping arrivals give sqrt(K)
times it. This is the standard energy-echogram -> noise tail synthesis.

Call sequence in main3D.cu:
    RTScene rtS = rt_build(sc);              // build once, before scene_free
    rt_trace_late(rtS, ..., p_rt);           // fill p_rt[8 * Nr * Nt] pressures
    // VALIDATION (amplitudes vs ISM): compare p_rt cells directly, or deposit
    //   p_ism += sqrt(e*e)          (magnitude, no random sign)
    // SYNTHESIS (audible diffuse tail): deposit
    //   p_ism += rt_sign(...) * sqrt(e*e)   (random phase; same magnitude)

Absorption is passed in flat as alpha_gb[group_id*8 + band] so this header stays
decoupled from the material/impedance code.
*/

#include "parsing3D.h"
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>

#ifndef RT_NBANDS
#define RT_NBANDS 8   // octave bands, matches the ISM section in main3D.cu
#endif

// ray: identical layout to raytracer.c (origin + unit direction). float to match
// the mesh (parsing3D uses float vec3), so no host<->device precision surprises.
typedef struct ray3 {
    float ox, oy, oz;   // origin
    float dx, dy, dz;   // unit direction
} ray3;

// reflected direction, vector form (raytracer.c::compute_reflected). Sign of N
// is irrelevant here (formula is symmetric in N), so we skip normal orientation.
inline static void rt_compute_reflected(vec3& ref, const vec3& inc, const vec3& N){
    const float k = 2.f*dot3(inc, N);
    ref = v3(inc.x - k*N.x, inc.y - k*N.y, inc.z - k*N.z);
    const float l = std::sqrt(norm2_3(ref));
    if(l > 1e-20f) ref = mul3(ref, 1.f/l);
}

// cheap integer hash -> pseudo-random sign, so the diffuse tail is incoherent
// noise (deterministic, reproducible). Swap for a real Poisson process if the
// tail statistics ever need to be exact.
inline static float rt_sign(unsigned int s){
    s ^= s>>16; s *= 0x7feb352du; s ^= s>>15; s *= 0x846ca68bu; s ^= s>>16;
    return (s & 1u) ? 1.f : -1.f;
}

// hash -> uniform float in [0,1). Same mixer as rt_sign; used for the stochastic
// specular/diffuse direction draw (below).
inline static float rt_u01(unsigned int s){
    s ^= s>>16; s *= 0x7feb352du; s ^= s>>15; s *= 0x846ca68bu; s ^= s>>16;
    return (float)(s >> 8) * (1.f/16777216.f);   // 24-bit mantissa
}

// cosine-weighted (Lambert) direction about surface normal N. u1,u2 in [0,1).
// This is how a diffusely-scattered ray CONTINUES to propagate -- carrying its
// full reflected energy in a new random direction, so multiple scattering (and
// the reverberant tail) survives instead of being dropped after one deposit.
inline static vec3 rt_lambert_dir(const vec3& N, float u1, float u2){
    const float r = std::sqrt(u1), phi = 6.283185307f*u2;
    const float x = r*std::cos(phi), y = r*std::sin(phi), z = std::sqrt(std::max(0.f,1.f-u1));
    vec3 a = (std::fabs(N.z) < 0.999f) ? v3(0,0,1) : v3(1,0,0);
    vec3 t = cross3(a, N); float tl = std::sqrt(norm2_3(t)); t = (tl>1e-8f)? mul3(t,1.f/tl):v3(1,0,0);
    vec3 bt = cross3(N, t);
    vec3 dir = v3(x*t.x+y*bt.x+z*N.x, x*t.y+y*bt.y+z*N.y, x*t.z+y*bt.z+z*N.z);
    float dl = std::sqrt(norm2_3(dir));
    return (dl>1e-8f)? mul3(dir,1.f/dl) : N;
}

// Expand a single global scattering coefficient g in [0,1] into 8 octave-band values
// (63..8000 Hz) by linear interpolation of an ODEON-style reference table. Every column
// is monotonic in g and every row rises with frequency, so the interpolation reproduces
// the reference exactly at the anchors and stays monotonic between them. g is a roughness
// index (a selector), not "s at some frequency".
inline static void rt_scatter_bands(float g, float s_out[RT_NBANDS]){
    static const int NA = 10;
    static const float ga[NA] = {0.01f,0.02f,0.05f,0.10f,0.20f,0.30f,0.50f,0.80f,0.90f,1.00f};
    static const float tab[NA][RT_NBANDS] = {
        {0.010f,0.010f,0.010f,0.015f,0.020f,0.035f,0.040f,0.040f},
        {0.010f,0.010f,0.010f,0.015f,0.035f,0.080f,0.090f,0.100f},
        {0.010f,0.010f,0.010f,0.025f,0.080f,0.200f,0.250f,0.300f},
        {0.010f,0.010f,0.020f,0.050f,0.150f,0.260f,0.310f,0.350f},
        {0.010f,0.010f,0.030f,0.110f,0.290f,0.370f,0.420f,0.460f},
        {0.010f,0.020f,0.040f,0.190f,0.420f,0.490f,0.530f,0.570f},
        {0.010f,0.030f,0.100f,0.380f,0.620f,0.740f,0.770f,0.800f},
        {0.020f,0.050f,0.240f,0.710f,0.860f,0.880f,0.890f,0.900f},
        {0.040f,0.110f,0.680f,0.860f,0.920f,0.930f,0.940f,0.950f},
        {0.100f,0.370f,0.900f,1.000f,1.000f,1.000f,1.000f,1.000f},
    };
    if(g <= ga[0]){    for(int b=0;b<RT_NBANDS;++b) s_out[b]=tab[0][b];    return; }
    if(g >= ga[NA-1]){ for(int b=0;b<RT_NBANDS;++b) s_out[b]=tab[NA-1][b]; return; }
    int i=0; while(i<NA-1 && g>ga[i+1]) ++i;              // ga[i] <= g <= ga[i+1]
    const float t = (g-ga[i])/(ga[i+1]-ga[i]);
    for(int b=0;b<RT_NBANDS;++b) s_out[b] = tab[i][b] + t*(tab[i+1][b]-tab[i][b]);
}

// Fibonacci-sphere direction i of N (near-uniform on the unit sphere), origin o.
inline static ray3 rt_fib_ray(const vec3& o, int i, int N){
    const float ga = 2.399963229728653f;                 // golden angle, pi*(3-sqrt5)
    const float z  = 1.f - (2.f*(float)i + 1.f)/(float)N; // z in (-1,1)
    const float rho= std::sqrt(std::max(0.f, 1.f - z*z));
    const float phi= ga*(float)i;
    ray3 r; r.ox=o.x; r.oy=o.y; r.oz=o.z;
    r.dx=rho*std::cos(phi); r.dy=rho*std::sin(phi); r.dz=z;
    return r;
}

// ============================================================
// BVH scene
// ============================================================
struct BVHNode {
    vec3 bmin = v3(0,0,0), bmax = v3(0,0,0);
    int  left  = -1;    // internal: left child index; leaf: -1
    int  right = -1;    // internal: right child index; leaf: -1
    int  first = 0;     // leaf: first prim in the ordered arrays
    int  count = 0;     // leaf: prim count (>0); internal: 0
};

struct RTScene {
    // per-triangle, in BVH-leaf order (transformed to solver space once)
    std::vector<vec3> A, B, C;
    std::vector<int>  group;
    std::vector<BVHNode> nodes;
    vec3  shift = v3(0,0,0);   // endpoint transform, matches vertex_transformed
    float scale = 1.f;
    int   ngroups = 0;
};

// ---- build helpers (host only; one-shot, not perf-critical) ----
namespace rt_detail {
    struct Prim { vec3 a,b,c,cent,bmin,bmax; int group; };

    inline void expand(vec3& mn, vec3& mx, const vec3& p){
        mn.x=std::min(mn.x,p.x); mn.y=std::min(mn.y,p.y); mn.z=std::min(mn.z,p.z);
        mx.x=std::max(mx.x,p.x); mx.y=std::max(mx.y,p.y); mx.z=std::max(mx.z,p.z);
    }

    // recursive median split on the longest centroid axis; reorders idx[first,last).
    // Returns this node's index. Pushing self before recursing means child indices
    // are whatever build() returns -- we store both explicitly (no left+1 assumption).
    inline int build(std::vector<BVHNode>& nodes, std::vector<int>& idx,
                     const std::vector<Prim>& prim, int first, int last)
    {
        const int self = (int)nodes.size();
        nodes.push_back(BVHNode{});
        vec3 bmin=v3(FLT_MAX,FLT_MAX,FLT_MAX), bmax=v3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        vec3 cmin=bmin, cmax=bmax;
        for(int i=first;i<last;++i){
            expand(bmin,bmax,prim[(size_t)idx[i]].bmin);
            expand(bmin,bmax,prim[(size_t)idx[i]].bmax);
            expand(cmin,cmax,prim[(size_t)idx[i]].cent);
        }
        nodes[(size_t)self].bmin=bmin; nodes[(size_t)self].bmax=bmax;

        const int n = last-first;
        if(n <= 4){                         // leaf
            nodes[(size_t)self].left=-1; nodes[(size_t)self].right=-1;
            nodes[(size_t)self].first=first; nodes[(size_t)self].count=n;
            return self;
        }
        const float ex=cmax.x-cmin.x, ey=cmax.y-cmin.y, ez=cmax.z-cmin.z;
        const int axis = (ex>=ey && ex>=ez) ? 0 : (ey>=ez ? 1 : 2);
        const int mid = first + n/2;
        std::nth_element(idx.begin()+first, idx.begin()+mid, idx.begin()+last,
            [&](int u,int v){
                const vec3&cu=prim[(size_t)u].cent; const vec3&cv=prim[(size_t)v].cent;
                const float ku=(axis==0?cu.x:axis==1?cu.y:cu.z);
                const float kv=(axis==0?cv.x:axis==1?cv.y:cv.z);
                return ku<kv;
            });
        const int L = build(nodes, idx, prim, first, mid);
        const int R = build(nodes, idx, prim, mid,   last);
        nodes[(size_t)self].left=L; nodes[(size_t)self].right=R; nodes[(size_t)self].count=0;
        return self;
    }
}

// Build the BVH scene from a parsed mesh. Copies transformed triangles, so the
// caller may scene_free() right after (main3D.cu does exactly this).
inline static RTScene rt_build(const Scene3D& Sc){
    using namespace rt_detail;
    RTScene S;
    S.shift = Sc.shift; S.scale = Sc.scale;
    int max_gid=-1;
    std::vector<Prim> prim; prim.reserve(Sc.tris.size());
    for(const Tri& t : Sc.tris){
        Prim p;
        p.a = Sc.vertex_transformed(t.v0);
        p.b = Sc.vertex_transformed(t.v1);
        p.c = Sc.vertex_transformed(t.v2);
        p.cent = mul3(add3(add3(p.a,p.b),p.c), 1.f/3.f);
        p.bmin = p.a; p.bmax = p.a; expand(p.bmin,p.bmax,p.b); expand(p.bmin,p.bmax,p.c);
        p.group = t.group_id;
        max_gid = std::max(max_gid, t.group_id);
        prim.push_back(p);
    }
    S.ngroups = max_gid + 1;
    if(prim.empty()) return S;

    std::vector<int> idx(prim.size());
    for(size_t i=0;i<idx.size();++i) idx[i]=(int)i;
    S.nodes.reserve(2*prim.size());
    build(S.nodes, idx, prim, 0, (int)prim.size());

    // ordered triangle arrays follow the reordered idx (leaf ranges are contiguous)
    S.A.resize(prim.size()); S.B.resize(prim.size());
    S.C.resize(prim.size()); S.group.resize(prim.size());
    for(size_t i=0;i<idx.size();++i){
        S.A[i]=prim[(size_t)idx[i]].a; S.B[i]=prim[(size_t)idx[i]].b;
        S.C[i]=prim[(size_t)idx[i]].c; S.group[i]=prim[(size_t)idx[i]].group;
    }
    return S;
}

// slab test against [bmin,bmax] within [tmin,tmax]
inline static bool rt_slab(const vec3& bmin, const vec3& bmax, const vec3& ro,
                           const vec3& inv, float tmin, float tmax){
    float t0=tmin, t1=tmax, a, b;
    a=(bmin.x-ro.x)*inv.x; b=(bmax.x-ro.x)*inv.x; if(a>b) std::swap(a,b); t0=std::max(t0,a); t1=std::min(t1,b);
    a=(bmin.y-ro.y)*inv.y; b=(bmax.y-ro.y)*inv.y; if(a>b) std::swap(a,b); t0=std::max(t0,a); t1=std::min(t1,b);
    a=(bmin.z-ro.z)*inv.z; b=(bmax.z-ro.z)*inv.z; if(a>b) std::swap(a,b); t0=std::max(t0,a); t1=std::min(t1,b);
    return t1>=t0;
}

// closest facet hit. dir is unit => t_out is meters. tri_out is the ordered-array
// prim index (use S.group[tri_out], S.A/B/C[tri_out]). raytracer.c::intersect, BVH'd.
inline static bool rt_closest_hit(const RTScene& S, const ray3& r, float tmin,
                                  float& t_out, int& tri_out){
    if(S.nodes.empty()) return false;
    const vec3 ro=v3(r.ox,r.oy,r.oz), rd=v3(r.dx,r.dy,r.dz);
    const vec3 inv=v3(1.f/rd.x, 1.f/rd.y, 1.f/rd.z);
    float best=FLT_MAX; int bi=-1;
    int stack[64]; int sp=0; stack[sp++]=0;
    while(sp){
        const BVHNode& nd = S.nodes[(size_t)stack[--sp]];
        if(!rt_slab(nd.bmin, nd.bmax, ro, inv, tmin, best)) continue;
        if(nd.count>0){
            for(int k=0;k<nd.count;++k){
                const int p=nd.first+k; float tt;
                if(ray_triangle_intersect(ro,rd,S.A[(size_t)p],S.B[(size_t)p],S.C[(size_t)p],tmin,tt) && tt<best){
                    best=tt; bi=p;
                }
            }
        } else {
            if(sp<62){ stack[sp++]=nd.left; stack[sp++]=nd.right; }
        }
    }
    if(bi<0) return false;
    t_out=best; tri_out=bi; return true;
}

// unit normal of ordered triangle p
inline static vec3 rt_tri_normal(const RTScene& S, int p){
    vec3 n = cross3(sub3(S.B[(size_t)p],S.A[(size_t)p]), sub3(S.C[(size_t)p],S.A[(size_t)p]));
    const float l = std::sqrt(norm2_3(n));
    return (l>1e-20f) ? mul3(n,1.f/l) : v3(0,0,1);
}

// distance^2 from point q to segment [p0,p1], with the along-segment length at the
// closest approach returned in s_out (clamped to the segment).
inline static float rt_point_seg_dist2(const vec3& q, const vec3& p0, const vec3& p1, float& s_out){
    const vec3 d = sub3(p1,p0);
    const float L2 = norm2_3(d);
    float t = (L2>1e-20f) ? dot3(sub3(q,p0), d)/L2 : 0.f;
    t = std::max(0.f, std::min(1.f, t));
    const vec3 c = add3(p0, mul3(d,t));
    s_out = t*std::sqrt(L2);
    return norm2_3(sub3(q,c));
}

// ============================================================
// Late diffuse tail: shoot Fibonacci rays from each source, bounce specularly
// so no recursion is needed, and whenever a ray segment passes within
// recv_radius of a receiver ACCUMULATE ENERGY -- the square of the ISM-style per-ray
// pressure  p_ray[b] = ps * PROD sqrt(1-alpha_b) * air / (4*pi*d_run).
// Energy adds incoherently (random phase), which is the correct diffuse-field model;
// distinct arrivals in a bin add as energy, not as a coherent (all-positive) pressure
// sum. At the end the buffer is normalized (per-path multiplicity) and sqrt'd, so
// p_rt holds a PRESSURE amplitude: one arrival -> the ISM image amplitude, K arrivals
// -> sqrt(K) times it (the physical random-phase level). Deposit into the RIR with a
// per-sample random sign (rt_sign) for the audible tail. This is the standard
// energy-echogram -> noise synthesis (Vorlander/Kuttruff/Moorer).
//
// Outputs TWO energy buffers (NOT sqrt'd), each sized RT_NBANDS*Nr*Nt, indexed
// [(b*Nr + r)*Nt + sample]:  e_spec = specular (image-source) energy, e_diff = diffuse
// (scattered) energy. main combines and roots them to pressure, gating the specular at
// the ISM crossover:  t<=t_cross -> p += sign*sqrt(e_diff);  t>t_cross -> p +=
// sign*sqrt(e_spec + e_diff).  alpha_gb / sctr_gb are per group per band (gid*8+b), null
// => rigid / no scattering. Source/receiver positions are RAW (world), transformed like
// the walls so times line up with the ISM.
// ============================================================
inline static void rt_trace_late(const RTScene& S,
                                 const std::vector<float>& src_arr,
                                 const std::vector<float>& src_pressure,
                                 const std::vector<float>& rec_arr,
                                 const float* alpha_gb,   // [ngroups*8] absorption per group per band, or nullptr
                                 const float* m_air,      // [8] PRESSURE air-attenuation coeff (Np/m, halved); nullptr => none
                                 const float* sctr_gb,    // [ngroups*8] scattering per group per band, or nullptr => none
                                 const float* sctr_g,     // [ngroups] GLOBAL scattering per group (the draw), or nullptr => none
                                 float c, float fs, int Nt,
                                 int Nrays, int max_order, float recv_radius,
                                 std::vector<float>& e_spec,   // OUT: specular energy (NOT sqrt'd)
                                 std::vector<float>& e_diff){  // OUT: diffuse  energy (NOT sqrt'd)

    const int Ns=(int)(src_arr.size()/3), Nr=(int)(rec_arr.size()/3);
    const size_t NBUF=(size_t)RT_NBANDS*(size_t)Nr*(size_t)Nt;
    e_spec.assign(NBUF, 0.f);
    e_diff.assign(NBUF, 0.f);
    if(Ns==0 || Nr==0 || Nrays<=0 || S.nodes.empty() || recv_radius<=0.f) return;

    auto xform = [&](float x,float y,float z){
        return v3((x+S.shift.x)*S.scale, (y+S.shift.y)*S.scale, (z+S.shift.z)*S.scale);
    };
    std::vector<vec3> rec(Nr);
    for(int r=0;r<Nr;++r) rec[(size_t)r]=xform(rec_arr[(size_t)(3*r)],rec_arr[(size_t)(3*r+1)],rec_arr[(size_t)(3*r+2)]);

    const float R2 = recv_radius*recv_radius;
    const float t_end = (float)Nt/fs;      // stop rays past the impulse-response length
    const float Rp_min = 1e-4f;            // reflection floor: kill dead rays
    const float RT_INVPI = 0.318309886183791f;

    // PARTICLE / ENERGY estimator anchored to ps (ISM amplitude convention). Each ray is a
    // power chunk w_ray = ps^2/(4 pi Nrays). SPECULAR: one path found by n_path =
    // Nrays*r_d^2/(4 d^2) sphere crossings deposits a distance-independent energy per
    // crossing, so crossings sum to the ISM energy (ps Rp air/(4 pi d))^2. NEXT-EVENT
    // ESTIMATION (diffuse): a bounce scatters s[b] of the reflected power toward each visible
    // receiver as a Lambert secondary source, deposited as the POINT-receiver intensity
    //   e_s = w_ray * s * Rrefl^2 * air^2 * cos(theta) / (pi d_rec^2).
    // This is R-INDEPENDENT, matching the specular's ISM-anchored point convention (a
    // detection-sphere solid-angle capture ~R^2 would be the energy-histogram convention
    // and inconsistent here). Same w_ray currency as the specular deposit -> the s/(1-s)
    // split is honoured at the receiver. PROPAGATION: the ray carries the FULL reflected
    // power (no (1-s) drain) and scattering only redirects it (stochastic draw with the
    // global s, prob s -> Lambert), so multiple scattering survives and T60 stays
    // absorption-governed for any s. The specular (1-s)^k attenuation emerges from the
    // fraction of rays that never took a diffuse bounce -- see the detection guard below.
	// for each source
    for(int s = 0; s < Ns; s++){
        const vec3 o0 = xform(src_arr[(size_t)(3*s)],src_arr[(size_t)(3*s+1)],src_arr[(size_t)(3*s+2)]);
        const float ps = (s<(int)src_pressure.size()) ? src_pressure[(size_t)s] : 1.f;
        const float Kbase = ps*ps/(4.f*M_PI*(float)Nrays);   // w_ray = ps^2/(4 pi Nrays)
        const float Kspec = Kbase*RT_INVPI/R2;               // per specular crossing => ps^2/(4 pi^2 Nrays r_d^2)

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for(int ri = 0; ri < Nrays; ri++){			// breadth: independent samples
            ray3 r = rt_fib_ray(o0, ri, Nrays);		// depth: one linear path, in place
            float Rp[RT_NBANDS]; for(int b = 0; b < RT_NBANDS; b++) Rp[b]=1.f; // amplitude reflection product
            float d_travel = 0.f;            		// path length travelled so far [m]
            bool  scattered = false;         		// has this ray taken a diffuse bounce yet?

            for(int bounce = 0; bounce <= max_order; bounce++){
                float th; int tri;
                const bool hit = rt_closest_hit(S, r, 1e-4f, th, tri);
                // segment end: hit point, or a far point if the ray escapes
                const float seg = hit ? th : 1e4f;
                const vec3 o = v3(r.ox,r.oy,r.oz);
                const vec3 d = v3(r.dx,r.dy,r.dz);
                const vec3 p1 = add3(o, mul3(d, seg));

                // ---- SPECULAR detection on segment [o,p1] -> e_spec (distance-independent
                // energy per detection-sphere crossing). ONLY while the ray is still a pure
                // specular (image-source) path: once it has taken a diffuse bounce its energy
                // reaches the receiver via the next-event estimation below, so detecting it here too
                // would double-count. The specular (1-s)^k attenuation is NOT applied to the
                // energy -- it emerges from the fraction of rays that survive as specular,
                // since each diffuse draw (prob s) removes a ray from this population.
                for(int rr = 0; !scattered && rr < Nr; rr++){
                    float s_ca;
                    if(rt_point_seg_dist2(rec[(size_t)rr], o, p1, s_ca) <= R2){
                        const float d_run = d_travel + s_ca; // total path to the receiver
                        const float tt = d_run/c;
                        if(tt < t_end && d_run > 1e-4f){
                            const int n0 = (int)(tt*fs);
                            if(n0 >= 0 && n0 < Nt){
                                for(int b = 0;b < RT_NBANDS; b++){
                                    const float air2 = m_air ? std::exp(-2.f*m_air[b]*d_run) : 1.f; // air^2 (energy)
                                    const float e = Kspec * Rp[b]*Rp[b] * air2;
                                    float* cell = &e_spec[((size_t)b*Nr + rr)*Nt + n0];
                                    #ifdef _OPENMP
                                    #pragma omp atomic
                                    #endif
                                    *cell += e;
                                }
                            }
                        }
                    }
                }

                if(!hit) break;              // escaped the geometry
                d_travel += seg;
                if(d_travel/c >= t_end) break;

                // ---- reflection coefficient R_oblique(alpha, cos_theta) per band. Rrefl is
                // the amplitude AFTER the wall, BEFORE the scatter split.
                const int gid = S.group[(size_t)tri];
                const vec3 N = rt_tri_normal(S, tri);
                const float ct = std::fabs(dot3(d, N));   // cos of the incidence angle
                float Rrefl[RT_NBANDS];
                for(int b = 0;b < RT_NBANDS; b++){
                    float al = (alpha_gb && gid>=0 && gid<S.ngroups) ? alpha_gb[(size_t)gid*RT_NBANDS+b] : 0.f;
                    al = std::max(0.f, std::min(0.999f, al));
                    Rrefl[b] = Rp[b] * std::fabs(R_oblique(al, ct));
                }

                // ---- NEXT-EVENT ESTIMATION -> e_diff: deterministically connect the bounce
                // point B=p1 to each visible receiver and deposit the scattered contribution
                // s[b] of the reflected power as a Lambert secondary source (point-receiver
                // intensity). This is the low-variance diffuse DETECTOR; the ray's own energy
                // is untouched and keeps propagating via the stochastic draw below.
                if(sctr_gb && gid>=0 && gid<S.ngroups){
                    const vec3 No = (dot3(N, d) > 0.f) ? mul3(N,-1.f) : N;  // face the incoming side
                    for(int rr = 0; rr < Nr; rr++){
                        const vec3 br = sub3(rec[(size_t)rr], p1);
                        const float d2 = norm2_3(br);
                        if(d2 < 1e-6f) continue;
                        const float drec = std::sqrt(d2);
                        const vec3 u = mul3(br, 1.f/drec);
                        const float cosr = dot3(No, u);                    // cos(theta), Lambert weight
                        if(cosr <= 0.f) continue;                          // receiver behind the surface
                        ray3 sr; sr.ox=p1.x+No.x*1e-4f; sr.oy=p1.y+No.y*1e-4f; sr.oz=p1.z+No.z*1e-4f;
                        sr.dx=u.x; sr.dy=u.y; sr.dz=u.z;
                        float sh_t; int sh_tri;
                        if(rt_closest_hit(S, sr, 1e-4f, sh_t, sh_tri) && sh_t < drec-1e-3f) continue; // occluded
                        const float d_run = d_travel + drec;               // source -> B -> receiver
                        if(d_run/c >= t_end) continue;
                        const int n0 = (int)((d_run/c)*fs);
                        if(n0 < 0 || n0 >= Nt) continue;
                        // Lambert point-receiver INTENSITY: cos(theta)/(pi d_rec^2). This is
                        // Recv_radius-INDEPENDENT, matching the specular's ISM-anchored point convention
                        // (the specular bakes 1/r_d^2 into Kspec, so it too is Recv_radius-independent).
                        // A detection-sphere solid-angle capture (~R^2) would be the energy-
                        // histogram convention and is INCONSISTENT with the specular here.
                        const float geo = cosr * RT_INVPI / d2;   // cos(theta)/(pi d_rec^2)
                        for(int b = 0;b < RT_NBANDS; b++){
                            const float sb = std::max(0.f, std::min(1.f, sctr_gb[(size_t)gid*RT_NBANDS+b]));
                            if(sb <= 0.f) continue;
                            const float air2 = m_air ? std::exp(-2.f*m_air[b]*d_run) : 1.f;
                            const float e = Kbase * sb * Rrefl[b]*Rrefl[b] * air2 * geo; // s(1-a)
                            float* cell = &e_diff[((size_t)b*Nr + rr)*Nt + n0];
                            #ifdef _OPENMP
                            #pragma omp atomic
                            #endif
                            *cell += e;
                        }
                    }
                }

                // ---- ENERGY CONSERVATION: the ray carries the FULL reflected power (1-alpha)
                // forward -- NO (1-s) drain. Scattering changes only the DIRECTION, not the
                // energy, so scattered energy keeps bouncing (multiple scattering) and the tail
                // decays at the absorption-governed rate regardless of s. (The old (1-s) drain
                // dropped ~s of the energy every bounce, collapsing T60 as s grew.)
                float alive=0.f;
                for(int b = 0;b < RT_NBANDS; b++){ Rp[b] = Rrefl[b]; alive = std::max(alive, Rp[b]); }
                if(alive < Rp_min) break;

                // ---- STOCHASTIC direction: draw once with the GLOBAL s (one scalar per surface,
                // so no per-band trajectory conflict). Prob s -> diffuse (Lambert) reflection,
                // else specular. Over many rays a fraction s scatter and (1-s) stay specular,
                // reproducing the correct s/(1-s) split in aggregate while every ray keeps its
                // full energy. The per-band s[b] above weights only the diffuse-rain deposit.
                const unsigned int seed = ((unsigned)((s*Nrays+ri))*2654435761u) ^ ((unsigned)bounce*40503u);
                const float sg = (sctr_g && gid>=0 && gid<S.ngroups)
                               ? std::max(0.f, std::min(1.f, sctr_g[(size_t)gid])) : 0.f;
                vec3 ref;
                // The concept below can be found in Hanna Autio's thesis and in Brandão et al (2019, Aachen)
                if(sg > 0.f && rt_u01(seed) < sg){
                    const vec3 No = (dot3(N, d) > 0.f) ? mul3(N,-1.f) : N;   // outward-facing normal
                    ref = rt_lambert_dir(No, rt_u01(seed^0x9e3779b9u), rt_u01(seed^0x85ebca6bu));
                    scattered = true;                                        // no more specular sphere deposits
                } else {
                    rt_compute_reflected(ref, d, N);
                }
                r.ox=p1.x + ref.x*1e-4f; r.oy=p1.y + ref.y*1e-4f; r.oz=p1.z + ref.z*1e-4f;
                r.dx=ref.x; r.dy=ref.y; r.dz=ref.z;
            }
        }
    }
    // e_spec / e_diff hold physical ENERGY (not rooted). main sums the required channels
    // per bin and takes sqrt -> pressure (see the header note above).
}

// ============================================================
// Visualization: trace Nrays Fibonacci rays from src_world up to max_order bounces
// (purely specular, no bands/energy/receiver) and write every segment to `file`, one
// per line: "src_id ray_id x0 y0 z0 x1 y1 z1 order" (ray_id unique across all sources).
// Traces every source in `srcs`. Points are un-transformed to WORLD coords
// so they overlay the raw OBJ in the GUI. `order` is the bounce index (0 = source ->
// first hit, 1 = after 1st reflection, ...) for per-order colouring. Escaped rays get
// one stub segment of the scene-diagonal length so the launch direction stays visible.
// ============================================================
inline static void rt_dump_paths(const RTScene& S, const std::vector<vec3>& srcs,
                                 int Nrays, int max_order, const char* file){
    std::FILE* f = std::fopen(file, "w");
    if(!f) return;
    if(S.nodes.empty() || srcs.empty()){ std::fclose(f); return; }

    // world <-> solver: solver = (world + shift)*scale ; world = solver/scale - shift
    auto to_solver = [&](const vec3& w){
        return v3((w.x+S.shift.x)*S.scale, (w.y+S.shift.y)*S.scale, (w.z+S.shift.z)*S.scale); };
    auto to_world = [&](const vec3& p){
        return v3(p.x/S.scale - S.shift.x, p.y/S.scale - S.shift.y, p.z/S.scale - S.shift.z); };

    const vec3 bmn = S.nodes[0].bmin, bmx = S.nodes[0].bmax;
    const float stub = 0.25f * std::sqrt(norm2_3(sub3(bmx, bmn)));  // escaped-ray stub length

    int uid = 0;   // globally-unique ray id across all sources
    for(size_t si = 0; si < srcs.size(); si++){
        const vec3 o0 = to_solver(srcs[si]);
        for(int ri = 0; ri < Nrays; ri++, uid++){
            ray3 r = rt_fib_ray(o0, ri, Nrays);
            vec3 o = v3(r.ox, r.oy, r.oz);
            for(int b = 0; b < max_order; b++){
                const vec3 d = v3(r.dx, r.dy, r.dz);
                float th; int tri;
                const bool hit = rt_closest_hit(S, r, 1e-4f, th, tri);
                const vec3 p = hit ? add3(o, mul3(d, th)) : add3(o, mul3(d, stub));
                const vec3 ow = to_world(o), pw = to_world(p);
                const int seg_order = hit ? b : -1;   // -1 flags an escaped (leaked) stub
                std::fprintf(f, "%d %d %.6g %.6g %.6g %.6g %.6g %.6g %d\n",
                             (int)si, uid, ow.x,ow.y,ow.z, pw.x,pw.y,pw.z, seg_order);
                if(!hit) break;                       // escaped: stub drawn, stop
                vec3 N = rt_tri_normal(S, tri);
                vec3 ref; rt_compute_reflected(ref, d, N);
                o = v3(p.x + ref.x*1e-4f, p.y + ref.y*1e-4f, p.z + ref.z*1e-4f);
                r.ox=o.x; r.oy=o.y; r.oz=o.z; r.dx=ref.x; r.dy=ref.y; r.dz=ref.z;
            }
        }
    }
    std::fclose(f);
}
