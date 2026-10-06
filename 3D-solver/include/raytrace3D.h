#pragma once
/*
 * Copyright (c) 2025 Vito Romanelli Tricanico
/*
raytrace3D.h
------------
Acoustical ray-tracer.

Rays leave the source on a Fibonacci-sphere pattern (near-uniform angular spread).
The RT covers the specular paths ABOVE the ISM order (Q.order_used) and all scattered
energy (on-off scattering + diffuse rain), at all times.

Each detection deposits the ENERGY of an ISM-style arrival, the square of
    p_ray[b] = ps * PROD_bounces |R_oblique(alpha_b, cos_theta)| * air_b / (4*pi*d_run)
where d_run is the total path length to the receiver, R_oblique the angle-dependent
pressure reflection coefficient (alpha from alpha_gb), air_b = exp(-m_air_b * d_run),
ps that source's pressure. Energies add (incoherent), NOT pressures (which would add
coherently and over-inflate the dense tail).

Outputs are ENERGY echograms e_spec / e_diff [(b*Nr + r)*Nt + sample], not rooted. The
per-crossing energy Kspec already divides by the crossings per path (n_path =
Nrays*R^2/(4 d^2)), so the sums are Nrays- and R-independent: one arrival gives the ISM
image energy, K overlapping arrivals K times it. main3D sums, takes sqrt and applies a
random sign (energy-echogram -> noise synthesis).

Call sequence in main3D.cu:
    RTScene rtS = rt_build(sc);              // build once, before scene_free
    rt_trace_late(rtS, ..., e_spec, e_diff, Q.order_used);   // energies [8 * Nr * Nt]
    p += 4*pi * rt_sign(...) * sqrt(e_spec + e_diff)          // per bin (sqrt only, no sign, to validate)

Absorption is passed in flat as alpha_gb[group_id*8 + band] so this header stays
decoupled from the material/impedance code.
*/

#include "parsing3D.h"
#include <vector>
#include <string>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <numeric>

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
// This is how a diffusely-scattered ray CONTINUES to propagate, carrying its
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

// OPTION, off (see main3D.cu): the RT scatters with ONE value per surface (on-off, one
// trajectory per ray), so a per-band table can only feed the ISM, and then the early
// energy per band no longer sums to 1. Kept for reference.
//  Expand a single global scattering coefficient g in [0,1] into 8 octave-band values
//  (63..8000 Hz) by linear interpolation of an ODEON-style reference table. Every column
//  is monotonic in g and every row rises with frequency, so the interpolation reproduces
//  the reference exactly at the anchors and stays monotonic between them. g is a roughness
//  index (a selector), not "s at some frequency".
//inline static void rt_scatter_bands(float g, float s_out[RT_NBANDS]){
//    static const int NA = 10;
//    static const float ga[NA] = {0.01f,0.02f,0.05f,0.10f,0.20f,0.30f,0.50f,0.80f,0.90f,1.00f};
//    static const float tab[NA][RT_NBANDS] = {
//        {0.010f,0.010f,0.010f,0.015f,0.020f,0.035f,0.040f,0.040f},
//        {0.010f,0.010f,0.010f,0.015f,0.035f,0.080f,0.090f,0.100f},
//        {0.010f,0.010f,0.010f,0.025f,0.080f,0.200f,0.250f,0.300f},
//        {0.010f,0.010f,0.020f,0.050f,0.150f,0.260f,0.310f,0.350f},
//        {0.010f,0.010f,0.030f,0.110f,0.290f,0.370f,0.420f,0.460f},
//        {0.010f,0.020f,0.040f,0.190f,0.420f,0.490f,0.530f,0.570f},
//        {0.010f,0.030f,0.100f,0.380f,0.620f,0.740f,0.770f,0.800f},
//        {0.020f,0.050f,0.240f,0.710f,0.860f,0.880f,0.890f,0.900f},
//        {0.040f,0.110f,0.680f,0.860f,0.920f,0.930f,0.940f,0.950f},
//        {0.100f,0.370f,0.900f,1.000f,1.000f,1.000f,1.000f,1.000f},
//    };
//    if(g <= ga[0]){    for(int b=0;b<RT_NBANDS;++b) s_out[b]=tab[0][b];    return; }
//    if(g >= ga[NA-1]){ for(int b=0;b<RT_NBANDS;++b) s_out[b]=tab[NA-1][b]; return; }
//    int i=0; while(i<NA-1 && g>ga[i+1]) ++i;              // ga[i] <= g <= ga[i+1]
//    const float t = (g-ga[i])/(ga[i+1]-ga[i]);
//    for(int b=0;b<RT_NBANDS;++b) s_out[b] = tab[i][b] + t*(tab[i+1][b]-tab[i][b]);
//}

// t_cross: statistical time at which an order-N ISM is "complete". INFORMATION ONLY (main3D prints
// it); the ISM/RT handover is by reflection order (rt_trace_late's ism_order), not by time.
// An order-N reflection travels ~N mean free paths (mfp = 4V/S): t_complete = max_order * mfp / c.
// Returned as a SAMPLE INDEX (fs * t). Option, off: Polack mixing time t_mix = sqrt(N_OVERLAP * V /
// (4 pi c^3 DT_AUD)) (10 reflections within the 24 ms auditory window), t_cross = min(t_mix, t_complete).
#ifndef RT_TCROSS_NOVERLAP
#define RT_TCROSS_NOVERLAP 10.f    // reflections overlapping within the auditory window
#endif
#ifndef RT_TCROSS_DT_AUD
#define RT_TCROSS_DT_AUD   0.024f  // auditory temporal resolution [s]
#endif
inline static int rt_t_cross(float V, float S, float c, float fs, int max_order){
	// Polack
    //const float t_mix = std::sqrt(RT_TCROSS_NOVERLAP * V / (4.f*(float)M_PI * c*c*c * RT_TCROSS_DT_AUD)); // [s]
    const float mfp   = (S > 0.f) ? 4.0*V/S : 0.f; // mean free path [m] Kuttruff eq 5.14
    const float t_complete = (float)max_order * mfp / c; // [s] use equation above in Kuttruff eq 5.12
    //return (int)(fs * std::min(t_mix, t_complete));
    return (int)(fs*t_complete);
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
    // are whatever build() returns; we store both explicitly (no left+1 assumption).
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
// Shoot Fibonacci rays from each source; each ray is one linear path (specular or, with
// probability s, Lambert direction at each bounce: no recursion). Whenever a segment passes within
// recv_radius of a receiver, ACCUMULATE ENERGY, the square of the ISM-style per-ray pressure
//     p_ray[b] = ps * PROD |R_oblique(alpha_b, cos)| * air / (4*pi*d_run),
// plus the diffuse rain at every bounce. Energies add incoherently (random phase), the
// diffuse-field model (Vorlander/Kuttruff/Moorer).
//
// Outputs TWO energy buffers (NOT sqrt'd), each sized RT_NBANDS*Nr*Nt, indexed
// [(b*Nr + r)*Nt + sample]: e_spec = still-specular rays, orders > ism_order only;
// e_diff = rays scattered at least once + the rain. Both enter main3D at all times:
// p += 4*pi * sign * sqrt(e_spec + e_diff). alpha_gb per group per band (gid*8+b), null => rigid;
// sctr_g per group, null => no scattering. Source/receiver positions are RAW (world),
// transformed like the walls so times line up with the ISM.
// ============================================================
inline static void rt_trace_late(const RTScene& S,
                                 const std::vector<float>& src_arr,
                                 const std::vector<float>& src_pressure,
                                 const std::vector<float>& rec_arr,
                                 const float* alpha_gb,   // [ngroups*8] absorption per group per band, or nullptr
                                 const float* m_air,      // [8] PRESSURE air-attenuation coeff (Np/m, halved); nullptr => none
                                 const float* sctr_g,     // [ngroups] GLOBAL scattering per group (the draw), or nullptr => none
                                 float c, float fs, int Nt,
                                 int Nrays, int max_order, float recv_radius,
                                 std::vector<float>& e_spec,   // OUT: specular energy (NOT sqrt'd)
                                 std::vector<float>& e_diff,   // OUT: diffuse  energy (NOT sqrt'd)
                                 int ism_order = -1){          // specular paths of order <= ism_order are the ISM's
                                                               // (skipped here); -1 = none skipped (old behaviour)

    const int Ns=(int)(src_arr.size()/3), Nr=(int)(rec_arr.size()/3); // #sources, #receivers (arrays are xyz triplets)
    const size_t NBUF=(size_t)RT_NBANDS*(size_t)Nr*(size_t)Nt;        // size of each output buffer: band * receiver * time
    e_spec.assign(NBUF, 0.f);   // OUT: specular energy echogram, index [(b*Nr + r)*Nt + sample]
    e_diff.assign(NBUF, 0.f);   // OUT: diffuse  energy echogram, same layout
    if(Ns==0 || Nr==0 || Nrays<=0 || S.nodes.empty() || recv_radius<=0.f) return;

    // receiver positions transformed once into solver space (same transform the mesh got).
    auto xform = [&](float x,float y,float z){
        return v3((x+S.shift.x)*S.scale, (y+S.shift.y)*S.scale, (z+S.shift.z)*S.scale);
    };
    std::vector<vec3> rec(Nr);
    for(int r=0;r<Nr;++r) rec[(size_t)r]=xform(rec_arr[(size_t)(3*r)],rec_arr[(size_t)(3*r+1)],rec_arr[(size_t)(3*r+2)]);

    const float R2     = recv_radius*recv_radius;      // detection-sphere radius^2 (every ray; also floors the rain's d^2)
    const float t_end  = (float)Nt/fs;                 // IR length [s]; rays/paths past this are dropped
    const float Rp_min = 1e-4f;                        // amplitude floor: once every band is below this, the ray is dead
    const float RT_INVPI = 0.318309886183791f;          // 1/pi, precomputed

    // =====================================================================================
    // ENERGY BOOKKEEPING: both deposit constants derived from first principles below.
    // We work in |p|^2 ("energy") with p = ps/(4 pi d) at distance d; main3D multiplies the rooted
    // pressure by 4 pi to reach its ISM convention p = ps/d (Lp,1m).
    //
    // SOLID ANGLE. Directions are measured as area on the unit sphere (steradians, sr). The
    // whole sphere = 4 pi sr (all directions); a hemisphere = 2 pi sr. A free monopole source
    // radiates into every direction, i.e. the full 4 pi. A small area A seen from distance d
    // subtends the solid angle Omega = A / d^2 (its area projected onto the unit sphere).
    //
    // SOURCE POWER (|p|^2 flux through any sphere, distance-independent):
    //     W = integral over sphere of |p|^2 dA = (ps/(4 pi d))^2 * 4 pi d^2 = ps^2/(4 pi).
    // We launch Nrays uniformly over the full 4 pi, so the angular ray density is Nrays/(4 pi)
    // rays per sr, and each ray carries an equal share of the power
    //     w_ray = W / Nrays = ps^2 / (4 pi Nrays).                       <-- Kdiff below
    //
    // ------ SPECULAR (detection sphere) : Kspec = w_ray / (pi R^2) ------
    // A specular ray has ONE outgoing direction and can never pass exactly through a POINT
    // receiver, so we give the receiver a finite sphere of radius R = recv_radius and count
    // crossings. IMAGE SOURCE: unfolding a specular path (mirror the source across each wall)
    // makes it a straight ray from an image point. Specular reflection is solid-angle
    // preserving (it maps a bundle dOmega to an equal reflected bundle dOmega: etendue
    // conservation), so the image radiates with the SAME uniform density Nrays/(4 pi) sr^-1 as
    // the real source. The detection sphere presents a disk of area pi R^2, subtending solid
    // angle pi R^2 / d^2 at path-distance d, so the number of rays of THIS path that cross it:		
    //     n_path = density * solid_angle =
    //     n_path = (Nrays/(4 pi)) * (pi R^2 / d^2) = Nrays * R^2 / (4 d^2).
    // We WANT the n_path crossings to sum to the ISM image energy (ps Rp air /(4 pi d))^2.
    // Solve for the per-crossing energy e_spec:
    //     e_spec = (ps Rp air/(4 pi d))^2 / n_path
    //            = [ps^2 Rp^2 air^2 /(16 pi^2 d^2)] * [4 d^2 /(Nrays R^2)]
    //            = ps^2 Rp^2 air^2 / (4 pi^2 Nrays R^2)
    //            = w_ray/(pi R^2) * Rp^2 air^2  ==  Kspec * Rp^2 * air^2.
    // The d^2 cancels (per-crossing energy is distance-independent) and R^2 in n_path cancels
    // the 1/R^2 in Kspec, so the summed result is R-independent (R is a solver knob).
    //
    // ------ DIFFUSE (next-event estimation) : uses Kdiff = w_ray directly ------
    // LAMBERT SOURCE. A diffusely reflecting patch is a Lambertian (ideally diffuse) secondary
    // source: it re-radiates into the HEMISPHERE in front of it with radiant intensity that
    // falls off as cos(theta) from the surface normal (equally bright from every viewing angle).
    // If it re-radiates a total power Phi, its radiant intensity (power per sr) is
    //     I(theta) = Phi * cos(theta) / pi.
    // The 1/pi is the normalisation over the cosine-weighted HEMISPHERE:
    //     integral over hemisphere of (cos(theta)/pi) dOmega = 1   =>   integral of I dOmega = Phi.
    // (Note pi, not 2 pi: the cosine weighting of the 2 pi hemisphere integrates to pi.)
    // Here the patch B re-radiates the scattered fraction s of its reflected energy, so
    //     Phi = s * w_ray * Rrefl^2 * air^2.
    // The receiver is a POINT at distance d_rec in direction theta from B's normal, so the
    // energy per unit area arriving there is I(theta)/d_rec^2 (inverse-square spreading):
    //     e_diff = w_ray * s * Rrefl^2 * air^2 * cos(theta)/(pi d_rec^2)  ==  Kdiff * s * Rrefl^2 * air^2 * geo.
    // Note the two pi-family constants differ by role: SOURCE uses 4 pi (free monopole, full
    // sphere); a SCATTER patch uses 1/pi (cosine-weighted hemisphere).
    // =====================================================================================
    // GA TIMING for the GUI. Rays are traced in a SPREAD order (index stride P coprime with Nrays,
    // ~golden ratio), because Fibonacci ray numbers run pole to pole: the first 2 % traced are then
    // a uniform sample of ALL directions, so their time extrapolates to the whole run (rays x
    // bounces, t_end, receivers and scene are all inside the measurement). Results are unchanged:
    // each ray's direction and random draws depend on its number only, not on when it is traced.
    const long long NT = (long long)Ns * (long long)Nrays;
    long long P = (long long)(0.6180339887 * Nrays) | 1;
    while (P > 1 && std::gcd(P, (long long)Nrays) != 1) P += 2;
    if (P < 1 || Nrays < 3) P = 1;
    std::atomic<long long> rays_done{0};
    const long long n_sample = std::max(1LL, NT / 50);          // estimate after 2 %
    const long long n_step   = std::max(1LL, NT / 20);          // progress line every 5 %
    const auto t_rt0 = std::chrono::steady_clock::now();
    auto rt_elapsed = [&]{ return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_rt0).count(); };
    std::printf("[GA] tracing %lld rays (%d source(s) x %d)\n", NT, Ns, Nrays); std::fflush(stdout);

	// for each source
    for(int s = 0; s < Ns; s++){
        const vec3 o0 = xform(src_arr[(size_t)(3*s)],src_arr[(size_t)(3*s+1)],src_arr[(size_t)(3*s+2)]); // source in solver space
        const float ps = (s<(int)src_pressure.size()) ? src_pressure[(size_t)s] : 1.f;                    // this source's pressure
        const float Kdiff = ps*ps/(4.f*M_PI*(float)Nrays); // w_ray = ps^2/(4 pi Nrays): energy carried by one ray
        // Equivalent to Reuben Thomas's thesis equation (6.5): for small gamma --> gamma approx sin(gamma)
        // approx R/r, where R is detector's radius, r is distance from src to rec, so, expanding in Taylor
        // cos(gamma) approx 1 - gamma² / 2 + O(gamma⁴), and using that sin(gamma) approx R/r, then using the
        // expansion,
        // 1 - cos(gamma) approx gamma² / 2 = R²/(2 r²).
        // Plugging into the denominator of (6.5) ==> 2pi r² N (1 - cos(gamma))) = 2pi r² N R² / (2 r²) =
        // = pi N R², so
        // Er = Es / (pi N R²).
        // Kspec is energy --> ps² / (4 pi² N R²); main multiplies the PRESSURE by 4 pi (energy by 16 pi²), so
        // Kspec_final = 4 ps² / (N R²) = Es / (pi N R²) with Es = 4 pi ps² (the source energy for which
        // |p|² = ps²/d², main's ISM convention): the same formula as Thomas. Q.E.D..
        
        const float Kspec = Kdiff*RT_INVPI/R2; // w_ray/(pi R^2): energy per specular detection-sphere crossing

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for(int rk = 0; rk < Nrays; rk++){			// breadth: independent samples
            const int ri = (int)(((long long)rk * P) % Nrays);   // spread order (see GA TIMING)
            ray3 r = rt_fib_ray(o0, ri, Nrays);		// depth: one linear path, in place
            float Rp[RT_NBANDS]; for(int b = 0; b < RT_NBANDS; b++) Rp[b]=1.f; // amplitude reflection product
            float d_travel = 0.f;            		// path length travelled so far [m]
            bool  scattered = false;         		// has this ray taken a diffuse bounce yet?
            bool  after_diffuse = false;     		// current segment starts at a DIFFUSE bounce (see rain)

            for(int bounce = 0; bounce <= max_order; bounce++){
                float th; int tri;
                const bool hit = rt_closest_hit(S, r, 1e-4f, th, tri);
                // segment end: hit point, or a far point if the ray escapes
                const float seg = hit ? th : 1e4f;
                const vec3 o = v3(r.ox,r.oy,r.oz);
                const vec3 d = v3(r.dx,r.dy,r.dz);
                const vec3 p1 = add3(o, mul3(d, seg));

                // ---- DETECTION on segment [o,p1]: the sphere, every ray, full energy (distance-
                // independent energy per crossing). The ray's STATE picks the buffer: still a pure
                // specular path -> e_spec; scattered at least once -> e_diff (enters at all times).
                // HANDOVER BY ORDER, not by time: this segment follows `bounce` reflections, so a pure
                // specular detection here IS an order-`bounce` image path. Orders <= ism_order are the
                // ISM's exact, deterministic arrivals (direct = order 0) -> skipped, else counted twice.
                // EXCEPT the segment right after a DIFFUSE draw: "scatter at that bounce, then
                // straight to the receiver" is exactly what the diffuse rain below deposits
                // deterministically at that bounce; detecting it here too would count it twice.
                std::vector<float>& e_buf = scattered ? e_diff : e_spec;
                const bool ism_owned = (!scattered && bounce <= ism_order);
                for(int rr = 0; !after_diffuse && !ism_owned && rr < Nr; rr++){
                    float s_ca;
                    if(rt_point_seg_dist2(rec[(size_t)rr], o, p1, s_ca) <= R2){
                        const float d_run = d_travel + s_ca; // total path to the receiver
                        const float tt = d_run/c;
                        if(tt < t_end && d_run > 1e-4f){
                            const int n0 = (int)(tt*fs);
                            if(n0 >= 0 && n0 < Nt){
                                for(int b = 0;b < RT_NBANDS; b++){
                                    const float air2 = m_air ? std::exp(-2.f*m_air[b]*d_run) : 1.f; // air^2 (energy)
                                    // per-crossing energy (see derivation block above): sums over n_path
                                    // crossings to the ISM image energy (ps Rp air/(4 pi d))^2; d cancels, R-independent
                                    const float e = Kspec * Rp[b]*Rp[b] * air2;
                                    //const float e = Kspec * Rp[b]*Rp[b]; // amplitude validation
                                    float* cell = &e_buf[((size_t)b*Nr + rr)*Nt + n0];
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
                    //Rrefl[b] = Rp[b] * std::fabs(R_oblique(al, 1.0)); // amplitude validation
                }

                // ---- DIFFUSE RAIN (next-event estimation) -> e_diff, every bounce, every ray.
                // The event "scatter here (prob s), then go straight to receiver rr": B = p1 is a
                // Lambert secondary source of power s * w_ray * Rrefl^2, radiant intensity
                // cos(theta)/pi, so the receiver gets
                //     e = w_ray * s * Rrefl^2 * air^2 * cos(theta) / (pi * d_rec^2)
                // The on-off sphere estimate of the SAME event has the same expectation:
                //     s * (cos(theta) R^2/d^2) [Lambert direction hits the sphere] * w_ray/(pi R^2)
                // -> identical; the rain just has no variance, so the early scattered "grass" is
                // smooth with few rays. That sphere term is skipped (after_diffuse), so nothing is
                // counted twice, and every LATER segment of a scattered ray is still detected ->
                // multiple scattering keeps its energy. s = the SAME single value as the draw and the
                // ISM (one value per surface). d_rec^2 >= R^2: inside the detection radius the sphere
                // saturates as well; stops a bounce next to a receiver from spiking as 1/d^2.
                {
                    const float sgr = (sctr_g && gid>=0 && gid<S.ngroups)
                                    ? std::max(0.f, std::min(1.f, sctr_g[(size_t)gid])) : 0.f;
                    if(sgr > 0.f){
                        const vec3 No = (dot3(N, d) > 0.f) ? mul3(N,-1.f) : N;  // face the incoming side
                        for(int rr = 0; rr < Nr; rr++){
                            const vec3 br = sub3(rec[(size_t)rr], p1);
                            const float d2 = norm2_3(br);
                            if(d2 < 1e-8f) continue;
                            const float drec = std::sqrt(d2);
                            const vec3 u = mul3(br, 1.f/drec);
                            const float cosr = dot3(No, u);
                            if(cosr <= 0.f) continue;                          // receiver behind the surface
                            ray3 sr; sr.ox=p1.x+No.x*1e-4f; sr.oy=p1.y+No.y*1e-4f; sr.oz=p1.z+No.z*1e-4f;
                            sr.dx=u.x; sr.dy=u.y; sr.dz=u.z;
                            float sh_t; int sh_tri;
                            if(rt_closest_hit(S, sr, 1e-4f, sh_t, sh_tri) && sh_t < drec-1e-3f) continue; // occluded
                            const float d_run = d_travel + drec;
                            if(d_run/c >= t_end) continue;
                            const int n0 = (int)((d_run/c)*fs);
                            if(n0 < 0 || n0 >= Nt) continue;
                            const float geo = cosr * RT_INVPI / std::max(d2, R2);
                            for(int b = 0;b < RT_NBANDS; b++){
                                const float air2 = m_air ? std::exp(-2.f*m_air[b]*d_run) : 1.f;
                                const float e = Kdiff * sgr * Rrefl[b]*Rrefl[b] * air2 * geo;
                                float* cell = &e_diff[((size_t)b*Nr + rr)*Nt + n0];
                                #ifdef _OPENMP
                                #pragma omp atomic
                                #endif
                                *cell += e;
                            }
                        }
                    }
                }

                // ---- ENERGY CONSERVATION: the ray carries the FULL reflected power (1-alpha)
                // forward. Scattering changes only the DIRECTION, not the energy, so scattered
                // energy keeps bouncing (multiple scattering) and the tail decays at the
                // absorption-governed rate for any s.
                float alive=0.f;
                for(int b = 0;b < RT_NBANDS; b++){
                	Rp[b] = Rrefl[b];
                	alive = std::max(alive, Rp[b]);
                }
                if(alive < Rp_min) break;

                // ---- STOCHASTIC direction: draw once with the GLOBAL s (one scalar per surface,
                // so no per-band trajectory conflict). Prob s -> diffuse (Lambert) reflection,
                // else specular. Over many rays a fraction s scatter and (1-s) stay specular,
                // reproducing the correct s/(1-s) split in aggregate while every ray keeps its
                // full energy. The rain above uses the same s.
                const unsigned int seed = ((unsigned)((s*Nrays+ri))*2654435761u) ^ ((unsigned)bounce*40503u);
                const float sg = (sctr_g && gid>=0 && gid<S.ngroups)
                               ? std::max(0.f, std::min(1.f, sctr_g[(size_t)gid])) : 0.f;
                vec3 ref;
                
                // The concept below can be found in Hanna Autio's thesis and in Brandão et al (2019, Aachen)
                // called on-off scattering (Autio, Vardaxis, Hagberg) (2021)
                const bool diffuse = (sg > 0.f && rt_u01(seed) < sg);
                if(diffuse){
                    const vec3 No = (dot3(N, d) > 0.f) ? mul3(N,-1.f) : N;   // outward-facing normal
                    ref = rt_lambert_dir(No, rt_u01(seed^0x9e3779b9u), rt_u01(seed^0x85ebca6bu));
                    scattered = true;                                        // deposits go to e_diff from now on
                } else {
                    rt_compute_reflected(ref, d, N);
                }
                after_diffuse = diffuse;       // next segment's direct detection is the rain's job
                // No per-band reweighting here (tried s_b/sg, (1-s_b)/(1-sg) per bounce: the weights
                // compound over 30-60 bounces and a few rays end up carrying the tail -> spikes).
                // One trajectory, one value: the ISM removes the same sg from its images.
                //rt_compute_reflected(ref, d, N); // amplitude validation (comment block above)
                // change ray's direction
                r.ox=p1.x + ref.x*1e-4f; r.oy=p1.y + ref.y*1e-4f; r.oz=p1.z + ref.z*1e-4f;
                r.dx=ref.x; r.dy=ref.y; r.dz=ref.z;
            }
            const long long k = ++rays_done;                    // this ray is finished
            if (k == n_sample || k % n_step == 0 || k == NT){
                const double el = rt_elapsed();
                #ifdef _OPENMP
                #pragma omp critical(rt_progress)
                #endif
                {
                    if (k == n_sample)
                        std::printf("[GA] estimate: %.1f s for all rays (timed on %lld of %lld)\n",
                                    el * (double)NT / (double)k, k, NT);
                    if (k % n_step == 0 || k == NT)
                        std::printf("[GA] progress %d%% eta %.1f s\n", (int)(100LL * k / NT),
                                    el * (double)(NT - k) / (double)k);
                    std::fflush(stdout);
                }
            }
        }
    }
    std::printf("[GA] rays traced in %.1f s\n", rt_elapsed()); std::fflush(stdout);
    // e_spec / e_diff hold physical ENERGY (not rooted). main sums the required channels
    // per bin and takes sqrt -> pressure (see the header note above).
}

// ============================================================
// Visualization: trace Nrays Fibonacci rays from src_world up to max_order bounces
// (purely specular, no bands/energy/receiver) and write every segment to `file`, one
// per line: "src_id ray_id x0 y0 z0 x1 y1 z1 order group" (ray_id unique across all sources),
// after "# group <id> <name>" header lines. Traces every source in `srcs`. Points are
// un-transformed to WORLD coords so they overlay the raw OBJ in the GUI. `order` is the bounce
// index (0 = source -> first hit, 1 = after 1st reflection, ...) for per-order colouring. Escaped
// rays get one stub segment (a quarter of the scene diagonal, order = group = -1) so the
// launch direction stays visible.
// ============================================================
inline static void rt_dump_paths(const RTScene& S, const std::vector<vec3>& srcs,
                                 int Nrays, int max_order, const char* file,
                                 const std::vector<std::string>* group_names = nullptr){
    std::FILE* f = std::fopen(file, "w");
    if(!f) return;
    // "# group <id> <name>" per surface, so the GUI can filter rays by the surface they hit; the
    // 10th column of every segment line is the group id of the surface it ENDS on (-1 = escaped).
    if(group_names)
        for(size_t g = 0; g < group_names->size(); g++)
            std::fprintf(f, "# group %d %s\n", (int)g, (*group_names)[g].c_str());
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
                const int seg_group = hit ? S.group[(size_t)tri] : -1;   // surface hit at the segment end
                std::fprintf(f, "%d %d %.6g %.6g %.6g %.6g %.6g %.6g %d %d\n",
                             (int)si, uid, ow.x,ow.y,ow.z, pw.x,pw.y,pw.z, seg_order, seg_group);
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
