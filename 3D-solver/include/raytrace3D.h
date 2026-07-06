#pragma once
/*
 * Copyright (c) 2025 Vito Romanelli Tricanico
 *
 * SPDX-License-Identifier: BSD-2-Clause (see License)
 */

/*
raytrace3D.h
------------
Acoustical ray-tracer, adapted from raytracer.c (light path tracer). Same syntax:
the `ray` struct (ox/oy/oz origin, dx/dy/dz unit direction), compute_reflected(),
and the recursive_trace() bounce loop shape. Two differences from the light tracer:

  1. Primitive is a PLANAR FACET (OBJ triangle), not a sphere. Intersection reuses
     ray_triangle_intersect() from parsing3D.h -- no sphere quadratic.
  2. A BVH over the triangle soup makes it survive highly-patched OBJ scenes
     (O(log n) per bounce instead of O(n)).

Rays leave the source on a Fibonacci-sphere pattern (near-uniform angular spread).
This is the DIFFUSE LATE TAIL that continues the ISM past its max reflection order:
it fills the energy echogram after the ISM crossover sample. Energy (not pressure)
is traced -- 1/d^2 spreading falls out of ray divergence, so we never divide by d^2.

Call sequence in main3D.cu:
    RTScene rtS = rt_build(sc);              // build once, before scene_free
    rt_trace_late(rtS, ..., echo);           // fill echo[8 * Nr * Nt] energies
    // then, per (band b, receiver r): p += sign * sqrt(echo[(b*Nr+r)*Nt + t])

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
// noise (deterministic, reproducible). ponytail: good enough; swap for a real
// Poisson process if the tail statistics ever need to be exact.
inline static float rt_sign(unsigned int s){
    s ^= s>>16; s *= 0x7feb352du; s ^= s>>15; s *= 0x846ca68bu; s ^= s>>16;
    return (s & 1u) ? 1.f : -1.f;
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
// (raytracer.c::recursive_trace shape), deposit energy into per-band per-receiver
// echograms whenever a ray segment passes within recv_radius of a receiver.
//
// echo is sized RT_NBANDS * Nr * Nt and indexed echo[(b*Nr + r)*Nt + sample].
// alpha_gb (may be null => rigid) is absorption per group per band, alpha_gb[gid*8+b].
// Source/receiver positions are RAW (world) and get the same (p+shift)*scale as
// the walls, so times line up with the ISM.
// ============================================================
inline static void rt_trace_late(const RTScene& S,
                                 const std::vector<float>& src_arr,
                                 const std::vector<float>& src_pressure,
                                 const std::vector<float>& rec_arr,
                                 const float* alpha_gb,   // [ngroups*8] or nullptr
                                 float c, float fs, int Nt,
                                 int Nrays, int max_order, float recv_radius,
                                 std::vector<float>& echo)
{
    const int Ns=(int)(src_arr.size()/3), Nr=(int)(rec_arr.size()/3);
    echo.assign((size_t)RT_NBANDS*(size_t)Nr*(size_t)Nt, 0.f);
    if(Ns==0 || Nr==0 || Nrays<=0 || S.nodes.empty()) return;

    auto xform = [&](float x,float y,float z){
        return v3((x+S.shift.x)*S.scale, (y+S.shift.y)*S.scale, (z+S.shift.z)*S.scale);
    };
    std::vector<vec3> rec(Nr);
    for(int r=0;r<Nr;++r) rec[(size_t)r]=xform(rec_arr[(size_t)(3*r)],rec_arr[(size_t)(3*r+1)],rec_arr[(size_t)(3*r+2)]);

    const float R2 = recv_radius*recv_radius;
    const float t_end = (float)Nt/fs;      // stop rays past the echogram length
    const float e_min = 1e-8f;             // energy floor: kill exhausted rays

    for(int s=0;s<Ns;++s){
        const vec3 o0 = xform(src_arr[(size_t)(3*s)],src_arr[(size_t)(3*s+1)],src_arr[(size_t)(3*s+2)]);
        const float ps = (s<(int)src_pressure.size()) ? src_pressure[(size_t)s] : 1.f;
        const float E0 = (ps*ps)/(float)Nrays;   // power per ray

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for(int ri=0; ri<Nrays; ++ri){
            ray3 r = rt_fib_ray(o0, ri, Nrays);
            float E[RT_NBANDS]; for(int b=0;b<RT_NBANDS;++b) E[b]=E0;
            float cum = 0.f;                 // path length so far [m]

            for(int bounce=0; bounce<=max_order; ++bounce){
                float th; int tri;
                const bool hit = rt_closest_hit(S, r, 1e-4f, th, tri);
                // segment end: hit point, or a far point if the ray escapes
                const float seg = hit ? th : 1e4f;
                const vec3 o = v3(r.ox,r.oy,r.oz);
                const vec3 d = v3(r.dx,r.dy,r.dz);
                const vec3 p1 = add3(o, mul3(d, seg));

                // receiver detection along this segment
                for(int rr=0; rr<Nr; ++rr){
                    float s_ca;
                    if(rt_point_seg_dist2(rec[(size_t)rr], o, p1, s_ca) <= R2){
                        const float tt = (cum + s_ca)/c;
                        if(tt < t_end){
                            const int n0 = (int)(tt*fs);
                            if(n0>=0 && n0<Nt){
                                for(int b=0;b<RT_NBANDS;++b){
                                    float* cell = &echo[((size_t)b*Nr + rr)*Nt + n0];
                                    #ifdef _OPENMP
                                    #pragma omp atomic
                                    #endif
                                    *cell += E[b];
                                }
                            }
                        }
                    }
                }

                if(!hit) break;              // escaped the geometry
                cum += seg;
                if(cum/c >= t_end) break;

                // absorb, then reflect (specular), raytracer.c bounce shape
                const int gid = S.group[(size_t)tri];
                float alive=0.f;
                for(int b=0;b<RT_NBANDS;++b){
                    float al = (alpha_gb && gid>=0 && gid<S.ngroups) ? alpha_gb[gid*RT_NBANDS+b] : 0.f;
                    al = std::max(0.f, std::min(0.99f, al));
                    E[b] *= (1.f-al);
                    alive = std::max(alive, E[b]);
                }
                if(alive < e_min) break;

                vec3 N = rt_tri_normal(S, tri);
                vec3 ref; rt_compute_reflected(ref, d, N);
                const vec3 hitp = add3(o, mul3(d, th));
                r.ox=hitp.x + ref.x*1e-4f; r.oy=hitp.y + ref.y*1e-4f; r.oz=hitp.z + ref.z*1e-4f;
                r.dx=ref.x; r.dy=ref.y; r.dz=ref.z;
            }
        }
    }
}
