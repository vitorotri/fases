#pragma once
/*
ism3D.h
-------
Deterministic image-source method (ISM) for early reflections, built on top of
parsing3D.h. Reuses that file's vec3 math, ray_triangle_intersect (occlusion),
point_triangle_dist2 (in-facet test), Scene3D::vertex_transformed, and the
per-triangle group_id -> group_names material key. parsing3D.h is NOT modified.

What this returns (you compute the reflection coefficients yourself)
--------------------------------------------------------------------
For every (source, receiver) pair, a list of valid paths. Each path carries:
  - length : the SINGLE total unfolded path length (m). For one image-source the
             per-segment distances are already summed; this equals the straight
             line from the image to the receiver. One value per path.
  - reflections[] : one entry per bounce, in source->receiver order, each holding
             the surface's complex octave-band impedance (Z_re[8], Z_im[8]) and the
             angle of incidence (cos_theta and theta). You form R_i(Z_i, theta_i, band)
             and combine them however you like.
  - amp    : geometric amplitude factor. 1 for planar paths; the spherical-mirror
             curvature spreading for curved-surface paths.
  - order, curved, approx : bookkeeping.

Impedances live at  <imp_dir>/<material_name>.imp  with line 1 = real values and
line 2 = imaginary values, one value per octave band (ISM_NBANDS). Material name is
taken to be the OBJ group name (Scene3D.group_names), matching your name-based
loading. Load once with ism_load_impedances(imp_dir, Sc.group_names) and pass the
table in. imp_dir comes from your manifest; this file does not parse the manifest.

Coordinate frame
----------------
Triangles live in solver world via vertex_transformed = (v + shift)*scale, shift =
-bmin + 3h (the +3h keeps surfaces off the WENO5 ghost layers). cfg.frame = AUTO
resolves each endpoint into the geometry frame by AABB containment, so the pad
cannot misalign anything regardless of how src/rec are authored.

Threading: host + optional OpenMP over receivers.
*/

#include "parsing3D.h"
#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>

#ifdef _OPENMP
  #include <omp.h>
#endif

static constexpr int ISM_NBANDS = 8;   // octave bands (63..8000 Hz), matching your .imp/.mat layout

enum class ISMFrame { AUTO, SOLVER_WORLD, OBJECT_SPACE };

// ============================================================
// Configuration
// ============================================================
struct ISMConfig {
    int   max_order            = 3;        // max planar reflection order
    float coplanar_cos         = 0.9998f;  // |dot(n1,n2)| to call two facets parallel (~1.1 deg)
    float coplanar_dist        = 1e-3f;    // plane-offset match, world units (m)
    float min_plane_area       = 1e-4f;    // drop merged planes below this area from image gen
    int   max_planes_per_group = 12;       // a group with more distinct planes => curved
    float facet_tol            = 1e-4f;    // in-facet slack, world units (m)
    float seg_eps              = 1e-5f;    // segment-fraction guard for t-bounds and occlusion

    ISMFrame frame             = ISMFrame::AUTO;

    bool  curved_first_order   = true;     // capture first-order specular off curved entities
    float curved_cone_cos      = 0.5f;     // prune facets to a cone (~60 deg) around the bisector
    float curved_amp_cap       = 4.0f;     // clamp curvature gain (concave focusing) and flag
    int   convex_unknown_as_convex = 1;    // ambiguous curvature sign -> treat convex (defocus, safe)
};

// ============================================================
// Material impedance (complex, per octave band)
// ============================================================
struct MaterialImpedance {
    std::array<float, ISM_NBANDS> Z_re;
    std::array<float, ISM_NBANDS> Z_im;
    bool loaded = false;
    MaterialImpedance(){ Z_re.fill(0.f); Z_im.fill(0.f); }
};

// Loads <imp_dir>/<group_name>.imp for each group (line1=real, line2=imag).
// Indexed by group_id. Missing/short files are left loaded=false (zeros); boundary
// or unnamed groups with no material simply stay unloaded.
inline static std::vector<MaterialImpedance> ism_load_impedances(
    const std::string& imp_dir, const std::vector<std::string>& group_names)
{
    std::vector<MaterialImpedance> out(group_names.size());

    auto parse_row = [](const std::string& s, std::array<float, ISM_NBANDS>& dst)->int{
        std::istringstream iss(s); float v; int n=0;
        while(n < ISM_NBANDS && (iss >> v)){ dst[(size_t)n]=v; ++n; }
        return n;
    };

    for(size_t g=0; g<group_names.size(); ++g){
        MaterialImpedance M;
        std::string path = imp_dir;
        if(!path.empty() && path.back()!='/' && path.back()!='\\') path += '/';
        path += group_names[g] + ".imp";

        std::ifstream f(path);
        if(f){
            std::string line; std::vector<std::string> rows;
            while(std::getline(f, line)){
                const size_t a = line.find_first_not_of(" \t\r\n");
                if(a == std::string::npos) continue;     // blank
                if(line[a] == '#') continue;             // comment
                rows.push_back(line);
                if(rows.size() == 2) break;
            }
            if(rows.size() >= 2){
                const int nr = parse_row(rows[0], M.Z_re);
                const int ni = parse_row(rows[1], M.Z_im);
                M.loaded = (nr == ISM_NBANDS && ni == ISM_NBANDS);
                if(!M.loaded)
                    std::fprintf(stderr, "[ISM] %s: expected %d values/line (re,im); got re=%d im=%d\n",
                                 path.c_str(), ISM_NBANDS, nr, ni);
            } else {
                std::fprintf(stderr, "[ISM] %s: fewer than two data lines\n", path.c_str());
            }
        }
        out[g] = M;
    }
    return out;
}

// ============================================================
// Reflector planes / curved reflectors
// ============================================================
struct Reflector {
    vec3  n   = v3(0,0,1);
    float d   = 0.f;
    int   group_id = -1;
    float area = 0.f;
    std::vector<int> tri_ids;
};

struct CurvedReflector {
    vec3  center = v3(0,0,0);
    float radius = 0.f;
    int   group_id = -1;
    int   convex = +1;        // +1 convex (defocus), -1 concave (focus)
    std::vector<int> tri_ids;
};

// One bounce: material impedance + incidence angle at that surface.
struct ISMReflection {
    int   group_id = -1;
    float cos_theta = 1.f;    // cosine of the angle of incidence (0 = grazing, 1 = normal)
    float theta     = 0.f;    // angle of incidence (radians)
    std::array<float, ISM_NBANDS> Z_re;
    std::array<float, ISM_NBANDS> Z_im;
    ISMReflection(){ Z_re.fill(0.f); Z_im.fill(0.f); }
};

// One validated path. You build reflection coefficients from reflections[].
struct ISMPath {
    float length = 0.f;                       // single total unfolded length (m)
    int   order  = 0;
    float amp    = 1.f;                        // geometric factor (1 planar; curvature for curved)
    bool  curved = false;
    bool  approx = false;
    std::vector<ISMReflection> reflections;   // source->receiver order; empty for the direct path
};

// ============================================================
// Small geometry helpers (host)
// ============================================================
inline static vec3 ism_tri_normal_area(const Scene3D& Sc, const Tri& t, float& area_out){
    const vec3 a = Sc.vertex_transformed(t.v0);
    const vec3 b = Sc.vertex_transformed(t.v1);
    const vec3 c = Sc.vertex_transformed(t.v2);
    vec3 nc = cross3(sub3(b,a), sub3(c,a));
    const float len = std::sqrt(norm2_3(nc));
    area_out = 0.5f * len;
    if(len > 1e-20f) nc = mul3(nc, 1.f/len); else nc = v3(0,0,1);
    return nc;
}

inline static vec3 ism_normalize(const vec3& v){
    const float l = std::sqrt(norm2_3(v));
    return (l > 1e-20f) ? mul3(v, 1.f/l) : v3(0,0,0);
}

inline static vec3 ism_canon_normal(vec3 n){
    const float ax=std::fabs(n.x), ay=std::fabs(n.y), az=std::fabs(n.z);
    float s = 1.f;
    if(az>=ax && az>=ay)      s = (n.z<0.f)?-1.f:1.f;
    else if(ay>=ax)           s = (n.y<0.f)?-1.f:1.f;
    else                      s = (n.x<0.f)?-1.f:1.f;
    return mul3(n, s);
}

inline static vec3 ism_reflect_point(const vec3& p, const vec3& n, float d){
    const float sd = dot3(n,p) - d;
    return sub3(p, mul3(n, 2.f*sd));
}

inline static bool ism_point_in_aabb(const vec3& p, const vec3& mn, const vec3& mx, float eps){
    return p.x >= mn.x-eps && p.x <= mx.x+eps &&
           p.y >= mn.y-eps && p.y <= mx.y+eps &&
           p.z >= mn.z-eps && p.z <= mx.z+eps;
}

inline static vec3 ism_resolve_endpoint(const Scene3D& Sc, const vec3& raw,
                                        const vec3& aabb_min, const vec3& aabb_max,
                                        ISMFrame frame, float eps){
    const vec3 as_world  = raw;
    const vec3 as_object = mul3(add3(raw, Sc.shift), Sc.scale);
    if(frame == ISMFrame::SOLVER_WORLD) return as_world;
    if(frame == ISMFrame::OBJECT_SPACE) return as_object;
    const bool in_w = ism_point_in_aabb(as_world,  aabb_min, aabb_max, eps);
    const bool in_o = ism_point_in_aabb(as_object, aabb_min, aabb_max, eps);
    if(in_w) return as_world;
    if(in_o) return as_object;
    std::fprintf(stderr,
        "[ISM] endpoint (%.3g,%.3g,%.3g) outside scene in both frames; using solver-world.\n",
        (double)raw.x,(double)raw.y,(double)raw.z);
    return as_world;
}

inline static bool ism_point_on_reflector(const Scene3D& Sc, const std::vector<int>& tri_ids,
                                          const vec3& x, float tol){
    const float tol2 = tol*tol;
    for(int t : tri_ids){
        const Tri& tri = Sc.tris[(size_t)t];
        const float d2 = point_triangle_dist2(x,
            Sc.vertex_transformed(tri.v0),
            Sc.vertex_transformed(tri.v1),
            Sc.vertex_transformed(tri.v2));
        if(d2 <= tol2) return true;
    }
    return false;
}

inline static bool ism_occluded(const Scene3D& Sc, const std::vector<int>& tri2plane,
                                const vec3& p0, const vec3& p1,
                                int ignoreA, int ignoreB, float eps, int ignore_group=-1){
    const vec3 dir = sub3(p1, p0);
    const int nt = (int)Sc.tris.size();
    for(int t=0; t<nt; ++t){
        const Tri& tri = Sc.tris[(size_t)t];
        if(ignore_group >= 0 && tri.group_id == ignore_group) continue;
        const int pl = tri2plane[(size_t)t];
        if(pl >= 0 && (pl==ignoreA || pl==ignoreB)) continue;
        float tt;
        if(ray_triangle_intersect(p0, dir,
               Sc.vertex_transformed(tri.v0),
               Sc.vertex_transformed(tri.v1),
               Sc.vertex_transformed(tri.v2), eps, tt)){
            if(tt < 1.f - eps) return true;
        }
    }
    return false;
}

// ============================================================
// 1) Coplanar merge + curved-entity classification
// ============================================================
inline static void ism_build_reflectors(const Scene3D& Sc, const ISMConfig& cfg,
                                        std::vector<Reflector>& planes,
                                        std::vector<CurvedReflector>& curved,
                                        std::vector<int>& tri2plane,
                                        std::vector<uint8_t>& group_is_curved)
{
    planes.clear(); curved.clear();
    tri2plane.assign(Sc.tris.size(), -1);

    int max_gid = -1;
    for(const auto& t : Sc.tris) max_gid = std::max(max_gid, t.group_id);
    group_is_curved.assign((size_t)std::max(0, max_gid+1), 0);
    if(max_gid < 0) return;

    struct PlaneAccum { vec3 n; float d; float area; std::vector<int> tris; };

    for(int gid=0; gid<=max_gid; ++gid){
        std::vector<PlaneAccum> acc; acc.reserve(8);
        std::vector<int> group_tris;

        for(int t=0; t<(int)Sc.tris.size(); ++t){
            const Tri& tri = Sc.tris[(size_t)t];
            if(tri.group_id != gid) continue;
            group_tris.push_back(t);

            float area;
            vec3 n = ism_canon_normal(ism_tri_normal_area(Sc, tri, area));
            const float d = dot3(n, Sc.vertex_transformed(tri.v0));

            int hit = -1;
            for(int p=0; p<(int)acc.size(); ++p){
                if(dot3(acc[(size_t)p].n, n) > cfg.coplanar_cos &&
                   std::fabs(acc[(size_t)p].d - d) < cfg.coplanar_dist){ hit = p; break; }
            }
            if(hit < 0) acc.push_back({n,d,area,{t}});
            else { acc[(size_t)hit].area += area; acc[(size_t)hit].tris.push_back(t); }
        }
        if(group_tris.empty()) continue;

        const bool is_curved = ((int)acc.size() > cfg.max_planes_per_group);
        group_is_curved[(size_t)gid] = is_curved ? 1u : 0u;

        if(is_curved){
            vec3 csum = v3(0,0,0); int vcount = 0;
            for(int t : group_tris){
                const Tri& tri = Sc.tris[(size_t)t];
                csum = add3(csum, Sc.vertex_transformed(tri.v0));
                csum = add3(csum, Sc.vertex_transformed(tri.v1));
                csum = add3(csum, Sc.vertex_transformed(tri.v2));
                vcount += 3;
            }
            vec3 C = (vcount>0) ? mul3(csum, 1.f/(float)vcount) : v3(0,0,0);

            float rsum = 0.f; int rcount = 0; double convex_acc = 0.0;
            for(int t : group_tris){
                const Tri& tri = Sc.tris[(size_t)t];
                const vec3 a=Sc.vertex_transformed(tri.v0),
                           b=Sc.vertex_transformed(tri.v1),
                           c=Sc.vertex_transformed(tri.v2);
                const vec3 fc = mul3(add3(add3(a,b),c), 1.f/3.f);
                rsum += std::sqrt(norm2_3(sub3(fc, C))); ++rcount;
                float ar; const vec3 nf = ism_tri_normal_area(Sc, tri, ar);
                convex_acc += (double)dot3(nf, sub3(fc, C));
            }
            CurvedReflector Cr;
            Cr.center  = C;
            Cr.radius  = (rcount>0) ? rsum/(float)rcount : 0.f;
            Cr.group_id= gid;
            if(convex_acc > 0.0)      Cr.convex = +1;
            else if(convex_acc < 0.0) Cr.convex = -1;
            else                      Cr.convex = cfg.convex_unknown_as_convex ? +1 : -1;
            Cr.tri_ids = group_tris;
            if(Cr.radius > 1e-6f) curved.push_back(std::move(Cr));
            continue;
        }

        for(const auto& pa : acc){
            if(pa.area < cfg.min_plane_area) continue;
            Reflector R; R.n=pa.n; R.d=pa.d; R.group_id=gid; R.area=pa.area; R.tri_ids=pa.tris;
            const int pidx = (int)planes.size();
            for(int t : R.tri_ids) tri2plane[(size_t)t] = pidx;
            planes.push_back(std::move(R));
        }
    }
}

// ============================================================
// 2) Planar image tree (per source, receiver-independent)
// ============================================================
struct ISMImageNode { vec3 image; int plane=-1; int parent=-1; int order=0; };

inline static void ism_build_images(const std::vector<Reflector>& planes,
                                    const vec3& source, int max_order,
                                    std::vector<ISMImageNode>& nodes)
{
    nodes.clear();
    nodes.push_back({source, -1, -1, 0});
    size_t begin = 0;
    for(int order=1; order<=max_order; ++order){
        const size_t end = nodes.size();
        for(size_t pi=begin; pi<end; ++pi){
            const ISMImageNode parent = nodes[pi];
            for(int p=0; p<(int)planes.size(); ++p){
                if(p == parent.plane) continue;
                const Reflector& R = planes[(size_t)p];
                if(std::fabs(dot3(R.n, parent.image) - R.d) < 1e-6f) continue;
                ISMImageNode child;
                child.image  = ism_reflect_point(parent.image, R.n, R.d);
                child.plane  = p; child.parent = (int)pi; child.order = order;
                nodes.push_back(child);
            }
        }
        begin = end;
    }
}

// ============================================================
// Validate a planar leaf against a receiver.
// On success: group_chain_out and cos_theta_out (both source->receiver order) and length.
// ============================================================
inline static bool ism_validate_path(const Scene3D& Sc,
                                     const std::vector<Reflector>& planes,
                                     const std::vector<int>& tri2plane,
                                     const std::vector<ISMImageNode>& nodes,
                                     int leaf, const vec3& source, const vec3& receiver,
                                     const ISMConfig& cfg,
                                     std::vector<int>& group_chain_out,
                                     std::vector<float>& cos_theta_out,
                                     float& length_out)
{
    group_chain_out.clear(); cos_theta_out.clear();
    std::vector<int>  planes_seq;   // leaf .. first
    std::vector<vec3> pts;          // leaf .. first

    vec3 point = receiver;
    int node = leaf;
    while(nodes[(size_t)node].plane >= 0){
        const ISMImageNode& nd = nodes[(size_t)node];
        const Reflector& R = planes[(size_t)nd.plane];
        const vec3 dir = sub3(nd.image, point);
        const float denom = dot3(R.n, dir);
        if(std::fabs(denom) < 1e-12f) return false;
        const float t = (R.d - dot3(R.n, point)) / denom;
        if(t <= cfg.seg_eps || t >= 1.f - cfg.seg_eps) return false;
        const vec3 refl = add3(point, mul3(dir, t));
        if(!ism_point_on_reflector(Sc, R.tri_ids, refl, cfg.facet_tol)) return false;
        planes_seq.push_back(nd.plane);
        pts.push_back(refl);
        point = refl;
        node  = nd.parent;
    }

    const int m = (int)pts.size();

    // occlusion: receiver -> pts[0] -> ... -> pts[m-1] -> source
    {
        const int  ignA = (m>0) ? planes_seq[0] : -1;
        const vec3 next = (m>0) ? pts[0] : source;
        if(ism_occluded(Sc, tri2plane, receiver, next, ignA, -1, cfg.seg_eps)) return false;
    }
    for(int s=0; s<m; ++s){
        const vec3 a = pts[(size_t)s];
        const int  ia = planes_seq[(size_t)s];
        const vec3 b = (s+1<m) ? pts[(size_t)(s+1)] : source;
        const int  ib = (s+1<m) ? planes_seq[(size_t)(s+1)] : -1;
        if(ism_occluded(Sc, tri2plane, a, b, ia, ib, cfg.seg_eps)) return false;
    }

    length_out = std::sqrt(norm2_3(sub3(receiver, nodes[(size_t)leaf].image)));

    // incidence angle at each reflection point (leaf..first order), then reverse.
    std::vector<float> cos_ls((size_t)m, 1.f);
    for(int s=0; s<m; ++s){
        const vec3 P    = pts[(size_t)s];
        const vec3 prev = (s==0)   ? receiver : pts[(size_t)(s-1)]; // receiver side
        const vec3 din  = ism_normalize(sub3(P, prev));
        const vec3 n    = planes[(size_t)planes_seq[(size_t)s]].n;  // unit
        float ct = std::fabs(dot3(din, n));
        ct = std::min(1.f, std::max(0.f, ct));
        cos_ls[(size_t)s] = ct;
    }

    group_chain_out.resize((size_t)m);
    cos_theta_out.resize((size_t)m);
    for(int s=0; s<m; ++s){
        const int idx = m-1-s;                                  // reverse to source->receiver
        group_chain_out[(size_t)s] = planes[(size_t)planes_seq[(size_t)idx]].group_id;
        cos_theta_out[(size_t)s]   = cos_ls[(size_t)idx];
    }
    return true;
}

// ============================================================
// Curved first-order specular (exact facet geometry + curvature amplitude).
// Returns the specular facet's group_id, the incidence cosine, length and amp.
// ============================================================
inline static bool ism_curved_first_order(const Scene3D& Sc, const std::vector<int>& tri2plane,
                                          const CurvedReflector& Cr,
                                          const vec3& S, const vec3& Rc,
                                          const ISMConfig& cfg,
                                          int& group_out, float& cos_theta_out,
                                          float& length_out, float& amp_out, bool& approx_out)
{
    const vec3 bis = ism_normalize(add3(ism_normalize(sub3(S, Cr.center)),
                                        ism_normalize(sub3(Rc, Cr.center))));
    int   best_tri = -1; float best_centrality = -1.f;
    float best_ct = 1.f, best_di = 0.f, best_dr = 0.f;

    for(int t : Cr.tri_ids){
        const Tri& tri = Sc.tris[(size_t)t];
        float ar; vec3 nf = ism_tri_normal_area(Sc, tri, ar);
        if(ar <= 0.f) continue;
        if(std::fabs(dot3(nf, bis)) < cfg.curved_cone_cos) continue;

        const float d = dot3(nf, Sc.vertex_transformed(tri.v0));
        const vec3 img = ism_reflect_point(S, nf, d);
        const vec3 dir = sub3(img, Rc);
        const float denom = dot3(nf, dir);
        if(std::fabs(denom) < 1e-12f) continue;
        const float tt = (d - dot3(nf, Rc)) / denom;
        if(tt <= cfg.seg_eps || tt >= 1.f - cfg.seg_eps) continue;
        const vec3 P = add3(Rc, mul3(dir, tt));

        const std::vector<int> one{t};
        if(!ism_point_on_reflector(Sc, one, P, cfg.facet_tol)) continue;
        if(ism_occluded(Sc, tri2plane, S,  P, -1, -1, cfg.seg_eps, Cr.group_id)) continue;
        if(ism_occluded(Sc, tri2plane, Rc, P, -1, -1, cfg.seg_eps, Cr.group_id)) continue;

        const vec3 a=Sc.vertex_transformed(tri.v0),
                   b=Sc.vertex_transformed(tri.v1),
                   c=Sc.vertex_transformed(tri.v2);
        const vec3 fc = mul3(add3(add3(a,b),c), 1.f/3.f);
        const float centrality = 1.f / (1e-6f + std::sqrt(norm2_3(sub3(P, fc))));
        if(centrality > best_centrality){
            best_centrality = centrality;
            best_tri = t;
            const vec3 uS = ism_normalize(sub3(S, P));
            best_ct = std::min(1.f, std::max(0.f, std::fabs(dot3(nf, uS))));
            best_di = std::sqrt(norm2_3(sub3(S, P)));
            best_dr = std::sqrt(norm2_3(sub3(Rc, P)));
        }
    }
    if(best_tri < 0) return false;

    group_out     = Cr.group_id;
    cos_theta_out = best_ct;
    length_out    = best_di + best_dr;

    // spherical-mirror divergence (one principal plane). f = R*cos(theta)/2.
    //   1/di' = 1/di + sgn*2/(R*cos(theta)) ,  sgn=-1 convex, +1 concave
    //   amp   = (di+dr)/(|di'|+dr)
    // For a full sphere both principal directions curve; if your reference defines
    // the spreading as the product over both, use amp -> amp*amp. Sign/exponent are
    // yours to confirm; geometry/length/angle above are exact regardless.
    approx_out = false;
    const double R  = (double)Cr.radius;
    const double ct = std::max(1e-3, (double)best_ct);
    const double di = (double)best_di, dr = (double)best_dr;
    const double sgn = (Cr.convex >= 0) ? -1.0 : +1.0;
    const double inv = 1.0/di + sgn*2.0/(R*ct);
    const double di_p = (std::fabs(inv) > 1e-12) ? 1.0/inv : 1e12;
    double amp = (di + dr) / (std::fabs(di_p) + dr);
    if(!(amp > 0.0) || amp > (double)cfg.curved_amp_cap){
        amp = std::min(std::max(amp, 1e-3), (double)cfg.curved_amp_cap);
        approx_out = true;   // near a focus: GA invalid, hand to the wave solver
    }
    amp_out = (float)amp;
    return true;
}

// ============================================================
// 3) Top-level: paths for every (source, receiver) pair
// ============================================================
inline static void ism_paths_for_pairs(
    const Scene3D& Sc,
    const std::vector<float>& src_arr,
    const std::vector<float>& rec_arr,
    const std::vector<MaterialImpedance>& impedance_by_group,   // ism_load_impedances(...)
    const ISMConfig& cfg,
    std::vector<std::vector<std::vector<ISMPath>>>& out_paths,
    std::vector<Reflector>*        planes_out = nullptr,
    std::vector<CurvedReflector>*  curved_out = nullptr,
    std::vector<uint8_t>*          group_is_curved_out = nullptr)
{
    std::vector<Reflector>       planes;
    std::vector<CurvedReflector> curved;
    std::vector<int>             tri2plane;
    std::vector<uint8_t>         curvedflag;
    ism_build_reflectors(Sc, cfg, planes, curved, tri2plane, curvedflag);
    if(planes_out)          *planes_out = planes;
    if(curved_out)          *curved_out = curved;
    if(group_is_curved_out) *group_is_curved_out = curvedflag;

    vec3 aabb_min, aabb_max;
    mesh_bbox_transformed(Sc, aabb_min, aabb_max);
    const float diag = std::sqrt(norm2_3(sub3(aabb_max, aabb_min)));
    const float aabb_eps = 0.02f * diag;

    auto make_refl = [&](int gid, float ct)->ISMReflection{
        ISMReflection R;
        R.group_id  = gid;
        R.cos_theta = ct;
        R.theta     = std::acos(std::min(1.f, std::max(0.f, ct)));
        if(gid >= 0 && gid < (int)impedance_by_group.size()){
            R.Z_re = impedance_by_group[(size_t)gid].Z_re;
            R.Z_im = impedance_by_group[(size_t)gid].Z_im;
        }
        return R;
    };

    const int Ns = (int)(src_arr.size()/3);
    const int Nr = (int)(rec_arr.size()/3);
    out_paths.assign((size_t)Ns, std::vector<std::vector<ISMPath>>((size_t)Nr));

    for(int s=0; s<Ns; ++s){
        const vec3 source = ism_resolve_endpoint(Sc,
            v3(src_arr[(size_t)(3*s+0)], src_arr[(size_t)(3*s+1)], src_arr[(size_t)(3*s+2)]),
            aabb_min, aabb_max, cfg.frame, aabb_eps);

        std::vector<ISMImageNode> nodes;
        ism_build_images(planes, source, cfg.max_order, nodes);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for(int r=0; r<Nr; ++r){
            const vec3 receiver = ism_resolve_endpoint(Sc,
                v3(rec_arr[(size_t)(3*r+0)], rec_arr[(size_t)(3*r+1)], rec_arr[(size_t)(3*r+2)]),
                aabb_min, aabb_max, cfg.frame, aabb_eps);

            std::vector<ISMPath>& dst = out_paths[(size_t)s][(size_t)r];

            // direct path
            if(!ism_occluded(Sc, tri2plane, source, receiver, -1, -1, cfg.seg_eps)){
                ISMPath p0;
                p0.length = std::sqrt(norm2_3(sub3(receiver, source)));
                dst.push_back(std::move(p0));
            }

            // planar reflected paths
            std::vector<int> gchain; std::vector<float> cchain; float len;
            for(int leaf=1; leaf<(int)nodes.size(); ++leaf){
                if(ism_validate_path(Sc, planes, tri2plane, nodes, leaf,
                                     source, receiver, cfg, gchain, cchain, len)){
                    ISMPath path;
                    path.length = len;
                    path.order  = (int)gchain.size();
                    path.reflections.reserve(gchain.size());
                    for(size_t k=0;k<gchain.size();++k)
                        path.reflections.push_back(make_refl(gchain[k], cchain[k]));
                    dst.push_back(std::move(path));
                }
            }

            // curved first-order specular paths
            if(cfg.curved_first_order){
                for(const auto& Cr : curved){
                    int gid; float ct, clen, camp; bool capprox;
                    if(ism_curved_first_order(Sc, tri2plane, Cr, source, receiver, cfg,
                                              gid, ct, clen, camp, capprox)){
                        ISMPath path;
                        path.length = clen;
                        path.order  = 1;
                        path.amp    = camp;
                        path.curved = true;
                        path.approx = capprox;
                        path.reflections.push_back(make_refl(gid, ct));
                        dst.push_back(std::move(path));
                    }
                }
            }

            std::sort(dst.begin(), dst.end(),
                      [](const ISMPath& a, const ISMPath& b){ return a.length < b.length; });
        }
    }
}
