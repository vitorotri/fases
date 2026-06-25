#pragma once
/*
ism3D.h
-------
Deterministic image-source method (ISM) for early reflections, on top of parsing3D.h.
Reuses vec3 math, ray_triangle_intersect (occlusion), point_triangle_dist2 (in-facet),
Scene3D::vertex_transformed, per-triangle group_id. parsing3D.h is NOT modified.

Three reflector kinds, one unified image tree
---------------------------------------------
  * Planar walls  : coplanar facets merged by GEOMETRY ALONE across all groups
                    (different objects in the same plane share ONE image).
  * Curved proxies: a dense cluster of small non-coplanar facets (sphere, cylinder,
                    bump, free-form) is SIMPLIFIED -- region-grown by normal coherence
                    into a few larger planar patches. Each patch is a reflector whose
                    local curvature radius (per-patch least-squares sphere fit) drives
                    a spherical-mirror amplitude correction. A sphere becomes ~N caps
                    automatically; nothing is sphere-specific.
  * (flat proxies): a patch that fits no sphere (planar region) is just a flat reflector.

All reflectors live in one list and feed one image tree. Curved bounces are capped at
cfg.max_order_curved so a simplified sphere's proxies don't re-explode the deep tree.

Plane vs material are SEPARATED: a reflector carries no material. Material is resolved
at the HIT FACET via group_to_material (your oid/layer map), so a proxy/plane spanning
several objects or layers resolves each bounce correctly.

Per path you get (you build the reflection coefficients yourself):
  length        : SINGLE total unfolded length (m), one value per image-source.
  reflections[] : per bounce (source->receiver): material_id, hit group_id, incidence
                  angle (cos_theta, theta), complex octave-band impedance (Z_re[8],Z_im[8]).
  amp           : product of per-bounce curvature factors (1 for purely planar paths).
  order, curved, approx.

Frame: vertex_transformed=(v+shift)*scale, shift=-bmin+3h. cfg.frame=AUTO resolves each
endpoint by AABB containment, so the pad can't misalign anything.

Threading: host + optional OpenMP over receivers.
*/

#include "parsing3D.h"
#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

#ifdef _OPENMP
  #include <omp.h>
#endif

static constexpr int ISM_NBANDS = 8;
enum class ISMFrame { AUTO, SOLVER_WORLD, OBJECT_SPACE };

// ============================================================
// Configuration
// ============================================================
struct ISMConfig {
    int   max_order            = 3;
    float coplanar_cos         = 0.9998f;  // wall merge: parallel threshold
    float coplanar_dist        = 1e-3f;    // wall merge: offset match (m)
    float min_plane_area       = 1e-4f;
    int   max_planes_per_group = 12;       // group with more distinct planes => simplify it
    float facet_tol            = 1e-4f;
    float seg_eps              = 1e-5f;
    ISMFrame frame             = ISMFrame::AUTO;

    // --- simplification of dense small-triangle clusters into proxy patches ---
    float proxy_normal_cos     = 0.906f;   // region-grow tolerance (~25 deg); looser => fewer, bigger proxies
    float min_proxy_area       = 1e-4f;    // drop proxy slivers (kept as occluders)
    int   proxy_max_per_group  = 512;      // safety cap; warn if exceeded
    float flat_radius_factor   = 50.f;     // sphere-fit radius > factor*patch_extent => treat patch as flat
    int   max_order_curved     = 1;        // max curved bounces allowed in a single path
    float curved_amp_cap       = 4.0f;     // clamp curvature gain (concave focus) and flag approx
    int   convex_unknown_as_convex = 1;
};

// ============================================================
// Material impedance (complex, per octave band) keyed by MATERIAL id
// ============================================================
struct MaterialImpedance {
    std::array<float, ISM_NBANDS> Z_re;
    std::array<float, ISM_NBANDS> Z_im;
    bool loaded = false;
    MaterialImpedance(){ Z_re.fill(0.f); Z_im.fill(0.f); }
};

inline static std::vector<MaterialImpedance> ism_load_impedances(
    const std::string& imp_dir, const std::vector<std::string>& material_names)
{
    std::vector<MaterialImpedance> out(material_names.size());
    auto parse_row = [](const std::string& s, std::array<float, ISM_NBANDS>& dst)->int{
        std::istringstream iss(s); float v; int n=0;
        while(n < ISM_NBANDS && (iss >> v)){ dst[(size_t)n]=v; ++n; }
        return n;
    };
    for(size_t m=0; m<material_names.size(); ++m){
        MaterialImpedance M;
        std::string path = imp_dir;
        if(!path.empty() && path.back()!='/' && path.back()!='\\') path += '/';
        path += material_names[m] + ".imp";
        std::ifstream f(path);
        if(f){
            std::string line; std::vector<std::string> rows;
            while(std::getline(f, line)){
                const size_t a = line.find_first_not_of(" \t\r\n");
                if(a==std::string::npos) continue;
                if(line[a]=='#') continue;
                rows.push_back(line);
                if(rows.size()==2) break;
            }
            if(rows.size()>=2){
                const int nr=parse_row(rows[0],M.Z_re), ni=parse_row(rows[1],M.Z_im);
                M.loaded=(nr==ISM_NBANDS && ni==ISM_NBANDS);
                if(!M.loaded) std::fprintf(stderr,"[ISM] %s: expected %d values/line; got re=%d im=%d\n",
                                           path.c_str(),ISM_NBANDS,nr,ni);
            } else std::fprintf(stderr,"[ISM] %s: fewer than two data lines\n", path.c_str());
        }
        out[m]=M;
    }
    return out;
}

// Convenience for the solver's layer-name array: names[0]="BoundingBox", real
// materials at names[1..]. Every OBJ group has faces so lid==gid, hence names[gid+1]
// is the material of group gid -- the SAME indexing your .fit loop uses. Returns a
// table indexed by group_id; call ism_paths_for_pairs with group_to_material={} so
// reflection.material_id == group_id, and its name is names[material_id+1].
inline static std::vector<MaterialImpedance> ism_load_impedances_layered(
    const std::string& imp_dir, const std::vector<std::string>& names)
{
    if(names.size() < 2) return {};                       // only BoundingBox, or empty
    std::vector<std::string> mats(names.begin()+1, names.end());
    return ism_load_impedances(imp_dir, mats);
}

// ============================================================
// Reflector (geometry only; may be flat wall, curved proxy, or flat proxy)
// ============================================================
struct Reflector {
    vec3  n   = v3(0,0,1);
    float d   = 0.f;
    float area = 0.f;
    bool  curved = false;          // carries a finite curvature radius
    float radius = FLT_MAX;        // local radius of curvature (FLT_MAX => flat)
    int   convex = +1;             // +1 convex (defocus), -1 concave (focus)
    float hit_band = 1e-4f;        // in-facet tolerance (widened for proxies by patch sag)
    std::vector<int> tri_ids;
};

struct ISMReflection {
    int   material_id = -1;
    int   group_id    = -1;
    float cos_theta = 1.f;
    float theta     = 0.f;
    std::array<float, ISM_NBANDS> Z_re;
    std::array<float, ISM_NBANDS> Z_im;
    ISMReflection(){ Z_re.fill(0.f); Z_im.fill(0.f); }
};

struct ISMPath {
    float length = 0.f;
    int   order  = 0;
    float amp    = 1.f;
    bool  curved = false;
    bool  approx = false;
    std::vector<ISMReflection> reflections;   // source->receiver; empty for direct
};

// ============================================================
// Geometry helpers
// ============================================================
inline static vec3 ism_tri_normal_area(const Scene3D& Sc, const Tri& t, float& area_out){
    const vec3 a=Sc.vertex_transformed(t.v0), b=Sc.vertex_transformed(t.v1), c=Sc.vertex_transformed(t.v2);
    vec3 nc=cross3(sub3(b,a),sub3(c,a));
    const float len=std::sqrt(norm2_3(nc));
    area_out=0.5f*len;
    if(len>1e-20f) nc=mul3(nc,1.f/len); else nc=v3(0,0,1);
    return nc;
}
inline static vec3 ism_normalize(const vec3& v){ const float l=std::sqrt(norm2_3(v)); return (l>1e-20f)?mul3(v,1.f/l):v3(0,0,0); }
inline static vec3 ism_canon_normal(vec3 n){
    const float ax=std::fabs(n.x),ay=std::fabs(n.y),az=std::fabs(n.z); float s=1.f;
    if(az>=ax&&az>=ay) s=(n.z<0.f)?-1.f:1.f; else if(ay>=ax) s=(n.y<0.f)?-1.f:1.f; else s=(n.x<0.f)?-1.f:1.f;
    return mul3(n,s);
}
inline static vec3 ism_reflect_point(const vec3& p, const vec3& n, float d){ return sub3(p,mul3(n,2.f*(dot3(n,p)-d))); }
inline static bool ism_point_in_aabb(const vec3& p,const vec3& mn,const vec3& mx,float e){
    return p.x>=mn.x-e&&p.x<=mx.x+e&&p.y>=mn.y-e&&p.y<=mx.y+e&&p.z>=mn.z-e&&p.z<=mx.z+e;
}
inline static vec3 ism_resolve_endpoint(const Scene3D& Sc,const vec3& raw,const vec3& mn,const vec3& mx,ISMFrame f,float e){
    const vec3 w=raw, o=mul3(add3(raw,Sc.shift),Sc.scale);
    if(f==ISMFrame::SOLVER_WORLD) return w;
    if(f==ISMFrame::OBJECT_SPACE) return o;
    if(ism_point_in_aabb(w,mn,mx,e)) return w;
    if(ism_point_in_aabb(o,mn,mx,e)) return o;
    std::fprintf(stderr,"[ISM] endpoint (%.3g,%.3g,%.3g) outside scene in both frames; using solver-world.\n",
                 (double)raw.x,(double)raw.y,(double)raw.z);
    return w;
}
inline static int ism_hit_tri(const Scene3D& Sc,const std::vector<int>& tri_ids,const vec3& x,float band){
    const float b2=band*band; int best=-1; float bd2=b2;
    for(int t:tri_ids){ const Tri& tr=Sc.tris[(size_t)t];
        const float d2=point_triangle_dist2(x,Sc.vertex_transformed(tr.v0),Sc.vertex_transformed(tr.v1),Sc.vertex_transformed(tr.v2));
        if(d2<=bd2){ bd2=d2; best=t; } }
    return best;
}
inline static bool ism_occluded(const Scene3D& Sc,const std::vector<int>& tri2plane,const vec3& p0,const vec3& p1,
                                int ignoreA,int ignoreB,float eps){
    const vec3 dir=sub3(p1,p0); const int nt=(int)Sc.tris.size();
    for(int t=0;t<nt;++t){
        const int pl=tri2plane[(size_t)t];
        if(pl>=0 && (pl==ignoreA||pl==ignoreB)) continue;
        const Tri& tr=Sc.tris[(size_t)t]; float tt;
        if(ray_triangle_intersect(p0,dir,Sc.vertex_transformed(tr.v0),Sc.vertex_transformed(tr.v1),Sc.vertex_transformed(tr.v2),eps,tt))
            if(tt<1.f-eps) return true;
    }
    return false;
}

// 4x4 solve (Gaussian elimination, partial pivot). false if singular.
inline static bool ism_solve4(double A[4][4], double b[4], double x[4]){
    for(int c=0;c<4;++c){
        int piv=c; double mx=std::fabs(A[c][c]);
        for(int r=c+1;r<4;++r){ const double v=std::fabs(A[r][c]); if(v>mx){mx=v;piv=r;} }
        if(mx<1e-18) return false;
        if(piv!=c){ for(int k=0;k<4;++k) std::swap(A[piv][k],A[c][k]); std::swap(b[piv],b[c]); }
        for(int r=c+1;r<4;++r){ const double f=A[r][c]/A[c][c];
            for(int k=c;k<4;++k) A[r][k]-=f*A[c][k]; b[r]-=f*b[c]; }
    }
    for(int r=3;r>=0;--r){ double s=b[r]; for(int k=r+1;k<4;++k) s-=A[r][k]*x[k]; x[r]=s/A[r][r]; }
    return true;
}
// Least-squares sphere fit over points. false if <4 pts or planar/ill-conditioned.
inline static bool ism_fit_sphere(const std::vector<vec3>& P, vec3& c, float& R){
    const int n=(int)P.size(); if(n<4) return false;
    double A[4][4]={{0}}, b[4]={0};
    for(const auto& p:P){ const double row[4]={p.x,p.y,p.z,1.0};
        const double bb=(double)p.x*p.x+(double)p.y*p.y+(double)p.z*p.z;
        for(int i=0;i<4;++i){ for(int j=0;j<4;++j) A[i][j]+=row[i]*row[j]; b[i]+=row[i]*bb; } }
    double u[4]; if(!ism_solve4(A,b,u)) return false;
    c=v3((float)(u[0]*0.5),(float)(u[1]*0.5),(float)(u[2]*0.5));
    const double R2=u[3]+(double)c.x*c.x+(double)c.y*c.y+(double)c.z*c.z;
    if(R2<=0) return false; R=(float)std::sqrt(R2); return true;
}

// ============================================================
// Build reflectors: walls (geometry merge) + simplified curved/flat proxies
// ============================================================
inline static void ism_build_reflectors(const Scene3D& Sc, const ISMConfig& cfg,
                                        std::vector<Reflector>& planes,
                                        std::vector<int>& tri2plane,
                                        std::vector<uint8_t>& group_is_curved)
{
    planes.clear(); tri2plane.assign(Sc.tris.size(),-1);
    int max_gid=-1; for(const auto& t:Sc.tris) max_gid=std::max(max_gid,t.group_id);
    group_is_curved.assign((size_t)std::max(0,max_gid+1),0);
    if(max_gid<0) return;

    struct Proto { vec3 n; float d; float area; std::vector<int> tris; };
    std::vector<Proto> proto;        // planar walls from non-curved groups

    for(int gid=0; gid<=max_gid; ++gid){
        std::vector<Proto> local; std::vector<int> group_tris;
        for(int t=0;t<(int)Sc.tris.size();++t){
            const Tri& tr=Sc.tris[(size_t)t]; if(tr.group_id!=gid) continue;
            group_tris.push_back(t);
            float area; vec3 n=ism_canon_normal(ism_tri_normal_area(Sc,tr,area));
            const float d=dot3(n,Sc.vertex_transformed(tr.v0));
            int hit=-1;
            for(int p=0;p<(int)local.size();++p)
                if(dot3(local[(size_t)p].n,n)>cfg.coplanar_cos && std::fabs(local[(size_t)p].d-d)<cfg.coplanar_dist){hit=p;break;}
            if(hit<0) local.push_back({n,d,area,{t}});
            else { local[(size_t)hit].area+=area; local[(size_t)hit].tris.push_back(t); }
        }
        if(group_tris.empty()) continue;

        const bool dense = ((int)local.size() > cfg.max_planes_per_group);
        group_is_curved[(size_t)gid] = dense?1u:0u;

        if(!dense){
            for(auto& L:local){ if(L.area>=cfg.min_plane_area) proto.push_back(std::move(L)); }
            continue;
        }

        // ---- SIMPLIFY: region-grow this group's facets into proxy patches ----
        const int G=(int)group_tris.size();
        // global-vertex -> local facet positions, for shared-vertex adjacency
        std::unordered_map<int,std::vector<int>> v2f; v2f.reserve((size_t)G*2);
        std::vector<vec3>  fnrm((size_t)G); std::vector<float> farea((size_t)G); std::vector<vec3> fcent((size_t)G);
        for(int li=0; li<G; ++li){
            const Tri& tr=Sc.tris[(size_t)group_tris[(size_t)li]];
            float ar; fnrm[(size_t)li]=ism_tri_normal_area(Sc,tr,ar); farea[(size_t)li]=ar;
            const vec3 a=Sc.vertex_transformed(tr.v0),b=Sc.vertex_transformed(tr.v1),c=Sc.vertex_transformed(tr.v2);
            fcent[(size_t)li]=mul3(add3(add3(a,b),c),1.f/3.f);
            v2f[tr.v0].push_back(li); v2f[tr.v1].push_back(li); v2f[tr.v2].push_back(li);
        }
        std::vector<int> region((size_t)G,-1);
        int nproxies=0;
        for(int seed=0; seed<G; ++seed){
            if(region[(size_t)seed]!=-1) continue;
            std::vector<int> members; std::vector<int> stack;
            region[(size_t)seed]=1; stack.push_back(seed); members.push_back(seed);
            vec3 sumAN = mul3(fnrm[(size_t)seed], farea[(size_t)seed]);
            while(!stack.empty()){
                const int g=stack.back(); stack.pop_back();
                const Tri& tg=Sc.tris[(size_t)group_tris[(size_t)g]];
                const int vs[3]={tg.v0,tg.v1,tg.v2};
                const vec3 avg=ism_normalize(sumAN);
                for(int vi=0;vi<3;++vi){
                    auto it=v2f.find(vs[vi]); if(it==v2f.end()) continue;
                    for(int nb:it->second){
                        if(region[(size_t)nb]!=-1) continue;
                        if(dot3(fnrm[(size_t)nb], avg) > cfg.proxy_normal_cos){
                            region[(size_t)nb]=1; sumAN=add3(sumAN,mul3(fnrm[(size_t)nb],farea[(size_t)nb]));
                            stack.push_back(nb); members.push_back(nb);
                        }
                    }
                }
            }
            // finalize proxy from members
            vec3 n=ism_normalize(sumAN); float area=0.f; vec3 cw=v3(0,0,0);
            std::vector<int> tg; std::vector<vec3> pts; pts.reserve(members.size());
            for(int li:members){ area+=farea[(size_t)li]; cw=add3(cw,mul3(fcent[(size_t)li],farea[(size_t)li]));
                tg.push_back(group_tris[(size_t)li]); pts.push_back(fcent[(size_t)li]); }
            if(area<=0.f) continue;
            vec3 C=mul3(cw,1.f/area); float d=dot3(n,C);
            ++nproxies;
            if(area<cfg.min_proxy_area) continue;   // sliver: leave facets as occluders (tri2plane=-1)

            Reflector R; R.n=n; R.d=d; R.area=area; R.tri_ids=tg;

            // hit band: how far the real facets stand off this chord plane (+slack)
            float band=cfg.facet_tol;
            for(int li:members){ const Tri& tr=Sc.tris[(size_t)group_tris[(size_t)li]];
                const int vv[3]={tr.v0,tr.v1,tr.v2};
                for(int k=0;k<3;++k) band=std::max(band, std::fabs(dot3(n,Sc.vertex_transformed(vv[k]))-d)); }
            R.hit_band = band + cfg.facet_tol;

            // local curvature from a per-patch sphere fit
            vec3 sc; float sr;
            float extent=0.f; for(const auto& p:pts) extent=std::max(extent,std::sqrt(norm2_3(sub3(p,C))));
            if(ism_fit_sphere(pts,sc,sr) && sr < cfg.flat_radius_factor*std::max(extent,1e-6f)){
                R.curved=true; R.radius=sr;
                double cx=0.0; for(int li:members) cx+=(double)dot3(fnrm[(size_t)li], sub3(fcent[(size_t)li],sc));
                R.convex = (cx>0.0)?+1 : (cx<0.0)?-1 : (cfg.convex_unknown_as_convex?+1:-1);
            } else { R.curved=false; R.radius=FLT_MAX; }   // patch is effectively flat

            const int pidx=(int)planes.size();
            for(int t:R.tri_ids) tri2plane[(size_t)t]=pidx;
            planes.push_back(std::move(R));
        }
        if(nproxies>cfg.proxy_max_per_group)
            std::fprintf(stderr,"[ISM] group %d simplified into %d proxies (> %d); loosen proxy_normal_cos.\n",
                         gid,nproxies,cfg.proxy_max_per_group);
    }

    // merge coplanar walls across groups
    for(auto& P:proto){
        int hit=-1;
        for(int r=0;r<(int)planes.size();++r)
            if(!planes[(size_t)r].curved &&
               dot3(planes[(size_t)r].n,P.n)>cfg.coplanar_cos &&
               std::fabs(planes[(size_t)r].d-P.d)<cfg.coplanar_dist &&
               planes[(size_t)r].radius==FLT_MAX){ /* only merge into flat walls */
                // also require it is a wall (not a flat proxy) to avoid cross-object surprises:
                hit=r; break; }
        if(hit<0){ Reflector R; R.n=P.n; R.d=P.d; R.area=P.area; R.hit_band=cfg.facet_tol; R.tri_ids=std::move(P.tris); planes.push_back(std::move(R)); }
        else { planes[(size_t)hit].area+=P.area; auto& dst=planes[(size_t)hit].tri_ids; dst.insert(dst.end(),P.tris.begin(),P.tris.end()); }
    }
    for(int r=0;r<(int)planes.size();++r) for(int t:planes[(size_t)r].tri_ids) tri2plane[(size_t)t]=r;
}

// ============================================================
// Image tree (curved bounces capped at cfg.max_order_curved)
// ============================================================
struct ISMImageNode { vec3 image; int plane=-1; int parent=-1; int order=0; int curved_count=0; };

inline static void ism_build_images(const std::vector<Reflector>& planes,const vec3& source,
                                    const ISMConfig& cfg, std::vector<ISMImageNode>& nodes){
    nodes.clear(); nodes.push_back({source,-1,-1,0,0});
    size_t begin=0;
    for(int order=1; order<=cfg.max_order; ++order){
        const size_t end=nodes.size();
        for(size_t pi=begin; pi<end; ++pi){
            const ISMImageNode parent=nodes[pi];
            for(int p=0;p<(int)planes.size();++p){
                if(p==parent.plane) continue;
                const Reflector& R=planes[(size_t)p];
                if(R.curved && parent.curved_count>=cfg.max_order_curved) continue;   // cap curved bounces
                if(std::fabs(dot3(R.n,parent.image)-R.d)<1e-6f) continue;
                ISMImageNode c; c.image=ism_reflect_point(parent.image,R.n,R.d);
                c.plane=p; c.parent=(int)pi; c.order=order; c.curved_count=parent.curved_count+(R.curved?1:0);
                nodes.push_back(c);
            }
        }
        begin=end;
    }
}

// ============================================================
// Validate a leaf; outputs hit tri + incidence cosine per bounce (source->receiver),
// total length, and the accumulated curvature amplitude factor.
// ============================================================
inline static bool ism_validate_path(const Scene3D& Sc,const std::vector<Reflector>& planes,
                                     const std::vector<int>& tri2plane,const std::vector<ISMImageNode>& nodes,
                                     int leaf,const vec3& source,const vec3& receiver,const ISMConfig& cfg,
                                     std::vector<int>& hit_tri_out,std::vector<float>& cos_theta_out,
                                     float& length_out,float& amp_out,bool& approx_out)
{
    hit_tri_out.clear(); cos_theta_out.clear();
    std::vector<int> planes_seq, hit_ls; std::vector<vec3> pts;   // leaf..first

    vec3 point=receiver; int node=leaf;
    while(nodes[(size_t)node].plane>=0){
        const ISMImageNode& nd=nodes[(size_t)node];
        const Reflector& R=planes[(size_t)nd.plane];
        const vec3 dir=sub3(nd.image,point); const float denom=dot3(R.n,dir);
        if(std::fabs(denom)<1e-12f) return false;
        const float t=(R.d-dot3(R.n,point))/denom;
        if(t<=cfg.seg_eps||t>=1.f-cfg.seg_eps) return false;
        const vec3 refl=add3(point,mul3(dir,t));
        const int htri=ism_hit_tri(Sc,R.tri_ids,refl,R.hit_band);
        if(htri<0) return false;
        planes_seq.push_back(nd.plane); hit_ls.push_back(htri); pts.push_back(refl);
        point=refl; node=nd.parent;
    }
    const int m=(int)pts.size();
    { const int ignA=(m>0)?planes_seq[0]:-1; const vec3 next=(m>0)?pts[0]:source;
      if(ism_occluded(Sc,tri2plane,receiver,next,ignA,-1,cfg.seg_eps)) return false; }
    for(int s=0;s<m;++s){ const vec3 a=pts[(size_t)s]; const int ia=planes_seq[(size_t)s];
        const vec3 b=(s+1<m)?pts[(size_t)(s+1)]:source; const int ib=(s+1<m)?planes_seq[(size_t)(s+1)]:-1;
        if(ism_occluded(Sc,tri2plane,a,b,ia,ib,cfg.seg_eps)) return false; }

    length_out=std::sqrt(norm2_3(sub3(receiver,nodes[(size_t)leaf].image)));

    // incidence cosine per bounce (leaf..first)
    std::vector<float> cos_ls((size_t)m,1.f);
    for(int s=0;s<m;++s){ const vec3 P=pts[(size_t)s]; const vec3 prev=(s==0)?receiver:pts[(size_t)(s-1)];
        const vec3 din=ism_normalize(sub3(P,prev)); const vec3 n=planes[(size_t)planes_seq[(size_t)s]].n;
        cos_ls[(size_t)s]=std::min(1.f,std::max(0.f,std::fabs(dot3(din,n)))); }

    // physical polyline source -> bounces -> receiver, cumulative lengths
    std::vector<vec3> V; V.reserve((size_t)m+2);
    V.push_back(source); for(int k=m-1;k>=0;--k) V.push_back(pts[(size_t)k]); V.push_back(receiver);
    std::vector<float> cum((size_t)V.size(),0.f);
    for(size_t i=1;i<V.size();++i) cum[i]=cum[i-1]+std::sqrt(norm2_3(sub3(V[i],V[i-1])));
    const float total=cum.back();

    // accumulate curvature amplitude over curved bounces
    float amp=1.f; bool approx=false;
    for(int k=1;k<=m;++k){
        const int idx=planes_seq[(size_t)(m-k)];   // reflector at physical bounce k
        const Reflector& R=planes[(size_t)idx];
        if(!R.curved || R.radius==FLT_MAX) continue;
        const double di=cum[(size_t)k], dr=(double)total-cum[(size_t)k];
        const double ct=std::max(1e-3,(double)cos_ls[(size_t)(m-k)]);
        const double sgn=(R.convex>=0)?-1.0:+1.0;
        const double inv=1.0/di + sgn*2.0/((double)R.radius*ct);
        const double di_p=(std::fabs(inv)>1e-12)?1.0/inv:1e12;
        double a=(di+dr)/(std::fabs(di_p)+dr);
        if(!(a>0.0)||a>(double)cfg.curved_amp_cap){ a=std::min(std::max(a,1e-3),(double)cfg.curved_amp_cap); approx=true; }
        amp*=(float)a;
    }
    amp_out=amp; approx_out=approx;

    hit_tri_out.resize((size_t)m); cos_theta_out.resize((size_t)m);
    for(int s=0;s<m;++s){ const int idx=m-1-s; hit_tri_out[(size_t)s]=hit_ls[(size_t)idx]; cos_theta_out[(size_t)s]=cos_ls[(size_t)idx]; }
    return true;
}

// ============================================================
// Top-level: paths for every (source, receiver) pair
// ============================================================
inline static void ism_paths_for_pairs(
    const Scene3D& Sc,
    const std::vector<float>& src_arr,
    const std::vector<float>& rec_arr,
    const std::vector<int>& group_to_material,
    const std::vector<MaterialImpedance>& impedance_by_material,
    int max_order,                                   // user input: max reflection order
    std::vector<std::vector<std::vector<ISMPath>>>& out_paths,
    ISMConfig cfg = ISMConfig(),                     // optional tuning knobs; omit for defaults
    std::vector<Reflector>* planes_out = nullptr,
    std::vector<uint8_t>*   group_is_curved_out = nullptr)
{
    cfg.max_order = (max_order < 0) ? 0 : max_order; // call-site input wins

    std::vector<Reflector> planes; std::vector<int> tri2plane; std::vector<uint8_t> curvedflag;
    ism_build_reflectors(Sc,cfg,planes,tri2plane,curvedflag);
    if(planes_out)*planes_out=planes;
    if(group_is_curved_out)*group_is_curved_out=curvedflag;

    vec3 mn,mx; mesh_bbox_transformed(Sc,mn,mx);
    const float aabb_eps=0.02f*std::sqrt(norm2_3(sub3(mx,mn)));

    auto material_of=[&](int gid)->int{
        if(gid<0) return -1;
        if(!group_to_material.empty()) return (gid<(int)group_to_material.size())?group_to_material[(size_t)gid]:-1;
        return gid;
    };
    auto make_refl=[&](int gid,float ct)->ISMReflection{
        ISMReflection R; R.group_id=gid; R.material_id=material_of(gid);
        R.cos_theta=ct; R.theta=std::acos(std::min(1.f,std::max(0.f,ct)));
        const int mid=R.material_id;
        if(mid>=0&&mid<(int)impedance_by_material.size()){ R.Z_re=impedance_by_material[(size_t)mid].Z_re; R.Z_im=impedance_by_material[(size_t)mid].Z_im; }
        return R;
    };

    const int Ns=(int)(src_arr.size()/3), Nr=(int)(rec_arr.size()/3);
    out_paths.assign((size_t)Ns,std::vector<std::vector<ISMPath>>((size_t)Nr));

    for(int s=0;s<Ns;++s){
        const vec3 source=ism_resolve_endpoint(Sc,v3(src_arr[(size_t)(3*s+0)],src_arr[(size_t)(3*s+1)],src_arr[(size_t)(3*s+2)]),mn,mx,cfg.frame,aabb_eps);
        std::vector<ISMImageNode> nodes; ism_build_images(planes,source,cfg,nodes);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for(int r=0;r<Nr;++r){
            const vec3 receiver=ism_resolve_endpoint(Sc,v3(rec_arr[(size_t)(3*r+0)],rec_arr[(size_t)(3*r+1)],rec_arr[(size_t)(3*r+2)]),mn,mx,cfg.frame,aabb_eps);
            std::vector<ISMPath>& dst=out_paths[(size_t)s][(size_t)r];

            if(!ism_occluded(Sc,tri2plane,source,receiver,-1,-1,cfg.seg_eps)){
                ISMPath p0; p0.length=std::sqrt(norm2_3(sub3(receiver,source))); dst.push_back(std::move(p0));
            }
            std::vector<int> hit_tris; std::vector<float> cchain; float len,amp; bool approx;
            for(int leaf=1; leaf<(int)nodes.size(); ++leaf){
                if(ism_validate_path(Sc,planes,tri2plane,nodes,leaf,source,receiver,cfg,hit_tris,cchain,len,amp,approx)){
                    ISMPath path; path.length=len; path.order=(int)hit_tris.size(); path.amp=amp; path.approx=approx;
                    path.reflections.reserve(hit_tris.size());
                    for(size_t k=0;k<hit_tris.size();++k){
                        const int gid=Sc.tris[(size_t)hit_tris[k]].group_id;
                        path.reflections.push_back(make_refl(gid,cchain[k]));
                    }
                    // mark path curved if any bounce used a curved reflector (amp!=1 or flagged)
                    path.curved = (amp!=1.f) || approx;
                    dst.push_back(std::move(path));
                }
            }
            std::sort(dst.begin(),dst.end(),[](const ISMPath& a,const ISMPath& b){return a.length<b.length;});
        }
    }
}

// ============================================================
// Flat result view: exactly the lists needed to build RIRs.
// ============================================================
// One valid image source for a (source, receiver) pair. All four per-surface
// lists have the same length = reflection order, in source->receiver order.
struct ISMImagePath {
    float distance = 0.f;                                  // image-source -> receiver (total unfolded), 2a
    std::vector<float> cos_inc;                            // incidence cosine per crossed surface, 2b
    std::vector<float> angle_rad;                          // same angle in radians, 2b
    std::vector<std::array<float, ISM_NBANDS>> Z_re;       // impedance (real) per surface, per band, 2c
    std::vector<std::array<float, ISM_NBANDS>> Z_im;       // impedance (imag) per surface, per band, 2c
    float amp = 1.f;                                       // curvature amplitude factor (1 unless curved)
};

struct ISMResult {
    int Ns = 0, Nr = 0;
    std::vector<float>                      src_pressure;  // [Ns]: initial sound pressure per real source (val_s)
    std::vector<float>                      direct;        // [Ns*Nr]: direct distance, 0.0f if invalid (1)
    std::vector<std::vector<ISMImagePath>>  images;        // [Ns*Nr]: valid image sources (2)
    inline int idx(int s, int r) const { return s*Nr + r; }
    inline float pressure_at(int s) const { return src_pressure[(size_t)s]; }
    inline float direct_at(int s, int r) const { return direct[(size_t)(s*Nr + r)]; }
    inline const std::vector<ISMImagePath>& images_at(int s, int r) const { return images[(size_t)(s*Nr + r)]; }
};

// One-call entry point. Runs the ISM and returns the flat lists above.
// src_pressure is the per-source initial pressure (val_s, the 4th column of src.txt);
// pass {} to default every source to 1.0. It is NOT baked into distances/amplitudes --
// it is returned per source so you scale that source's contributions: p = val_s * amp/dist * Rprod.
inline static ISMResult ism_solve(
    const Scene3D& Sc,
    const std::vector<float>& src_arr,
    const std::vector<float>& rec_arr,
    const std::vector<int>& group_to_material,            // {} => material_id == group_id
    const std::vector<MaterialImpedance>& impedance_by_material,
    int max_order,
    const std::vector<float>& src_pressure = {},
    ISMConfig cfg = ISMConfig())
{
    std::vector<std::vector<std::vector<ISMPath>>> out;
    ism_paths_for_pairs(Sc, src_arr, rec_arr, group_to_material, impedance_by_material, max_order, out, cfg);

    ISMResult R;
    R.Ns = (int)out.size();
    R.Nr = R.Ns ? (int)out[0].size() : 0;
    R.direct.assign((size_t)R.Ns*R.Nr, 0.0f);
    R.images.assign((size_t)R.Ns*R.Nr, {});

    // per-source pressure: copy if supplied (must match Ns), else default to 1.0
    R.src_pressure.assign((size_t)R.Ns, 1.0f);
    if(!src_pressure.empty()){
        if((int)src_pressure.size() != R.Ns)
            std::fprintf(stderr,"[ISM] src_pressure size %zu != number of sources %d; using as far as it goes.\n",
                         src_pressure.size(), R.Ns);
        const int n = std::min((int)src_pressure.size(), R.Ns);
        for(int s=0;s<n;++s) R.src_pressure[(size_t)s] = src_pressure[(size_t)s];
    }

    for(int s=0; s<R.Ns; ++s){
        for(int r=0; r<R.Nr; ++r){
            const size_t key = (size_t)s*R.Nr + r;
            for(const ISMPath& p : out[(size_t)s][(size_t)r]){
                if(p.reflections.empty()){      // the direct path
                    R.direct[key] = p.length;   // (stays 0.0f if no direct path was valid)
                    continue;
                }
                ISMImagePath ip;
                ip.distance = p.length;
                ip.amp      = p.amp;
                ip.cos_inc.reserve(p.reflections.size());
                ip.angle_rad.reserve(p.reflections.size());
                ip.Z_re.reserve(p.reflections.size());
                ip.Z_im.reserve(p.reflections.size());
                for(const ISMReflection& rf : p.reflections){
                    ip.cos_inc.push_back(rf.cos_theta);
                    ip.angle_rad.push_back(rf.theta);
                    ip.Z_re.push_back(rf.Z_re);
                    ip.Z_im.push_back(rf.Z_im);
                }
                R.images[key].push_back(std::move(ip));
            }
        }
    }
    return R;
}
