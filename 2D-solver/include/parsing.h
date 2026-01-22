#pragma once

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// ============================================================
// Basic types
// ============================================================

struct vec2 { float x=0.f, y=0.f; };

struct Poly {
    std::string name;                 // "Boundary" or obstacle name (or layer name)
    std::vector<int>   face;          // vertex indices in winding order (0-based)
    std::vector<float> lines;         // optional: flat triples (a,b,c) per edge (unit normal)
};

struct Scene2D {
    std::vector<vec2>  vertices;      // shifted so min corner is at (0,0)
    std::vector<Poly>  polys;
    float  Lx=0.f, Ly=0.f;            // extents after shifting
    int    idx_boundary=-1;           // poly index for domain boundary
};

// ============================================================
// Small helpers
// ============================================================

inline static vec2 sub2(const vec2&a,const vec2&b){ return vec2{a.x-b.x,a.y-b.y}; }
inline static float dot2(const vec2&a,const vec2&b){ return a.x*b.x + a.y*b.y; }
inline static float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }

inline static void trim(std::string& s){
    auto is_ws = [](unsigned char c){ return c==' '||c=='\t'||c=='\r'||c=='\n'; };
    while(!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
    size_t i=0; while(i<s.size() && is_ws((unsigned char)s[i])) i++;
    if(i>0) s.erase(0,i);
}

inline static std::vector<std::string> split_ws(const std::string& s){
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string t;
    while(iss>>t) out.push_back(t);
    return out;
}

inline static int   parse_int(const std::string& s){ return std::atoi(s.c_str()); }
inline static float parse_float(const std::string& s){ return std::strtof(s.c_str(), nullptr); }

inline static std::string lower_copy(std::string s){
    for(char& c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// ============================================================
// Geometry helpers
// ============================================================

inline static float dist_point_seg(const vec2& p, const vec2& A, const vec2& B){
    vec2 AB = sub2(B,A);
    vec2 AP = sub2(p,A);
    float denom = AB.x*AB.x + AB.y*AB.y;
    float t = denom>0.0f ? dot2(AP,AB)/denom : 0.0f;
    t = clampf(t, 0.0f, 1.0f);
    vec2 C{ A.x + t*AB.x, A.y + t*AB.y };
    vec2 d = sub2(p,C);
    return std::sqrt(d.x*d.x + d.y*d.y);
}

// Ray-casting point-in-polygon
inline static int point_in_poly(const vec2& p, const Scene2D& S, const Poly& P){
    int inside = 0;
    const int m = (int)P.face.size();
    for(int i=0, j=m-1; i<m; j=i++){
        const vec2 A = S.vertices[P.face[i]];
        const vec2 B = S.vertices[P.face[j]];
        const bool cond = ((A.y > p.y) != (B.y > p.y));
        if(cond){
            const float xint = (B.x - A.x) * (p.y - A.y) / (B.y - A.y + 1e-30f) + A.x;
            if(p.x < xint) inside ^= 1;
        }
    }
    return inside;
}

inline static float dist_to_poly_edges(const vec2& p, const Scene2D& S, const Poly& P){
    float dmin = FLT_MAX;
    const int m = (int)P.face.size();
    for(int e=0; e<m; ++e){
        const vec2 A = S.vertices[P.face[e]];
        const vec2 B = S.vertices[P.face[(e+1==m)?0:e+1]];
        dmin = std::min(dmin, dist_point_seg(p,A,B));
    }
    return dmin;
}

// Optional: build outward line coefficients per edge (kept, harmless)
inline static void build_poly_lines(Scene2D& Sc, Poly& P){
    const int m = (int)P.face.size();
    P.lines.clear();
    P.lines.reserve((size_t)m*3);

    for(int e=0; e<m; ++e){
        const vec2 A = Sc.vertices[P.face[e]];
        const vec2 B = Sc.vertices[P.face[(e+1==m)?0:e+1]];
        const float dx = B.x - A.x;
        const float dy = B.y - A.y;

        float a =  dy;
        float b = -dx;
        float nrm = std::sqrt(a*a + b*b);
        if(nrm <= 1e-30f){ a=1.f; b=0.f; nrm=1.f; }
        a /= nrm; b /= nrm;
        const float c = -(a*A.x + b*A.y);

        P.lines.push_back(a);
        P.lines.push_back(b);
        P.lines.push_back(c);
    }
}

// ============================================================
// OBJ parsing (PUBLIC API expects parse_obj2d)
// ============================================================

inline static bool parse_obj_2d(
    const std::string& path,
    Scene2D& Sc,
    const std::string& boundary_name_hint = "boundary")
{
    std::ifstream f(path);
    if(!f.good()){
        std::fprintf(stderr,"[OBJ] cannot open: %s\n", path.c_str());
        return false;
    }

    Sc.vertices.clear();
    Sc.polys.clear();
    Sc.Lx = Sc.Ly = 0.f;
    Sc.idx_boundary = -1;

    std::string line;
    Poly cur;
    bool have_cur = false;
    std::string cur_name = "unnamed";

    auto flush_poly = [&](){
        if(!have_cur) return;
        if(cur.face.size() >= 3) Sc.polys.push_back(cur);
        cur = Poly{};
        have_cur = false;
    };

    while(std::getline(f,line)){
        trim(line);
        if(line.empty() || line[0]=='#') continue;

        auto tok = split_ws(line);
        if(tok.empty()) continue;

        if(tok[0] == "o" || tok[0] == "g"){
            flush_poly();
            cur_name = (tok.size()>=2)? tok[1] : "unnamed";
            cur.name = cur_name;
            have_cur = true;
        }
        else if(tok[0] == "v"){
            if(tok.size() < 3) continue;
            const float x = parse_float(tok[1]);
            const float y = parse_float(tok[2]);
            Sc.vertices.push_back({x,y});
        }
        else if(tok[0] == "f"){
            if(tok.size() < 4) continue;
            if(!have_cur){
                cur.name = cur_name;
                have_cur = true;
            }
            for(size_t k=1; k<tok.size(); ++k){
                std::string t = tok[k];
                size_t slash = t.find('/');
                if(slash != std::string::npos) t = t.substr(0,slash);
                int idx = parse_int(t);
                if(idx > 0) idx -= 1; // OBJ is 1-based
                cur.face.push_back(idx);
            }
        }
    }
    flush_poly();

    // Shift to origin + compute extents
    float xmin=+FLT_MAX, ymin=+FLT_MAX;
    float xmax=-FLT_MAX, ymax=-FLT_MAX;
    for(const auto& v: Sc.vertices){
        xmin = std::min(xmin, v.x);
        ymin = std::min(ymin, v.y);
        xmax = std::max(xmax, v.x);
        ymax = std::max(ymax, v.y);
    }
    if(!Sc.vertices.empty()){
        for(auto& v: Sc.vertices){
            v.x -= xmin;
            v.y -= ymin;
        }
        Sc.Lx = xmax - xmin;
        Sc.Ly = ymax - ymin;
    }

    // Build line coeffs + detect boundary
    const std::string bh = lower_copy(boundary_name_hint);
    for(int pid=0; pid<(int)Sc.polys.size(); ++pid){
        build_poly_lines(Sc, Sc.polys[pid]);
        if(Sc.idx_boundary < 0){
            const std::string nm = lower_copy(Sc.polys[pid].name);
            if(nm.find(bh) != std::string::npos) Sc.idx_boundary = pid;
        }
    }

    return true;
}

// ---- PUBLIC wrapper with the exact name your main.cu expects
inline static int parse_obj2d(const char* path, Scene2D* sc){
    if(!path || !sc) return 0;
    return parse_obj_2d(std::string(path), *sc, "boundary") ? 1 : 0;
}

// ============================================================
// SDF build (PUBLIC API expects build_sdf)
// ============================================================

inline static int build_sdf_cpu(
    const Scene2D& Sc,
    int Nx,
    int Ny,
    float h,
    float** S_out)
{
    if(Nx<=0 || Ny<=0 || !S_out) return 0;

    const size_t N = (size_t)Nx * (size_t)Ny;
    float* S = (float*)std::malloc(N*sizeof(float));
    if(!S){
        std::fprintf(stderr,"[SDF] malloc failed\n");
        return 0;
    }

    // Node-based SDF at p=(i*h, j*h)
    for(int j=0; j<Ny; ++j){
        for(int i=0; i<Nx; ++i){
            const vec2 p{ i*h, j*h };
            float dmin = FLT_MAX;
            int inside_any = 0;

            for(int pid=0; pid<(int)Sc.polys.size(); ++pid){
                if(pid == Sc.idx_boundary) continue;
                const Poly& P = Sc.polys[pid];
                dmin = std::min(dmin, dist_to_poly_edges(p, Sc, P));
                if(point_in_poly(p, Sc, P)) inside_any = 1;
            }

            const float s = inside_any ? -dmin : +dmin;
            S[j*Nx + i] = s;
        }
    }

    *S_out = S;
    return 1;
}

// ---- PUBLIC wrapper with the exact name your main.cu expects
inline static void build_sdf(Scene2D* sc, float h, int* Nx, int* Ny, float* Lx, float* Ly, float** S){
    if(!sc || !Nx || !Ny || !Lx || !Ly || !S) return;

    *Lx = sc->Lx;
    *Ly = sc->Ly;

    // Node grid: include both ends => +1
    *Nx = (int)std::ceil((*Lx)/h) + 1;
    *Ny = (int)std::ceil((*Ly)/h) + 1;

    (void)build_sdf_cpu(*sc, *Nx, *Ny, h, S);
}

// ============================================================
// Layer semantics (THIS is your requested change)
// ============================================================
//
// Layer grouping key: exact Poly.name
// - Boundary is layer 0 (if present)
// - Non-boundary layers are 1.. in first-appearance order
//
// This matches your main logic: counter_obj = 1; // 0 is boundary
// ============================================================

inline static const std::string& scene_layer_key(const Poly& P){
    return P.name;
}

inline static void scene_layer_names(const Scene2D& Sc, std::vector<std::string>& names){
    names.clear();
    names.reserve(Sc.polys.size());

    // boundary is always id 0 if present
    if(Sc.idx_boundary >= 0 && Sc.idx_boundary < (int)Sc.polys.size()){
        names.push_back(Sc.polys[(size_t)Sc.idx_boundary].name);
    }

    // unique non-boundary layer keys in first appearance order
    for(int pid=0; pid<(int)Sc.polys.size(); ++pid){
        if(pid == Sc.idx_boundary) continue;
        const std::string& key = scene_layer_key(Sc.polys[(size_t)pid]);

        bool found = false;
        for(const std::string& s : names){
            if(s == key){ found = true; break; }
        }
        if(!found) names.push_back(key);
    }
}

inline static int scene_layer_count(const Scene2D& Sc){
    std::vector<std::string> names;
    scene_layer_names(Sc, names);
    return (int)names.size();
}

// Backward-compatible helper names you showed earlier (kept)
// They now mean "layer count", "max layer id", "layer names".
inline static int scene_object_count(const Scene2D& Sc){
    return scene_layer_count(Sc);
}
inline static int scene_max_object_id(const Scene2D& Sc){
    const int n = scene_layer_count(Sc);
    return n>0 ? (n-1) : -1;
}
inline static void scene_object_names(const Scene2D& Sc, std::vector<std::string>& names){
    scene_layer_names(Sc, names);
}

// poly_id -> layer_id (size == Sc.polys.size())
inline static void scene_poly_to_layer_id(const Scene2D& Sc, std::vector<int>& poly2layer){
    std::vector<std::string> layer_names;
    scene_layer_names(Sc, layer_names);

    poly2layer.assign(Sc.polys.size(), -1);

    // boundary poly maps to 0 (if present)
    if(Sc.idx_boundary >= 0 && Sc.idx_boundary < (int)Sc.polys.size()){
        poly2layer[(size_t)Sc.idx_boundary] = 0;
    }

    for(int pid=0; pid<(int)Sc.polys.size(); ++pid){
        if(pid == Sc.idx_boundary) continue;
        const std::string& key = scene_layer_key(Sc.polys[(size_t)pid]);

        int lid = -1;
        for(int k=0; k<(int)layer_names.size(); ++k){
            if(layer_names[(size_t)k] == key){ lid = k; break; }
        }
        poly2layer[(size_t)pid] = lid;
    }
}

// ============================================================
// OID dense tagger (OID == LAYER ID)
// ============================================================

static constexpr uint8_t OID_NONE_U8 = 0xFF;

struct NearestPolyResult{
    int   poly_id=-1;
    float dist=FLT_MAX;
};

inline static NearestPolyResult nearest_poly_at_point(const Scene2D& Sc, const vec2& p){
    NearestPolyResult out;
    const int n = (int)Sc.polys.size();
    for(int pid=0; pid<n; ++pid){
        if(pid == Sc.idx_boundary) continue;
        const Poly& P = Sc.polys[pid];
        const float d = dist_to_poly_edges(p, Sc, P);
        if(d < out.dist){
            out.dist = d;
            out.poly_id = pid;
        }
    }
    return out;
}

inline static void grad_S_central(const float* S, int Nx, int Ny, int i, int j, float h, float& gx, float& gy){
    const int ip = std::min(i+1, Nx-1);
    const int im = std::max(i-1, 0);
    const int jp = std::min(j+1, Ny-1);
    const int jm = std::max(j-1, 0);
    gx = 0.5f*(S[j*Nx + ip] - S[j*Nx + im]) / h;
    gy = 0.5f*(S[jp*Nx + i] - S[jm*Nx + i]) / h;
}

// Build oid in the immersed-BC band: s>=0 and |s|<=tol
// IMPORTANT: oid stores layer id (grouped by name), not raw poly id.
inline static int build_oid_dense_cpu(
    const Scene2D& Sc,
    const float*   S,
    int            Nx,
    int            Ny,
    float          h,
    float          tol,
    uint8_t**      OID_out)
{
    if(!S || Nx<=0 || Ny<=0 || !OID_out) return 0;

    const size_t N = (size_t)Nx * (size_t)Ny;
    uint8_t* oid = (uint8_t*)std::malloc(N);
    if(!oid){ std::fprintf(stderr,"[OID] malloc failed\n"); return 0; }
    std::memset(oid, OID_NONE_U8, N);

    std::vector<int> poly2layer;
    scene_poly_to_layer_id(Sc, poly2layer);

    for(int j=1; j<Ny-1; ++j){
        for(int i=1; i<Nx-1; ++i){
            const int Cij = j*Nx + i;
            const float s = S[Cij];
            const float d = std::fabs(s);
            if(!(s >= 0.0f && d <= tol)) continue;

            float gx, gy;
            grad_S_central(S, Nx, Ny, i, j, h, gx, gy);
            const float gn2 = gx*gx + gy*gy;
            if(gn2 <= 1e-24f) continue;

            const float inv_gn = 1.0f/std::sqrt(gn2);
            const float nx = gx * inv_gn;
            const float ny = gy * inv_gn;

            const float Cx = i * h;
            const float Cy = j * h;
            const vec2 W{ Cx - s * nx, Cy - s * ny };

            NearestPolyResult np = nearest_poly_at_point(Sc, W);
            if(np.poly_id < 0) continue;

            const int lid = (np.poly_id < (int)poly2layer.size()) ? poly2layer[(size_t)np.poly_id] : -1;
            if(lid >= 0 && lid <= 254) oid[(size_t)Cij] = (uint8_t)lid;
        }
    }

    *OID_out = oid;
    return 1;
}

// ============================================================
// Scene free (PUBLIC API expects scene_free)
// ============================================================

inline static void scene_free(Scene2D* sc){
    if(!sc) return;
    sc->vertices.clear();
    sc->polys.clear();
    sc->Lx = sc->Ly = 0.f;
    sc->idx_boundary = -1;
}
