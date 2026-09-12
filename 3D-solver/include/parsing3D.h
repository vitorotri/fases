#pragma once
/*
parsing3D.h
-----------
- Loads 3D Wavefront OBJ meshes (v / f) into a triangle soup.
- Computes a signed distance field (SDF) on a 3D Cartesian grid.
- Optional CUDA kernel for SDF evaluation (naive O(Ntri) per voxel).

Scope:
- Supports: v, f, o, g (names recorded)
- Ignores: vt, vn, usemtl, mtllib, smoothing groups
- Polygons triangulated by fan.
- Sign via parity ray casting along +X (watertight meshes).
*/

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <climits>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#ifdef __CUDACC__
  #include <cuda_runtime.h>
#endif

// ============================================================
// CUDA-safe inline macro for tiny helpers
// ============================================================
#ifdef __CUDACC__
  #define HD __host__ __device__ __forceinline__
#else
  #define HD inline
#endif

__host__ __device__ size_t flat(int i, int j, int Nx){
    return (j * Nx) + i;
}

// ============================================================
// Basic math types (CUDA-safe)
// ============================================================

struct vec3 { float x=0.f, y=0.f, z=0.f; };

HD static vec3 v3(float x,float y,float z){ vec3 r; r.x=x; r.y=y; r.z=z; return r; }
HD static vec3 add3(const vec3&a,const vec3&b){ return v3(a.x+b.x,a.y+b.y,a.z+b.z); }
HD static vec3 sub3(const vec3&a,const vec3&b){ return v3(a.x-b.x,a.y-b.y,a.z-b.z); }
HD static vec3 mul3(const vec3&a,float s){ return v3(a.x*s,a.y*s,a.z*s); }
HD static float dot3(const vec3&a,const vec3&b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
HD static vec3 cross3(const vec3&a,const vec3&b){
    return v3(a.y*b.z - a.z*b.y,
              a.z*b.x - a.x*b.z,
              a.x*b.y - a.y*b.x);
}
HD static float norm2_3(const vec3&a){ return dot3(a,a); }

// ============================================================
// Mesh representation
// ============================================================

struct Tri {
    int v0=-1, v1=-1, v2=-1;     // vertex indices (0-based)
    int group_id=-1;             // group/object id
};

struct Mesh3D {
    std::vector<vec3> vertices;
    std::vector<Tri>  tris;

    // group/object names, group_id == index in this vector
    std::vector<std::string> group_names;

    vec3 bmin = v3(0,0,0);
    vec3 bmax = v3(0,0,0);

    vec3 shift = v3(0,0,0);
    float scale = 1.f;

    inline vec3 vertex_transformed(int i) const {
        const vec3 v = vertices[(size_t)i];
        return mul3(add3(v, shift), scale);
    }
};

// Surface area per group/material [m^2] (computed on transformed vertices,
// so shift/scale are accounted for). total gets the summed area if non-null.
inline static std::vector<float> mesh_group_areas(const Mesh3D& mesh, float* total = nullptr){
    std::vector<float> A(mesh.group_names.size(), 0.f);
    float tot = 0.f;
    for (const Tri& t : mesh.tris){
        const vec3 a = mesh.vertex_transformed(t.v0);
        const vec3 b = mesh.vertex_transformed(t.v1);
        const vec3 c = mesh.vertex_transformed(t.v2);
        const float ar = 0.5f * std::sqrt(norm2_3(cross3(sub3(b,a), sub3(c,a))));
        if (t.group_id >= 0 && t.group_id < (int)A.size()) A[(size_t)t.group_id] += ar;
        tot += ar;
    }
    if (total) *total = tot;
    return A;
}

// Number of air voxels: entries of S with S > 0 (interior).
inline static size_t count_air_voxels(const float* S, size_t N){
    size_t n = 0;
    for (size_t i = 0; i < N; i++) if (S[i] > 0.f) n++;
    return n;
}

// Ghost band added on EACH side of the mesh bbox so surfaces flush with the domain edge are
// still covered by the WENO5 stencil. The grid dims (Nx.., Lx..) INCLUDE it; the room ("air")
// dims strip it. pad = PARSE3D_GHOST*h per side, so the grid is 2*PARSE3D_GHOST cells / 2*pad m
// wider per axis than the room. (V = n_air*h^3 already excludes it: the ghost is outside the
// walls where S<0, so it is not counted.)
static const int PARSE3D_GHOST = 3;

// Air-only (room, no ghost) cell counts from the padded grid dims.
inline static void air_dims(int Nx,int Ny,int Nz, int* Nx_air,int* Ny_air,int* Nz_air){
    if(Nx_air) *Nx_air = Nx - 2*PARSE3D_GHOST;
    if(Ny_air) *Ny_air = Ny - 2*PARSE3D_GHOST;
    if(Nz_air) *Nz_air = Nz - 2*PARSE3D_GHOST;
}

// ============================================================
// OBJ parsing helpers
// ============================================================

inline static void trim(std::string& s){
    auto is_ws = [](unsigned char c){ return c==' '||c=='\t'||c=='\r'||c=='\n'; };
    while(!s.empty() && is_ws((unsigned char)s.front())) s.erase(s.begin());
    while(!s.empty() && is_ws((unsigned char)s.back())) s.pop_back();
}
inline static bool starts_with(const std::string& s, const char* pfx){
    return s.rfind(pfx, 0) == 0;
}

inline static bool parse_obj_index_token(const std::string& tok, int nverts, int& out_vi){
    int vi = 0;
    {
        size_t slash = tok.find('/');
        std::string a = (slash==std::string::npos) ? tok : tok.substr(0, slash);
        if(a.empty()) return false;
        vi = std::atoi(a.c_str());
    }
    if(vi > 0) vi = vi - 1;
    else if(vi < 0) vi = nverts + vi;
    else return false;

    if(vi < 0 || vi >= nverts) return false;
    out_vi = vi;
    return true;
}

// Loads OBJ into Mesh3D triangle soup.
// Records o/g names into mesh.group_names.
inline static bool load_obj_mesh3d(const std::string& path, Mesh3D& mesh, std::string* err=nullptr){
    std::ifstream in(path);
    if(!in){
        if(err) *err = "Could not open OBJ: " + path;
        return false;
    }

    mesh.vertices.clear();
    mesh.tris.clear();
    mesh.group_names.clear();

    // OBJ exporters often emit both:
    //     o chair1
    //     g chair1
    // for the same object. The old parser created one layer for every o/g line,
    // so every object appeared twice. Do NOT create a group at o/g time. Keep a
    // pending resolved name, create the actual layer only when the first face is
    // read, and reuse an existing layer id if the same name appears again.
    std::unordered_map<std::string, int> group_id_by_name;
    int current_group = -1;
    std::string current_group_name = "unnamed";
    bool have_current_group_name = false;

    auto resolve_group_name = [](const std::string& obj_line) -> std::string {
        std::istringstream ss(obj_line);
        std::string tag;
        ss >> tag;

        std::vector<std::string> parts;
        std::string tok;
        while(ss >> tok) parts.push_back(tok);

        // Match the GUI parser:
        //   "g Mesh1 ceiling Model" -> "ceiling"
        //   "o chair1" / "g chair1" -> "chair1"
        if(parts.size() >= 2) return parts[1];
        if(parts.size() == 1) return parts[0];
        return std::string("unnamed");
    };

    auto ensure_current_group = [&]() -> int {
        const std::string name = have_current_group_name ? current_group_name : std::string("unnamed");
        auto it = group_id_by_name.find(name);
        if(it != group_id_by_name.end()) {
            current_group = it->second;
            return current_group;
        }
        current_group = (int)mesh.group_names.size();
        mesh.group_names.push_back(name);
        group_id_by_name[name] = current_group;
        return current_group;
    };

    std::string line;
    while(std::getline(in, line)){
        trim(line);
        if(line.empty() || line[0]=='#') continue;

        if(starts_with(line, "v ")){
            std::istringstream ss(line);
            char vch; float x,y,z;
            ss >> vch >> x >> y >> z;
            if(!ss.fail()) mesh.vertices.push_back(v3(x,y,z));
        }
        else if(starts_with(line, "o ") || starts_with(line, "g ")){
            const std::string new_name = resolve_group_name(line);
            if(!have_current_group_name || new_name != current_group_name){
                current_group_name = new_name;
                have_current_group_name = true;
                current_group = -1; // lazy-create/reuse on the next face
            }
        }
        else if(starts_with(line, "f ")){
            std::istringstream ss(line);
            char fch; ss >> fch;

            std::vector<int> face;
            std::string tok;
            while(ss >> tok){
                int vi=-1;
                if(!parse_obj_index_token(tok, (int)mesh.vertices.size(), vi)){
                    if(err) *err = "Invalid face index token: '" + tok + "'";
                    return false;
                }
                face.push_back(vi);
            }
            if(face.size() < 3) continue;

            if(current_group < 0){
                current_group = ensure_current_group();
            }

            // Triangulate fan (KEEP THIS)
            const int v0 = face[0];
            for(size_t i=1; i+1<face.size(); ++i){
                Tri t;
                t.v0 = v0;
                t.v1 = face[i];
                t.v2 = face[i+1];
                t.group_id = current_group;
                mesh.tris.push_back(t);
            }
        }
    }

    if(mesh.vertices.empty() || mesh.tris.empty()){
        if(err) *err = "OBJ contained no vertices or no faces.";
        return false;
    }

    vec3 bmin = mesh.vertices[0];
    vec3 bmax = mesh.vertices[0];
    for(const auto& v : mesh.vertices){
        bmin.x = std::min(bmin.x, v.x); bmin.y = std::min(bmin.y, v.y); bmin.z = std::min(bmin.z, v.z);
        bmax.x = std::max(bmax.x, v.x); bmax.y = std::max(bmax.y, v.y); bmax.z = std::max(bmax.z, v.z);
    }
    mesh.bmin = bmin;
    mesh.bmax = bmax;
    return true;
}

// ============================================================
// Geometry helpers (CUDA-safe)
// ============================================================

HD static float point_triangle_dist2(const vec3& p, const vec3& a, const vec3& b, const vec3& c){
    const vec3 ab = sub3(b,a);
    const vec3 ac = sub3(c,a);
    const vec3 ap = sub3(p,a);

    const float d1 = dot3(ab, ap);
    const float d2 = dot3(ac, ap);
    if(d1 <= 0.f && d2 <= 0.f) return norm2_3(ap);

    const vec3 bp = sub3(p,b);
    const float d3 = dot3(ab, bp);
    const float d4 = dot3(ac, bp);
    if(d3 >= 0.f && d4 <= d3) return norm2_3(bp);

    const float vc = d1*d4 - d3*d2;
    if(vc <= 0.f && d1 >= 0.f && d3 <= 0.f){
        const float v = d1 / (d1 - d3);
        const vec3 proj = add3(a, mul3(ab, v));
        return norm2_3(sub3(p, proj));
    }

    const vec3 cp = sub3(p,c);
    const float d5 = dot3(ab, cp);
    const float d6 = dot3(ac, cp);
    if(d6 >= 0.f && d5 <= d6) return norm2_3(cp);

    const float vb = d5*d2 - d1*d6;
    if(vb <= 0.f && d2 >= 0.f && d6 <= 0.f){
        const float w = d2 / (d2 - d6);
        const vec3 proj = add3(a, mul3(ac, w));
        return norm2_3(sub3(p, proj));
    }

    const float va = d3*d6 - d5*d4;
    if(va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f){
        const vec3 bc = sub3(c,b);
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const vec3 proj = add3(b, mul3(bc, w));
        return norm2_3(sub3(p, proj));
    }

    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    const vec3 proj = add3(a, add3(mul3(ab, v), mul3(ac, w)));
    return norm2_3(sub3(p, proj));
}

HD static bool ray_triangle_intersect(const vec3& ro, const vec3& rd,
                                      const vec3& a, const vec3& b, const vec3& c,
                                      float tmin, float& out_t)
{
    const float EPS = 1e-8f;
    const vec3 e1 = sub3(b,a);
    const vec3 e2 = sub3(c,a);
    const vec3 pvec = cross3(rd, e2);
    const float det = dot3(e1, pvec);
    if(det > -EPS && det < EPS) return false;
    const float invDet = 1.f / det;

    const vec3 tvec = sub3(ro, a);
    const float u = dot3(tvec, pvec) * invDet;
    if(u < 0.f || u > 1.f) return false;

    const vec3 qvec = cross3(tvec, e1);
    const float v = dot3(rd, qvec) * invDet;
    if(v < 0.f || u + v > 1.f) return false;

    const float t = dot3(e2, qvec) * invDet;
    if(t > tmin){
        out_t = t;
        return true;
    }
    return false;
}

// ============================================================
// Grid + CPU SDF
// ============================================================

struct Grid3D {
    int Nx=0, Ny=0, Nz=0;
    vec3 origin = v3(0,0,0); // min-corner style origin
    vec3 h = v3(1,1,1);
};

inline static vec3 voxel_center(const Grid3D& g, int i, int j, int k){
    return v3(g.origin.x + (i + 0.5f)*g.h.x,
              g.origin.y + (j + 0.5f)*g.h.y,
              g.origin.z + (k + 0.5f)*g.h.z);
}

// Narrow-band SDF: exact signed distance within `band_radius` of the surface,
// sign-correct clamp (+/- band_radius) elsewhere. Same result as compute_sdf_cpu
// inside the band, ~100x faster. band_radius<0 => auto (8h).
inline static void compute_sdf_cpu(const Mesh3D& mesh, const Grid3D& grid,
                                              std::vector<float>& sdf_out,
                                              float band_radius = -1.f)
{
    const int Nx=grid.Nx, Ny=grid.Ny, Nz=grid.Nz;
    const size_t N=(size_t)Nx*(size_t)Ny*(size_t)Nz;
    const float hx=grid.h.x, hy=grid.h.y, hz=grid.h.z;
    if(band_radius < 0.f) band_radius = 8.f*hx;   // >= tol + (WENO stencil radius +1)*h

    // 1) |distance|, initialized to the clamp value.
    std::vector<float> absd(N, band_radius);

    // 2) Scatter exact distance into the band from each triangle's expanded AABB.
    for(const auto& t : mesh.tris){
        const vec3 a=mesh.vertex_transformed(t.v0),
                   b=mesh.vertex_transformed(t.v1),
                   c=mesh.vertex_transformed(t.v2);
        const float lox=std::min(a.x,std::min(b.x,c.x))-band_radius,
                    loy=std::min(a.y,std::min(b.y,c.y))-band_radius,
                    loz=std::min(a.z,std::min(b.z,c.z))-band_radius,
                    hix=std::max(a.x,std::max(b.x,c.x))+band_radius,
                    hiy=std::max(a.y,std::max(b.y,c.y))+band_radius,
                    hiz=std::max(a.z,std::max(b.z,c.z))+band_radius;
        int i0=(int)std::floor((lox-grid.origin.x)/hx-0.5f), i1=(int)std::ceil((hix-grid.origin.x)/hx-0.5f);
        int j0=(int)std::floor((loy-grid.origin.y)/hy-0.5f), j1=(int)std::ceil((hiy-grid.origin.y)/hy-0.5f);
        int k0=(int)std::floor((loz-grid.origin.z)/hz-0.5f), k1=(int)std::ceil((hiz-grid.origin.z)/hz-0.5f);
        if(i0<0)i0=0; if(j0<0)j0=0; if(k0<0)k0=0;
        if(i1>Nx-1)i1=Nx-1; if(j1>Ny-1)j1=Ny-1; if(k1>Nz-1)k1=Nz-1;
        for(int k=k0;k<=k1;++k) for(int j=j0;j<=j1;++j) for(int i=i0;i<=i1;++i){
            const vec3 p=voxel_center(grid,i,j,k);
            const float d=std::sqrt(point_triangle_dist2(p,a,b,c));
            const size_t idx=(size_t)k*Ny*Nx+(size_t)j*Nx+(size_t)i;
            if(d<absd[idx]) absd[idx]=d;
        }
    }

    // 3) Sign via per-(j,k) scanline parity along +X (matches brute-force convention).
    sdf_out.assign(N,0.f);
    const float tmin=1e-6f; const vec3 rd=v3(1.f,0.f,0.f);
    std::vector<float> xs; xs.reserve(64);
    for(int k=0;k<Nz;++k) for(int j=0;j<Ny;++j){
        // sub-voxel irrational jitter: keep the +X parity ray OFF axis-aligned mesh edges (the
// wall<->floor/ceiling junction). ray_triangle_intersect is edge-INCLUSIVE, so a ray grazing a
// shared edge counts on BOTH triangles -> parity flips -> a wrong-sign band along the floor row.
// x-plane wall crossings are unmoved by a tiny y,z shift, so every cell's sign is unchanged
// except it no longer double-counts those junctions (host test: 64 sign errors -> 0).
const float y0=grid.origin.y+(j+0.5f)*hy + 0.0137f*hy, z0=grid.origin.z+(k+0.5f)*hz + 0.0091f*hz;
        const vec3 ro=v3(grid.origin.x,y0,z0);
        xs.clear();
        for(const auto& t:mesh.tris){
            const vec3 a=mesh.vertex_transformed(t.v0),
                       b=mesh.vertex_transformed(t.v1),
                       c=mesh.vertex_transformed(t.v2);
            float tt; if(ray_triangle_intersect(ro,rd,a,b,c,0.0f,tt)) xs.push_back(ro.x+tt);
        }
        std::sort(xs.begin(),xs.end());
        for(int i=0;i<Nx;++i){
            const float xc=grid.origin.x+(i+0.5f)*hx;
            const int ahead=(int)(xs.end()-std::upper_bound(xs.begin(),xs.end(),xc+tmin));
            const size_t idx=(size_t)k*Ny*Nx+(size_t)j*Nx+(size_t)i;
            float s=absd[idx]; if(!(ahead&1)) s=-s; sdf_out[idx]=s;
        }
    }
}

// ============================================================
// CUDA SDF path (file-scope, no lambdas, no nested kernels)
// ============================================================
#ifdef __CUDACC__

struct dMesh3D {
    const vec3* d_vertices = nullptr;
    const Tri*  d_tris     = nullptr;
    int nverts = 0;
    int ntris  = 0;
    vec3 shift = v3(0,0,0);
    float scale = 1.f;
};

__device__ __forceinline__ vec3 d_vertex_transformed(const dMesh3D& m, int i){
    vec3 v = m.d_vertices[i];
    v.x = (v.x + m.shift.x) * m.scale;
    v.y = (v.y + m.shift.y) * m.scale;
    v.z = (v.z + m.shift.z) * m.scale;
    return v;
}

__global__ void sdf3d_kernel_naive(float* __restrict__ sdf, Grid3D grid, dMesh3D mesh)
{
    const int Nx=grid.Nx, Ny=grid.Ny, Nz=grid.Nz;
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int j = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int k = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if(i>=Nx || j>=Ny || k>=Nz) return;

    vec3 p = v3(grid.origin.x + (i + 0.5f)*grid.h.x,
                grid.origin.y + (j + 0.5f)*grid.h.y,
                grid.origin.z + (k + 0.5f)*grid.h.z);

    float mind2 = FLT_MAX;
    int hits = 0;
    const vec3 rd = v3(1.f,0.f,0.f);
    const float tmin = 1e-6f;

    for(int t=0; t<mesh.ntris; ++t){
        const Tri tri = mesh.d_tris[t];
        const vec3 a = d_vertex_transformed(mesh, tri.v0);
        const vec3 b = d_vertex_transformed(mesh, tri.v1);
        const vec3 c = d_vertex_transformed(mesh, tri.v2);

        mind2 = fminf(mind2, point_triangle_dist2(p,a,b,c));

        float tt;
        if(ray_triangle_intersect(p, rd, a,b,c, tmin, tt)) hits++;
    }

    float sd = sqrtf(mind2);
    if(!(hits & 1)) sd = -sd;
	//else sd = -sd;
    sdf[(size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i] = sd;
}

inline static bool upload_mesh_to_device(const Mesh3D& hmesh, dMesh3D& dmesh,
                                        vec3*& d_vertices_out, Tri*& d_tris_out,
                                        std::string* err=nullptr)
{
    dmesh.nverts = (int)hmesh.vertices.size();
    dmesh.ntris  = (int)hmesh.tris.size();
    dmesh.shift  = hmesh.shift;
    dmesh.scale  = hmesh.scale;

    if(dmesh.nverts<=0 || dmesh.ntris<=0){
        if(err) *err = "Mesh empty.";
        return false;
    }

    cudaError_t ce;
    ce = cudaMalloc((void**)&d_vertices_out, sizeof(vec3)*(size_t)dmesh.nverts);
    if(ce!=cudaSuccess){ if(err) *err="cudaMalloc verts failed"; return false; }

    ce = cudaMalloc((void**)&d_tris_out, sizeof(Tri)*(size_t)dmesh.ntris);
    if(ce!=cudaSuccess){
        cudaFree(d_vertices_out); d_vertices_out=nullptr;
        if(err) *err="cudaMalloc tris failed";
        return false;
    }

    ce = cudaMemcpy(d_vertices_out, hmesh.vertices.data(),
                    sizeof(vec3)*(size_t)dmesh.nverts, cudaMemcpyHostToDevice);
    if(ce!=cudaSuccess){
        cudaFree(d_vertices_out); cudaFree(d_tris_out);
        d_vertices_out=nullptr; d_tris_out=nullptr;
        if(err) *err="cudaMemcpy verts failed";
        return false;
    }

    ce = cudaMemcpy(d_tris_out, hmesh.tris.data(),
                    sizeof(Tri)*(size_t)dmesh.ntris, cudaMemcpyHostToDevice);
    if(ce!=cudaSuccess){
        cudaFree(d_vertices_out); cudaFree(d_tris_out);
        d_vertices_out=nullptr; d_tris_out=nullptr;
        if(err) *err="cudaMemcpy tris failed";
        return false;
    }

    dmesh.d_vertices = d_vertices_out;
    dmesh.d_tris     = d_tris_out;
    return true;
}

inline static void free_mesh_device(vec3*& d_vertices, Tri*& d_tris){
    if(d_vertices){ cudaFree(d_vertices); d_vertices=nullptr; }
    if(d_tris){ cudaFree(d_tris); d_tris=nullptr; }
}

#ifndef SDF_ROW_XCAP
#define SDF_ROW_XCAP 64   // max +X surface crossings per row; overflow -> safe fallback
#endif

__global__ void k_fill_u32(unsigned int* a, size_t n, unsigned int val){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) a[i]=val;
}

// one thread per triangle: scatter |distance| into the band via atomicMin
__global__ void k_sdf_scatter(unsigned int* __restrict__ absu, Grid3D grid, dMesh3D mesh, float band){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t>=mesh.ntris) return;
    const Tri tri = mesh.d_tris[t];
    const vec3 a=d_vertex_transformed(mesh,tri.v0), b=d_vertex_transformed(mesh,tri.v1), c=d_vertex_transformed(mesh,tri.v2);
    const int Nx=grid.Nx,Ny=grid.Ny,Nz=grid.Nz;
    const float hx=grid.h.x,hy=grid.h.y,hz=grid.h.z;
    const float lox=fminf(a.x,fminf(b.x,c.x))-band, hix=fmaxf(a.x,fmaxf(b.x,c.x))+band;
    const float loy=fminf(a.y,fminf(b.y,c.y))-band, hiy=fmaxf(a.y,fmaxf(b.y,c.y))+band;
    const float loz=fminf(a.z,fminf(b.z,c.z))-band, hiz=fmaxf(a.z,fmaxf(b.z,c.z))+band;
    int i0=(int)floorf((lox-grid.origin.x)/hx-0.5f), i1=(int)ceilf((hix-grid.origin.x)/hx-0.5f);
    int j0=(int)floorf((loy-grid.origin.y)/hy-0.5f), j1=(int)ceilf((hiy-grid.origin.y)/hy-0.5f);
    int k0=(int)floorf((loz-grid.origin.z)/hz-0.5f), k1=(int)ceilf((hiz-grid.origin.z)/hz-0.5f);
    if(i0<0)i0=0; if(j0<0)j0=0; if(k0<0)k0=0;
    if(i1>Nx-1)i1=Nx-1; if(j1>Ny-1)j1=Ny-1; if(k1>Nz-1)k1=Nz-1;
    for(int k=k0;k<=k1;++k){ const float pz=grid.origin.z+(k+0.5f)*hz;
      for(int j=j0;j<=j1;++j){ const float py=grid.origin.y+(j+0.5f)*hy;
        for(int i=i0;i<=i1;++i){ const float px=grid.origin.x+(i+0.5f)*hx;
            const vec3 p=v3(px,py,pz);
            const float d=sqrtf(point_triangle_dist2(p,a,b,c));   // d >= 0
            const size_t idx=(size_t)k*Ny*Nx+(size_t)j*Nx+(size_t)i;
            atomicMin(&absu[idx], __float_as_uint(d));            // uint order == float order for d>=0
        }
      }
    }
}

// one thread per (j,k) row: parity sign along +X, then finalize to signed float S
__global__ void k_sdf_sign_finalize(const unsigned int* __restrict__ absu, float* __restrict__ S,
                                     Grid3D grid, dMesh3D mesh){
    const int Nx=grid.Nx,Ny=grid.Ny,Nz=grid.Nz;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    if(j>=Ny || k>=Nz) return;
    const float hx=grid.h.x,hy=grid.h.y,hz=grid.h.z, tmin=1e-6f;
    // sub-voxel irrational jitter: keep the +X parity ray OFF axis-aligned mesh edges (the
// wall<->floor/ceiling junction). ray_triangle_intersect is edge-INCLUSIVE, so a ray grazing a
// shared edge counts on BOTH triangles -> parity flips -> a wrong-sign band along the floor row.
// x-plane wall crossings are unmoved by a tiny y,z shift, so every cell's sign is unchanged
// except it no longer double-counts those junctions (host test: 64 sign errors -> 0).
const float y0=grid.origin.y+(j+0.5f)*hy + 0.0137f*hy, z0=grid.origin.z+(k+0.5f)*hz + 0.0091f*hz;
    const vec3 ro=v3(grid.origin.x,y0,z0), rd=v3(1.f,0.f,0.f);

    float xs[SDF_ROW_XCAP]; int nc=0; bool overflow=false;
    for(int t=0;t<mesh.ntris;++t){
        const Tri tri=mesh.d_tris[t];
        const vec3 a=d_vertex_transformed(mesh,tri.v0),b=d_vertex_transformed(mesh,tri.v1),c=d_vertex_transformed(mesh,tri.v2);
        float tt; if(ray_triangle_intersect(ro,rd,a,b,c,0.0f,tt)){ if(nc<SDF_ROW_XCAP) xs[nc++]=ro.x+tt; else {overflow=true; break;} }
    }
    if(!overflow){
        for(int x=1;x<nc;++x){ float key=xs[x]; int y=x-1; while(y>=0&&xs[y]>key){xs[y+1]=xs[y];--y;} xs[y+1]=key; } // insertion sort
        for(int i=0;i<Nx;++i){
            const float xc=grid.origin.x+(i+0.5f)*hx;
            int ahead=0; for(int q=nc-1;q>=0;--q){ if(xs[q]>xc+tmin) ++ahead; else break; }
            const size_t idx=(size_t)k*Ny*Nx+(size_t)j*Nx+(size_t)i;
            const float d=__uint_as_float(absu[idx]); S[idx]=(ahead&1)? d : -d;   // odd ahead => interior => +  (your convention)
        }
    } else { // robust per-voxel fallback (exactly your brute-force parity)
        for(int i=0;i<Nx;++i){
            const float xc=grid.origin.x+(i+0.5f)*hx; const vec3 p=v3(xc,y0,z0); int hits=0;
            for(int t=0;t<mesh.ntris;++t){ const Tri tri=mesh.d_tris[t];
                const vec3 a=d_vertex_transformed(mesh,tri.v0),b=d_vertex_transformed(mesh,tri.v1),c=d_vertex_transformed(mesh,tri.v2);
                float tt; if(ray_triangle_intersect(p,rd,a,b,c,tmin,tt)) ++hits; }
            const size_t idx=(size_t)k*Ny*Nx+(size_t)j*Nx+(size_t)i;
            const float d=__uint_as_float(absu[idx]); S[idx]=(hits&1)? d : -d;
        }
    }
}

// Host driver: full narrow-band SDF on the GPU. grid.origin/h/N must match your CPU setup.
inline static bool build_sdf_gpu_narrowband(const Mesh3D& mesh, const Grid3D& grid, float band, float* S_host){
    const size_t N=(size_t)grid.Nx*grid.Ny*grid.Nz;
    vec3* d_v=nullptr; Tri* d_t=nullptr; dMesh3D dm; std::string err;
    if(!upload_mesh_to_device(mesh,dm,d_v,d_t,&err)) return false;
    unsigned int* d_absu=nullptr; float* d_S=nullptr;
    if(cudaMalloc(&d_absu,N*sizeof(unsigned int))!=cudaSuccess){ free_mesh_device(d_v,d_t); return false; }
    if(cudaMalloc(&d_S,N*sizeof(float))!=cudaSuccess){ cudaFree(d_absu); free_mesh_device(d_v,d_t); return false; }

    unsigned int ub; std::memcpy(&ub,&band,sizeof(ub));   // host-side __float_as_uint
    { int TPB=256; unsigned B=(unsigned)((N+TPB-1)/TPB); k_fill_u32<<<B,TPB>>>(d_absu,N,ub); }
    { int TPB=128; int B=(dm.ntris+TPB-1)/TPB;            k_sdf_scatter<<<B,TPB>>>(d_absu,grid,dm,band); }
    { dim3 TP(16,16); dim3 B((grid.Ny+15)/16,(grid.Nz+15)/16); k_sdf_sign_finalize<<<B,TP>>>(d_absu,d_S,grid,dm); }

    bool ok = (cudaDeviceSynchronize()==cudaSuccess);
    if(ok) ok = (cudaMemcpy(S_host,d_S,N*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    cudaFree(d_absu); cudaFree(d_S); free_mesh_device(d_v,d_t);
    return ok;
}

#endif // __CUDACC__

// ============================================================
// Public wrappers / “2D-like” API
// ============================================================

static constexpr uint8_t OID_NONE_U8 = 0xFF;
using Scene3D = Mesh3D;

inline static bool parse_obj_3d(const std::string& path, Scene3D& Sc, float h){
    std::string err;
    if(!load_obj_mesh3d(path, Sc, &err)){
        std::fprintf(stderr, "[OBJ3D] %s\n", err.c_str());
        return false;
    }
    // Force scene min corner to (0,0,0) in transformed space
    //Sc.shift = v3(-Sc.bmin.x, -Sc.bmin.y, -Sc.bmin.z);
    // Force scene min corner to (0,0,0) in transformed space, padded
    const int ng = PARSE3D_GHOST;
	const float pad = ng * h;
	// 1) shift so transformed bbox min becomes 0
	// 2) then inset the whole scene by +3h
	Sc.shift = v3(-Sc.bmin.x + pad, -Sc.bmin.y + pad, -Sc.bmin.z + pad);
	
    return true;
}

inline static int parse_obj3d(const char* path, Scene3D* sc, float h){
    if(!path || !sc) return 0;
    return parse_obj_3d(std::string(path), *sc, h) ? 1 : 0;
}

inline static void mesh_bbox_transformed(const Mesh3D& mesh, vec3& out_min, vec3& out_max){
    if(mesh.vertices.empty()){
        out_min = v3(0,0,0); out_max = v3(0,0,0);
        return;
    }
    vec3 mn = mesh.vertex_transformed(0);
    vec3 mx = mn;
    for(size_t i=1;i<mesh.vertices.size();++i){
        const vec3 v = mesh.vertex_transformed((int)i);
        mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
        mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
    }
    out_min = mn; out_max = mx;
}

inline static int build_sdf_cpu(const Scene3D& Sc, int Nx,int Ny,int Nz, float h, float** S_out){
    if(Nx<=0||Ny<=0||Nz<=0||!S_out) return 0;
    const size_t N = (size_t)Nx*(size_t)Ny*(size_t)Nz;

    float* S = (float*)std::malloc(sizeof(float)*N);
    if(!S){ std::fprintf(stderr,"[SDF3D] malloc failed\n"); return 0; }

    Grid3D grid;
    grid.Nx=Nx; grid.Ny=Ny; grid.Nz=Nz;
    grid.h=v3(h,h,h);
    // IMPORTANT:
    // The virtual SDF domain is defined to start at the origin (0,0,0) and extend
    // to (Lx+6h, Ly+6h, Lz+6h). The mesh itself is shifted by +3h so its transformed
    // bbox is [3h, size+3h]. Therefore the SDF grid origin MUST be (0,0,0), not the
    // transformed mesh bbox min.
    grid.origin=v3(0.f,0.f,0.f);

    try {
        std::vector<float> tmp;
        compute_sdf_cpu(Sc, grid, tmp);
        std::memcpy(S, tmp.data(), sizeof(float)*N);
    } catch(const std::bad_alloc&){
        std::fprintf(stderr,"[SDF3D] out of memory building CPU SDF scratch\n");
        std::free(S);
        return 0;
    }
    *S_out = S;
    return 1;
}

// ============================================================
// Memory-budget guard (prevents OOM freeze when h is too small)
// ------------------------------------------------------------
// A too-small h does NOT mean "user error". It means the requested grid does
// not fit in memory. We compute the voxel count in double/int64 (overflow-safe),
// compare the projected peak footprint against actually-available RAM/VRAM, and
// refuse early with a clear, actionable message + a suggested minimum h — BEFORE
// any allocation that would thrash the machine.
//
// Peak-memory model, bytes per voxel — these are the *coexisting* peaks during
// construction, not the final array (which is just S = 4N):
//   host: S(4) + absd(4) + sdf_out(4), all live at once in compute_sdf_cpu = 12
//   dev : d_absu(4) + d_S(4), both live at once in the narrow-band SDF      = 8
// (S_gpu, oid, d_oid live only in later phases and never beat these peaks.)
// The 0.70/0.85 fractions are the safety margin on top of the true peak.
// ============================================================

static constexpr size_t PHASES_HOST_BYTES_PER_VOXEL = 12;
static constexpr size_t PHASES_DEV_BYTES_PER_VOXEL  = 8;
static constexpr double PHASES_HOST_BUDGET_FRAC     = 0.70;
static constexpr double PHASES_DEV_BUDGET_FRAC      = 0.85;
static constexpr double PHASES_GB                   = 1.073741824e9;

// Available *physical* memory in bytes (best effort; 0 == unknown).
inline static size_t phases_host_avail_bytes(){
#if defined(_WIN32)
    MEMORYSTATUSEX st; st.dwLength = sizeof(st);
    if(GlobalMemoryStatusEx(&st)) return (size_t)st.ullAvailPhys;
    return 0;
#elif defined(__linux__)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psz   = sysconf(_SC_PAGESIZE);
    if(pages > 0 && psz > 0) return (size_t)pages * (size_t)psz;
    return 0;
#else
    return 0;
#endif
}

// Smallest h whose grid fits the host budget for this domain.
// N ~ (Lx/h)(Ly/h)(Lz/h) for small h  =>  h ~ cbrt(vol / Nmax).
inline static double phases_suggest_min_h(double Lx,double Ly,double Lz,size_t host_avail){
    if(host_avail == 0) return 0.0;
    const double Nmax = (PHASES_HOST_BUDGET_FRAC * (double)host_avail)
                        / (double)PHASES_HOST_BYTES_PER_VOXEL;
    if(Nmax <= 1.0) return 0.0;
    const double vol = (Lx>0?Lx:1.0) * (Ly>0?Ly:1.0) * (Lz>0?Lz:1.0);
    return std::cbrt(vol / Nmax);
}

// Overflow-safe grid sizing + budget check. Returns false with a reason in `err`
// instead of letting a tiny h blow past INT_MAX or exhaust RAM/VRAM.
inline static bool grid_dims_checked(float Lxf,float Lyf,float Lzf,float h,
                                     int* Nx,int* Ny,int* Nz,
                                     std::string* err)
{
    if(!(h > 0.f)){ if(err) *err = "grid spacing h must be > 0 (got <=0 or NaN)"; return false; }

    const double Lx=(double)Lxf, Ly=(double)Lyf, Lz=(double)Lzf;
    const double nxd = std::ceil(Lx/h);
    const double nyd = std::ceil(Ly/h);
    const double nzd = std::ceil(Lz/h);

    if(nxd < 2.0 || nyd < 2.0 || nzd < 2.0){
        if(err) *err = "degenerate grid (domain too small for this h)";
        return false;
    }
    if(nxd > (double)INT_MAX || nyd > (double)INT_MAX || nzd > (double)INT_MAX){
        if(err) *err = "a grid dimension exceeds 2^31 (h far too small)";
        return false;
    }

    const double Nd = nxd * nyd * nzd;           // exact enough; each factor < 2^31

    // --- host RAM budget ---
    const size_t host_avail = phases_host_avail_bytes();
    const double host_need  = Nd * (double)PHASES_HOST_BYTES_PER_VOXEL;
    if(host_avail > 0 && host_need > PHASES_HOST_BUDGET_FRAC * (double)host_avail){
        char buf[360];
        const double h_sug = phases_suggest_min_h(Lx,Ly,Lz,host_avail);
        std::snprintf(buf,sizeof(buf),
            "grid does not fit host RAM: h=%.6g gives ~%.3g voxels needing ~%.2f GB, "
            "but only ~%.2f GB is free. Increase h (suggested h >= %.6g).",
            (double)h, Nd, host_need/PHASES_GB, (double)host_avail/PHASES_GB, h_sug);
        if(err) *err = buf;
        return false;
    }
    // If availability is unknown, still refuse absurd footprints (~64 GB model).
    if(host_avail == 0 && host_need > 6.8e10){
        char buf[256];
        std::snprintf(buf,sizeof(buf),
            "grid likely too large: h=%.6g gives ~%.3g voxels (~%.2f GB). Increase h.",
            (double)h, Nd, host_need/PHASES_GB);
        if(err) *err = buf;
        return false;
    }

#ifdef __CUDACC__
    // --- device VRAM budget (the solver ultimately needs S+oid on the GPU) ---
    size_t dfree=0, dtot=0;
    if(cudaMemGetInfo(&dfree,&dtot) == cudaSuccess && dfree > 0){
        const double dev_need = Nd * (double)PHASES_DEV_BYTES_PER_VOXEL;
        if(dev_need > PHASES_DEV_BUDGET_FRAC * (double)dfree){
            char buf[320];
            std::snprintf(buf,sizeof(buf),
                "grid does not fit GPU memory: ~%.3g voxels need ~%.2f GB device, "
                "but only ~%.2f GB is free. Increase h.",
                Nd, dev_need/PHASES_GB, (double)dfree/PHASES_GB);
            if(err) *err = buf;
            return false;
        }
    }
#endif

    *Nx = (int)nxd; *Ny = (int)nyd; *Nz = (int)nzd;
    return true;
}

// Returns 1 on success, 0 on failure (invalid h, overflow, OOM, or alloc fail).
// On failure: *Nx=*Ny=*Nz=0, *S=nullptr, and a reason is printed to stderr.
inline static int build_sdf(Scene3D* sc, float h,
                            int* Nx,int* Ny,int* Nz,
                            float* Lx,float* Ly,float* Lz,
                            float** S)
{
    if(!sc||!Nx||!Ny||!Nz||!Lx||!Ly||!Lz||!S) return 0;
    *Nx=*Ny=*Nz=0; *S=nullptr;

    const int ng = PARSE3D_GHOST;
	const float pad = ng * h;

	vec3 bmin, bmax;
	mesh_bbox_transformed(*sc, bmin, bmax);

	// Domain is [0, (object_max + 3h)]
	vec3 dom_min = v3(0,0,0);
	vec3 dom_max = v3(bmax.x + pad, bmax.y + pad, bmax.z + pad);
	
	*Lx = dom_max.x - dom_min.x;   // = size_x + 6h
	*Ly = dom_max.y - dom_min.y;
	*Lz = dom_max.z - dom_min.z;

    // Overflow-safe sizing + RAM/VRAM budget check BEFORE any allocation.
    std::string gerr;
    if(!grid_dims_checked(*Lx,*Ly,*Lz, h, Nx,Ny,Nz, &gerr)){
        std::fprintf(stderr, "[SDF3D] %s\n", gerr.c_str());
        *Nx=*Ny=*Nz=0;
        return 0;
    }

    // Always compute CPU first
    if(!build_sdf_cpu(*sc, *Nx,*Ny,*Nz, h, S)){
        std::fprintf(stderr, "[SDF3D] host allocation failed for %lld voxels\n",
                     (long long)*Nx*(long long)*Ny*(long long)*Nz);
        *Nx=*Ny=*Nz=0; *S=nullptr;
        return 0;
    }

#ifdef __CUDACC__
    Grid3D grid; grid.Nx=*Nx; grid.Ny=*Ny; grid.Nz=*Nz; grid.h=v3(h,h,h); grid.origin=dom_min;
    const float band = 8.f*h;          // keep >= tol + (WENO stencil radius + 1)*h ; tol=3h here
    float* S_gpu = (float*)std::malloc(sizeof(float)*(size_t)(*Nx)*(*Ny)*(*Nz));
    if(S_gpu && build_sdf_gpu_narrowband(*sc, grid, band, S_gpu)){
        if(*S) std::free(*S);
        *S = S_gpu;                     // GPU result wins
    } else {
        if(S_gpu) std::free(S_gpu);     // keep the CPU narrow-band result
    }
#endif

    // A scene that encloses nothing is not simulatable: every cell is exterior, the
    // immersed BC clamps the whole field to zero, and the run completes "successfully"
    // with an all-zero IR and blank animation frames. That silent failure costs far more
    // to diagnose than it costs to check, so refuse the run here instead.
    // Typical cause: an OBJ of loose plates/panels with no watertight shell, or a
    // "BoundingBox" group whose faces reference the wrong vertices.
    {
        const size_t Ncell = (size_t)(*Nx) * (size_t)(*Ny) * (size_t)(*Nz);
        size_t n_in = 0;
        for(size_t idx = 0; idx < Ncell; ++idx) if((*S)[idx] > 0.f) ++n_in;
        if(n_in == 0){
            std::fprintf(stderr,
                "[SDF3D] Scene encloses no volume: 0 of %lld cells are inside the mesh.\n"
                "        Nothing can propagate — the run would output an all-zero field.\n"
                "        The OBJ needs a WATERTIGHT shell (floor + walls + ceiling);\n"
                "        loose plates and open surfaces do not enclose a domain.\n",
                (long long)Ncell);
            std::free(*S); *S = nullptr; *Nx = *Ny = *Nz = 0;
            return 0;
        }
        if(n_in * 1000 < Ncell)
            std::fprintf(stderr,
                "[SDF3D] warning: only %lld of %lld cells (%.3f %%) are inside the mesh —\n"
                "        check that the shell is closed and correctly oriented.\n",
                (long long)n_in, (long long)Ncell,
                100.0 * (double)n_in / (double)Ncell);
    }
    return 1;
}

// ============================================================
// SDF regularization: retire geometry the grid cannot resolve (host, once)
// ============================================================
// The immersed BC assumes the SDF is LOCALLY ONE PLANE: a cell's normal points at one wall
// and its probe (s+2h away from the footpoint) reads interior solution. Three KDEBUG traces
// on the BRAS auditorium ceiling (4.5 ppw) were exactly the places that assumption fails:
//   * a fluid slot ~3.5 cells tall  -> both walls' 3h bands overlap; the two nearest cells
//     probe ONTO EACH OTHER and reflect each other's reflection forever (10x / ~35 steps)
//   * a solid plate 1 cell thick    -> S>0 on both sides within a cell; no interior to anchor
//   * a kink between surfaces        -> |grad S| = 0.55, the central-difference normal is fiction
// Patching those in the kernel means guessing per cell per substep. Deciding ONCE here is
// deterministic, inspectable (see the dump), and leaves the kernel's tuned probe alone.
//
// What it does (all axis-aligned, so concave CORNERS are NOT rounded -- a corner cell has
// solid on one side per axis only, and is left exactly as the triangle SDF placed it):
//   1. fluid runs bounded by solid on both ends and shorter than 2*tol (band overlap) -> solid
//   2. solid runs shorter than 3 cells bounded by fluid on both ends -> grow by 1 each side
//   3. repair S ONLY around converted cells: fluid keeps min(S_exact, dist to new solid) so
//      the exact triangle SDF -- and every flat wall's sub-cell precision -- is untouched
//      elsewhere; converted cells get -(dist to nearest fluid). Interface at the cell FACE
//      (0.5h), matching how the exact SDF places a wall between two cell centres.
// Diagonal slots (walls at 45 deg) are not caught by axis runs; the kernel's rigid fallback
// remains as the net for those. ponytail: local brute-force distance repair in a cube of
// radius R around each converted cell, not an exact EDT -- O(n_conv * (2R+1)^3), fine once.
inline static int regularize_sdf(float* S, int Nx, int Ny, int Nz, float h, float tol,
                                 const char* dump_path)
{
    if(!S || Nx < 8 || Ny < 8 || Nz < 8) return 0;
    const size_t N = (size_t)Nx * (size_t)Ny * (size_t)Nz;
    auto id = [&](int i, int j, int k) -> size_t {
        return (size_t)i + (size_t)Nx * ((size_t)j + (size_t)Ny * (size_t)k);
    };
    const int GAP   = (int)std::ceil(2.0f * tol / h);   // fluid run <  GAP cells  -> fill  (band overlap)
    const int PLATE = 3;                                 // solid run <  PLATE cells -> thicken to >= 3

    std::vector<uint8_t> solid(N), mark(N, 0);          // mark: 1 = fill (gap), 2 = thicken (plate)
    for(size_t q = 0; q < N; ++q) solid[q] = (S[q] <= 0.f) ? 1 : 0;

    // scan every axis-aligned line for short bounded runs
    auto scan_line = [&](int ax, int b, int c){
        const int L = (ax == 0) ? Nx : (ax == 1) ? Ny : Nz;
        auto at = [&](int t) -> size_t {
            return ax == 0 ? id(t, b, c) : ax == 1 ? id(b, t, c) : id(b, c, t);
        };
        int t = 0;
        while(t < L){
            const uint8_t v = solid[at(t)];
            int t1 = t;
            while(t1 < L && solid[at(t1)] == v) ++t1;
            const int len = t1 - t;
            const bool bounded = (t > 0) && (t1 < L);     // not touching the domain edge
            if(bounded){
                if(!v && len < GAP)                        // thin FLUID slab between two solids
                    for(int u = t; u < t1; ++u) mark[at(u)] = 1;
                if(v && len < PLATE){                      // thin SOLID plate between two fluids
                    mark[at(t - 1)]  = std::max<uint8_t>(mark[at(t - 1)],  2);
                    mark[at(t1)]     = std::max<uint8_t>(mark[at(t1)],     2);
                }
            }
            t = t1;
        }
    };
    for(int k = 0; k < Nz; ++k) for(int j = 0; j < Ny; ++j) scan_line(0, j, k);
    for(int k = 0; k < Nz; ++k) for(int i = 0; i < Nx; ++i) scan_line(1, i, k);
    for(int j = 0; j < Ny; ++j) for(int i = 0; i < Nx; ++i) scan_line(2, i, j);

    // collect conversions (fluid cells only -- a solid cell marked 2 is already solid)
    std::vector<size_t> conv;
    int n_gap = 0, n_plate = 0;
    for(size_t q = 0; q < N; ++q){
        if(mark[q] && !solid[q]){
            conv.push_back(q);
            (mark[q] == 1) ? ++n_gap : ++n_plate;
        }
    }
    if(conv.empty()){
        std::printf("[SDF3D] regularize: nothing below resolution (gap < %d cells, plate < %d cells)\n",
                    GAP, PLATE);
        return 0;
    }

    // apply: converted cells become solid
    std::vector<uint8_t> conv_mask(N, 0);
    for(size_t q : conv){ conv_mask[q] = 1; solid[q] = 1; }

    // repair S in a shell around the conversions. Interface sits at the cell face: a fluid
    // cell one centre away from new solid is 0.5h from the wall, not 1h.
    const int R = (int)std::ceil(tol / h) + 3;
    for(size_t q : conv){
        const int i0 = (int)(q % Nx), j0 = (int)((q / Nx) % Ny), k0 = (int)(q / ((size_t)Nx * Ny));
        float d_fluid = 1e30f;                              // for the converted cell's own S
        for(int dk = -R; dk <= R; ++dk){
            const int k = k0 + dk; if(k < 0 || k >= Nz) continue;
            for(int dj = -R; dj <= R; ++dj){
                const int j = j0 + dj; if(j < 0 || j >= Ny) continue;
                for(int di = -R; di <= R; ++di){
                    const int i = i0 + di; if(i < 0 || i >= Nx) continue;
                    const size_t r = id(i, j, k);
                    if(r == q) continue;
                    const float d = h * std::sqrt((float)(di*di + dj*dj + dk*dk)) - 0.5f * h;
                    if(!solid[r]){                          // fluid neighbour: new wall may be nearer
                        if(d < S[r]) S[r] = std::max(d, 0.25f * h);
                        if(d < d_fluid) d_fluid = d;
                    }
                }
            }
        }
        S[q] = -std::max(d_fluid, 0.5f * h);
    }

    std::printf("[SDF3D] regularize: %d cell(s) below resolution converted to solid "
                "(%d in fluid gaps < %d cells, %d thickening plates < %d cells); S repaired within %d cells.\n",
                (int)conv.size(), n_gap, GAP, n_plate, PLATE, R);
    if(dump_path){
        if(FILE* f = std::fopen(dump_path, "w")){
            std::fprintf(f, "# cells converted fluid->solid by regularize_sdf  (i j k reason)\n"
                            "# reason: gap = fluid slab < %d cells between solids; plate = grown to thicken a solid < %d cells\n",
                         GAP, PLATE);
            for(size_t q : conv)
                std::fprintf(f, "%d %d %d %s\n", (int)(q % Nx), (int)((q / Nx) % Ny),
                             (int)(q / ((size_t)Nx * Ny)), mark[q] == 1 ? "gap" : "plate");
            std::fclose(f);
        }
    }
    return (int)conv.size();
}

// ============================================================
// OID hole filling (host, once, after build_oid_dense)
// ============================================================
// A band cell (0 < S < tol) with no material is a HOLE: kernel_apply_BC_immersed_sdf returns
// on oid==0xFF before writing, the band is not overwritten there, and the static solid
// interior behind it can ring through the gap (a fixed spot at one frequency, growing).
// regularize_sdf creates exactly these: it repairs S so a cell sees a newly FILLED gap as
// its wall (S = 0.5h, one face away), but build_oid_dense assigns material by distance to
// the ORIGINAL triangles, which are farther than tol -> no material. Audit of the BRAS CR4
// run found 61 such cells, all at the ceiling (k=87..93), S = 0.37h..0.50h.
// Fix: give every material-less band cell the material of the nearest band cell that has
// one (expanding cube, radius 1..R). Deterministic, and it cannot invent a material.
inline static int fill_oid_holes(const float* S, uint8_t* oid, int Nx, int Ny, int Nz, float tol)
{
    if(!S || !oid) return 0;
    auto id = [&](int i, int j, int k) -> size_t {
        return (size_t)i + (size_t)Nx * ((size_t)j + (size_t)Ny * (size_t)k);
    };
    auto valid = [&](uint8_t o){ return o != 0 && o != 0xFF; };
    const int R = 4;
    int filled = 0, left = 0;
    // collect first, assign after: a hole must not be filled from another hole's fresh value
    std::vector<std::pair<size_t, uint8_t>> fixes;
    for(int k = 0; k < Nz; ++k) for(int j = 0; j < Ny; ++j) for(int i = 0; i < Nx; ++i){
        const size_t q = id(i, j, k);
        if(!(S[q] > 0.f && S[q] < tol) || valid(oid[q])) continue;
        uint8_t best = 0xFF; float bestd = 1e30f;
        for(int r = 1; r <= R && best == 0xFF; ++r){
            for(int dk = -r; dk <= r; ++dk) for(int dj = -r; dj <= r; ++dj) for(int di = -r; di <= r; ++di){
                if(std::max({std::abs(di), std::abs(dj), std::abs(dk)}) != r) continue;   // shell only
                const int a = i + di, b = j + dj, c = k + dk;
                if(a < 0 || b < 0 || c < 0 || a >= Nx || b >= Ny || c >= Nz) continue;
                const size_t t = id(a, b, c);
                if(!(S[t] > 0.f && S[t] < tol) || !valid(oid[t])) continue;
                const float d = (float)(di*di + dj*dj + dk*dk);
                if(d < bestd){ bestd = d; best = oid[t]; }
            }
        }
        if(best != 0xFF) fixes.emplace_back(q, best); else ++left;
    }
    for(auto& f : fixes){ oid[f.first] = f.second; ++filled; }
    if(filled || left)
        std::printf("[SDF3D] oid holes: %d band cell(s) had no material -> filled from nearest band cell; %d left (no material within %d cells)\n",
                    filled, left, R);
    return filled;
}

// ============================================================
// OID helpers (CPU)
// ============================================================

inline static void grad_S_central3D(const float* S, int Nx,int Ny,int Nz,
                                   int i,int j,int k, float h,
                                   float& gx,float& gy,float& gz)
{
    const size_t im = (size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)(i-1);
    const size_t ip = (size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)(i+1);
    const size_t jm = (size_t)k*Ny*Nx + (size_t)(j-1)*Nx + (size_t)i;
    const size_t jp = (size_t)k*Ny*Nx + (size_t)(j+1)*Nx + (size_t)i;
    const size_t km = (size_t)(k-1)*Ny*Nx + (size_t)j*Nx + (size_t)i;
    const size_t kp = (size_t)(k+1)*Ny*Nx + (size_t)j*Nx + (size_t)i;

    gx = (S[ip] - S[im]) / (2.0f*h);
    gy = (S[jp] - S[jm]) / (2.0f*h);
    gz = (S[kp] - S[km]) / (2.0f*h);
}

struct NearestTriResult {
    int tri_id=-1;
    int group_id=-1;
    float d2=FLT_MAX;
};

inline static NearestTriResult nearest_tri_at_point(const Mesh3D& mesh, const vec3& p){
    NearestTriResult best;
    for(size_t t=0;t<mesh.tris.size();++t){
        const Tri tri = mesh.tris[t];
        const vec3 a = mesh.vertex_transformed(tri.v0);
        const vec3 b = mesh.vertex_transformed(tri.v1);
        const vec3 c = mesh.vertex_transformed(tri.v2);
        const float d2 = point_triangle_dist2(p,a,b,c);
        if(d2 < best.d2){
            best.d2=d2;
            best.tri_id=(int)t;
            best.group_id=tri.group_id;
        }
    }
    return best;
}

inline static int build_oid_dense_cpu(const Scene3D& Sc, const float* S,
                                      int Nx,int Ny,int Nz, float h, float tol,
                                      uint8_t** OID_out)
{
    if(!S||Nx<=0||Ny<=0||Nz<=0||!OID_out) return 0;
    const size_t N = (size_t)Nx*(size_t)Ny*(size_t)Nz;

    uint8_t* oid = (uint8_t*)std::malloc(N);
    if(!oid){ std::fprintf(stderr,"[OID3D] malloc failed\n"); return 0; }
    std::memset(oid, OID_NONE_U8, N);

    int max_gid=-1;
    for(const auto& t: Sc.tris) max_gid = std::max(max_gid, t.group_id);
    if(max_gid < 0){ *OID_out = oid; return 1; }

    std::vector<int> gid2lid((size_t)(max_gid+1), -1);
    int lid_counter=0;
    for(int gid=0; gid<=max_gid; ++gid){
        bool appears=false;
        for(const auto& t: Sc.tris){ if(t.group_id==gid){ appears=true; break; } }
        if(!appears) continue;
        if(lid_counter <= 254) gid2lid[(size_t)gid] = lid_counter++ + 1;
    }

    // OID classification must use the same virtual grid origin as the SDF.
    // The virtual domain starts at (0,0,0) even though the mesh is shifted by +3h.
    const vec3 dom_min = v3(0.f,0.f,0.f);

    for(int k=1;k<Nz-1;++k){
        for(int j=1;j<Ny-1;++j){
            for(int i=1;i<Nx-1;++i){
                const size_t idx = (size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i;
                const float s = S[idx];
                const float d = std::fabs(s);
                if(!(s >= 0.0f && d <= tol)) continue;

                float gx,gy,gz;
                grad_S_central3D(S, Nx,Ny,Nz, i,j,k, h, gx,gy,gz);
                const float gn2 = gx*gx + gy*gy + gz*gz;
                if(gn2 <= 1e-24f) continue;

                const float inv_gn = 1.0f/std::sqrt(gn2);
                const vec3 n = v3(gx*inv_gn, gy*inv_gn, gz*inv_gn);

                const vec3 C = v3(dom_min.x + (i + 0.5f)*h,
                                  dom_min.y + (j + 0.5f)*h,
                                  dom_min.z + (k + 0.5f)*h);
                const vec3 W = v3(C.x - s*n.x, C.y - s*n.y, C.z - s*n.z);

                NearestTriResult nt = nearest_tri_at_point(Sc, W);
                if(nt.group_id < 0 || nt.group_id > max_gid) continue;

                const int lid = gid2lid[(size_t)nt.group_id];
                if(lid >= 0 && lid <= 254) oid[idx] = (uint8_t)lid;
            }
        }
    }

    *OID_out = oid;
    return 1;
}

// ============================================================
// OID helpers (GPU) — one thread per voxel, brute-force nearest triangle.
// Bit-for-bit mirrors build_oid_dense_cpu's classification (same gid->lid order,
// same SDF origin (0,0,0), same 1/sqrt normalization). Strict speed win since the
// CPU path is the identical O(Ntri) search, just serial.
// ============================================================
#ifdef __CUDACC__

__global__ void k_build_oid_dense(const float* __restrict__ S, uint8_t* __restrict__ oid,
                                  Grid3D grid, dMesh3D mesh,
                                  const int* __restrict__ gid2lid, int max_gid,
                                  float h, float tol)
{
    const int Nx=grid.Nx, Ny=grid.Ny, Nz=grid.Nz;
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z*blockDim.z + threadIdx.z;
    if(i<1 || j<1 || k<1 || i>=Nx-1 || j>=Ny-1 || k>=Nz-1) return;

    const size_t idx = (size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i;
    const float s = S[idx];
    const float d = fabsf(s);
    if(!(s >= 0.0f && d <= tol)) return;

    // central gradient (interior guaranteed by bounds above)
    const size_t im=(size_t)k*Ny*Nx+(size_t)j*Nx+(size_t)(i-1);
    const size_t ip=(size_t)k*Ny*Nx+(size_t)j*Nx+(size_t)(i+1);
    const size_t jm=(size_t)k*Ny*Nx+(size_t)(j-1)*Nx+(size_t)i;
    const size_t jp=(size_t)k*Ny*Nx+(size_t)(j+1)*Nx+(size_t)i;
    const size_t km=(size_t)(k-1)*Ny*Nx+(size_t)j*Nx+(size_t)i;
    const size_t kp=(size_t)(k+1)*Ny*Nx+(size_t)j*Nx+(size_t)i;
    const float gx=(S[ip]-S[im])/(2.0f*h);
    const float gy=(S[jp]-S[jm])/(2.0f*h);
    const float gz=(S[kp]-S[km])/(2.0f*h);
    const float gn2 = gx*gx+gy*gy+gz*gz;
    if(gn2 <= 1e-24f) return;

    const float inv_gn = 1.0f/sqrtf(gn2);   // match CPU exactly (not rsqrtf)
    const vec3 n = v3(gx*inv_gn, gy*inv_gn, gz*inv_gn);

    const vec3 C = v3(grid.origin.x + (i+0.5f)*h,
                      grid.origin.y + (j+0.5f)*h,
                      grid.origin.z + (k+0.5f)*h);
    const vec3 W = v3(C.x - s*n.x, C.y - s*n.y, C.z - s*n.z);

    float best_d2 = FLT_MAX; int best_gid = -1;
    for(int t=0; t<mesh.ntris; ++t){
        const Tri tri = mesh.d_tris[t];
        const vec3 a = d_vertex_transformed(mesh, tri.v0);
        const vec3 b = d_vertex_transformed(mesh, tri.v1);
        const vec3 c = d_vertex_transformed(mesh, tri.v2);
        const float dd = point_triangle_dist2(W,a,b,c);
        if(dd < best_d2){ best_d2 = dd; best_gid = tri.group_id; }
    }
    if(best_gid < 0 || best_gid > max_gid) return;
    const int lid = gid2lid[best_gid];
    if(lid >= 0 && lid <= 254) oid[idx] = (uint8_t)lid;
}

// Drop-in replacement for build_oid_dense_cpu: same signature, returns host oid.
// Returns 0 on any device error so the caller can fall back to the CPU path.
inline static int build_oid_dense_gpu(const Scene3D& Sc, const float* S,
                                      int Nx,int Ny,int Nz, float h, float tol,
                                      uint8_t** OID_out)
{
    if(!S||Nx<=0||Ny<=0||Nz<=0||!OID_out) return 0;
    const size_t N = (size_t)Nx*(size_t)Ny*(size_t)Nz;

    uint8_t* oid = (uint8_t*)std::malloc(N);
    if(!oid){ std::fprintf(stderr,"[OID3D-GPU] host malloc failed\n"); return 0; }

    // gid -> local id, identical ordering to the CPU builder
    int max_gid=-1;
    for(const auto& t: Sc.tris) max_gid = std::max(max_gid, t.group_id);
    if(max_gid < 0){ std::memset(oid, OID_NONE_U8, N); *OID_out = oid; return 1; }

    std::vector<int> gid2lid((size_t)(max_gid+1), -1);
    int lid_counter=0;
    for(int gid=0; gid<=max_gid; ++gid){
        bool appears=false;
        for(const auto& t: Sc.tris){ if(t.group_id==gid){ appears=true; break; } }
        if(!appears) continue;
        if(lid_counter <= 254) gid2lid[(size_t)gid] = lid_counter++ + 1;
    }

    vec3* d_v=nullptr; Tri* d_t=nullptr; dMesh3D dm; std::string merr;
    if(!upload_mesh_to_device(Sc, dm, d_v, d_t, &merr)){
        std::fprintf(stderr,"[OID3D-GPU] %s\n", merr.c_str());
        std::free(oid); return 0;
    }

    float* d_S=nullptr; uint8_t* d_oid=nullptr; int* d_map=nullptr;
    bool ok = true;
    if(cudaMalloc(&d_S,   N*sizeof(float))                  != cudaSuccess) ok=false;
    if(ok && cudaMalloc(&d_oid, N*sizeof(uint8_t))          != cudaSuccess) ok=false;
    if(ok && cudaMalloc(&d_map,(size_t)(max_gid+1)*sizeof(int)) != cudaSuccess) ok=false;
    if(ok) ok = (cudaMemcpy(d_S, S, N*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
    if(ok) ok = (cudaMemcpy(d_map, gid2lid.data(), (size_t)(max_gid+1)*sizeof(int),
                            cudaMemcpyHostToDevice)==cudaSuccess);
    if(ok) ok = (cudaMemset(d_oid, OID_NONE_U8, N)==cudaSuccess);

    if(ok){
        Grid3D grid; grid.Nx=Nx; grid.Ny=Ny; grid.Nz=Nz; grid.h=v3(h,h,h); grid.origin=v3(0.f,0.f,0.f);
        dim3 TP(8,8,8);
        dim3 B((Nx+7)/8,(Ny+7)/8,(Nz+7)/8);
        k_build_oid_dense<<<B,TP>>>(d_S, d_oid, grid, dm, d_map, max_gid, h, tol);
        ok = (cudaDeviceSynchronize()==cudaSuccess);
    }
    if(ok) ok = (cudaMemcpy(oid, d_oid, N*sizeof(uint8_t), cudaMemcpyDeviceToHost)==cudaSuccess);

    if(d_S)   cudaFree(d_S);
    if(d_oid) cudaFree(d_oid);
    if(d_map) cudaFree(d_map);
    free_mesh_device(d_v, d_t);

    if(!ok){
        std::fprintf(stderr,"[OID3D-GPU] device path failed\n");
        std::free(oid);
        return 0;
    }
    *OID_out = oid;
    return 1;
}

#endif // __CUDACC__

// Auto-dispatch: try the GPU builder when compiled with nvcc, fall back to CPU
// on any device error. Same signature/result as build_oid_dense_cpu.
inline static int build_oid_dense(const Scene3D& Sc, const float* S,
                                  int Nx,int Ny,int Nz, float h, float tol,
                                  uint8_t** OID_out)
{
#ifdef __CUDACC__
    if(build_oid_dense_gpu(Sc, S, Nx,Ny,Nz, h, tol, OID_out)) return 1;
    std::fprintf(stderr,"[OID3D] GPU path unavailable, using CPU...\n");
#endif
    return build_oid_dense_cpu(Sc, S, Nx,Ny,Nz, h, tol, OID_out);
}

inline static int scene_layer_count_3d(const Scene3D& sc){
    return (int)sc.group_names.size() + 1; // Boundary + groups
}

// compatibility with 2D case
inline static void scene_layer_names_3d(const Scene3D& sc, std::vector<std::string>& names){
    names.clear();
    names.push_back("BoundingBox");
    for(const auto& g : sc.group_names) names.push_back(g);
}
inline static void scene_free(Scene3D* sc){
    if(!sc) return;
    sc->vertices.clear();
    sc->tris.clear();
    sc->group_names.clear();
    sc->bmin = v3(0,0,0);
    sc->bmax = v3(0,0,0);
    sc->shift = v3(0,0,0);
    sc->scale = 1.f;
}

// bolique incidence reflection coefficient for a locally reacting surface
__host__ float R_oblique(float alpha, float cos_theta){
	float R = sqrtf(1.0 - alpha);
	if (R > 0.99999) return R; // near-rigid safeguard: xi -> inf
	float xi = (1.0 + R) / (1.0 - R);
	return (xi*cos_theta - 1.0) / (xi*cos_theta + 1.0);
}

#undef HD
