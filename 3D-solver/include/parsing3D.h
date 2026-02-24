#pragma once
/*
parsing3d.h
-----------
- Loads genuine 3D Wavefront OBJ meshes (v / f) into a triangle soup.
- Computes a signed distance field (SDF) on a 3D Cartesian grid.
- Provides CUDA kernels for SDF evaluation (naive O(Ntri) per voxel).

Assumptions / scope (by design, to keep this header self-contained):
- Supports: v, f, o, g, usemtl (names recorded but not required)
- Ignores: vt, vn, smoothing groups, materials library (mtllib)
- Faces with >3 vertices are triangulated with a fan.
- SDF sign is computed via parity ray casting along +X direction.
  (For watertight meshes. For open meshes, sign is not well-defined.)

If your meshes are large and/or grid is huge, the naive kernels will be slow.
In that case, add an acceleration structure (BVH / uniform grid) and traverse it
in the kernel. The interfaces below were chosen so you can swap the backend.

Vito: This is meant as a drop-in replacement for the old 2D "fake OBJ" parser,
but for real 3D geometry.
*/

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
// Basic math types
// ============================================================

struct vec3 { float x=0.f, y=0.f, z=0.f; };

inline static vec3 v3(float x,float y,float z){ return vec3{x,y,z}; }
inline static vec3 add3(const vec3&a,const vec3&b){ return v3(a.x+b.x,a.y+b.y,a.z+b.z); }
inline static vec3 sub3(const vec3&a,const vec3&b){ return v3(a.x-b.x,a.y-b.y,a.z-b.z); }
inline static vec3 mul3(const vec3&a,float s){ return v3(a.x*s,a.y*s,a.z*s); }
inline static float dot3(const vec3&a,const vec3&b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline static vec3 cross3(const vec3&a,const vec3&b){
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
inline static float norm2_3(const vec3&a){ return dot3(a,a); }
inline static float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }

// ============================================================
// Mesh representation
// ============================================================

struct Tri {
    int v0=-1, v1=-1, v2=-1;     // vertex indices (0-based)
    int group_id=-1;             // optional: which group/object this tri belongs to
};

struct Mesh3D {
    std::vector<vec3> vertices;
    std::vector<Tri>  tris;

    // Bounding box (computed on load)
    vec3 bmin = v3(0,0,0);
    vec3 bmax = v3(0,0,0);

    // If you want to shift/scale the mesh for your solver:
    vec3 shift = v3(0,0,0);      // applied to vertices (v + shift)
    float scale = 1.f;           // applied after shift (scale*(v+shift))

    inline vec3 vertex_transformed(int i) const {
        const vec3 v = vertices[(size_t)i];
        return mul3(add3(v, shift), scale);
    }
};

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

// Parse one OBJ index token: "v", "v/vt", "v//vn", "v/vt/vn".
// Returns 0-based vertex index in *out_vi*. vt/vn are ignored.
inline static bool parse_obj_index_token(const std::string& tok, int nverts, int& out_vi){
    // Split by '/'
    int vi = 0;
    {
        // take substring before first '/'
        size_t slash = tok.find('/');
        std::string a = (slash==std::string::npos) ? tok : tok.substr(0, slash);
        if(a.empty()) return false;
        vi = std::atoi(a.c_str());
    }

    // OBJ indices: 1..N, or negative relative to end (-1 is last).
    if(vi > 0) vi = vi - 1;
    else if(vi < 0) vi = nverts + vi; // vi is negative
    else return false;

    if(vi < 0 || vi >= nverts) return false;
    out_vi = vi;
    return true;
}

// Loads an OBJ file into Mesh3D as triangles (triangle soup).
// - triangulates polygons with a fan around the first vertex.
// - computes bounding box in transformed space (shift/scale applied after load if you set them).
inline static bool load_obj_mesh3d(const std::string& path, Mesh3D& mesh, std::string* err=nullptr){
    std::ifstream in(path);
    if(!in){
        if(err) *err = "Could not open OBJ: " + path;
        return false;
    }

    mesh.vertices.clear();
    mesh.tris.clear();
    int current_group = -1;
    int group_counter = -1;

    std::string line;
    while(std::getline(in, line)){
        trim(line);
        if(line.empty() || line[0]=='#') continue;

        if(starts_with(line, "v ")){
            std::istringstream ss(line);
            char vch; float x,y,z;
            ss >> vch >> x >> y >> z;
            if(!ss.fail()){
                mesh.vertices.push_back(v3(x,y,z));
            }
        }
        else if(starts_with(line, "o ") || starts_with(line, "g ")){
            // new object/group: track a numeric id if you need per-object materials later
            ++group_counter;
            current_group = group_counter;
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

            // Triangulate fan: (v0, vi, vi+1)
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
        // else: ignore (vt, vn, usemtl, mtllib, s, etc.)
    }

    if(mesh.vertices.empty() || mesh.tris.empty()){
        if(err) *err = "OBJ contained no vertices or no faces.";
        return false;
    }

    // Compute bbox in *raw* vertex space for now (caller can apply shift/scale separately)
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
// Geometry: point-triangle distance + ray-triangle intersection
// ============================================================

// Closest point on triangle (a,b,c) to point p, and squared distance.
inline static float point_triangle_dist2(const vec3& p, const vec3& a, const vec3& b, const vec3& c){
    // From "Real-Time Collision Detection" (Christer Ericson), robust and branchy-but-fine.
    const vec3 ab = sub3(b,a);
    const vec3 ac = sub3(c,a);
    const vec3 ap = sub3(p,a);

    const float d1 = dot3(ab, ap);
    const float d2 = dot3(ac, ap);
    if(d1 <= 0.f && d2 <= 0.f) return norm2_3(ap); // barycentric (1,0,0)

    const vec3 bp = sub3(p,b);
    const float d3 = dot3(ab, bp);
    const float d4 = dot3(ac, bp);
    if(d3 >= 0.f && d4 <= d3) return norm2_3(bp); // (0,1,0)

    const float vc = d1*d4 - d3*d2;
    if(vc <= 0.f && d1 >= 0.f && d3 <= 0.f){
        const float v = d1 / (d1 - d3);
        const vec3 proj = add3(a, mul3(ab, v));
        return norm2_3(sub3(p, proj)); // edge AB
    }

    const vec3 cp = sub3(p,c);
    const float d5 = dot3(ab, cp);
    const float d6 = dot3(ac, cp);
    if(d6 >= 0.f && d5 <= d6) return norm2_3(cp); // (0,0,1)

    const float vb = d5*d2 - d1*d6;
    if(vb <= 0.f && d2 >= 0.f && d6 <= 0.f){
        const float w = d2 / (d2 - d6);
        const vec3 proj = add3(a, mul3(ac, w));
        return norm2_3(sub3(p, proj)); // edge AC
    }

    const float va = d3*d6 - d5*d4;
    if(va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f){
        const vec3 bc = sub3(c,b);
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const vec3 proj = add3(b, mul3(bc, w));
        return norm2_3(sub3(p, proj)); // edge BC
    }

    // inside face
    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    const vec3 proj = add3(a, add3(mul3(ab, v), mul3(ac, w)));
    return norm2_3(sub3(p, proj));
}

// Möller–Trumbore ray-triangle intersection.
// Ray: origin ro, direction rd (need not be unit). Returns true if intersects at t>tmin.
// For parity sign: you typically use rd=(1,0,0) and tmin ~ 1e-6 to avoid self hits.
inline static bool ray_triangle_intersect(const vec3& ro, const vec3& rd,
                                         const vec3& a, const vec3& b, const vec3& c,
                                         float tmin, float& out_t){
    const float EPS = 1e-8f;
    const vec3 e1 = sub3(b,a);
    const vec3 e2 = sub3(c,a);
    const vec3 pvec = cross3(rd, e2);
    const float det = dot3(e1, pvec);
    if(std::fabs(det) < EPS) return false; // parallel
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
// SDF (CPU): naive reference implementation
// ============================================================

struct Grid3D {
    int Nx=0, Ny=0, Nz=0;
    vec3 origin = v3(0,0,0);     // world coord of cell (0,0,0) center? choose consistently
    vec3 h = v3(1,1,1);          // spacing (dx,dy,dz)
};

// World coordinate of voxel center (i,j,k).
inline static vec3 voxel_center(const Grid3D& g, int i, int j, int k){
    return v3(g.origin.x + (i + 0.5f)*g.h.x,
              g.origin.y + (j + 0.5f)*g.h.y,
              g.origin.z + (k + 0.5f)*g.h.z);
}

// Computes SDF into sdf_out (size Nx*Ny*Nz, row-major with i fastest).
// Convention: negative = inside mesh, positive = outside.
inline static void compute_sdf_cpu(const Mesh3D& mesh, const Grid3D& grid, std::vector<float>& sdf_out){
    const int Nx=grid.Nx, Ny=grid.Ny, Nz=grid.Nz;
    sdf_out.assign((size_t)Nx*Ny*Nz, 0.f);

    const vec3 rd = v3(1.f, 0.f, 0.f);
    const float tmin = 1e-6f;

    for(int k=0;k<Nz;++k){
        for(int j=0;j<Ny;++j){
            for(int i=0;i<Nx;++i){
                const vec3 p = voxel_center(grid,i,j,k);

                float mind2 = FLT_MAX;
                int hits = 0;

                for(const auto& t : mesh.tris){
                    const vec3 a = mesh.vertex_transformed(t.v0);
                    const vec3 b = mesh.vertex_transformed(t.v1);
                    const vec3 c = mesh.vertex_transformed(t.v2);

                    mind2 = std::min(mind2, point_triangle_dist2(p,a,b,c));

                    float tt;
                    if(ray_triangle_intersect(p, rd, a,b,c, tmin, tt)){
                        hits++;
                    }
                }

                float sd = std::sqrt(mind2);
                if(hits & 1) sd = -sd; // odd => inside
                sdf_out[(size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i] = sd;
            }
        }
    }
}

#ifdef __CUDACC__
// ============================================================
// CUDA helpers + kernels (naive, but massively parallel)
// ============================================================

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

__device__ __forceinline__ float d_point_triangle_dist2(const vec3& p, const vec3& a, const vec3& b, const vec3& c){
    // Same as CPU version (Ericson). Kept inline for performance.
    const vec3 ab = v3(b.x-a.x, b.y-a.y, b.z-a.z);
    const vec3 ac = v3(c.x-a.x, c.y-a.y, c.z-a.z);
    const vec3 ap = v3(p.x-a.x, p.y-a.y, p.z-a.z);

    const float d1 = ab.x*ap.x + ab.y*ap.y + ab.z*ap.z;
    const float d2 = ac.x*ap.x + ac.y*ap.y + ac.z*ap.z;
    if(d1 <= 0.f && d2 <= 0.f) return ap.x*ap.x + ap.y*ap.y + ap.z*ap.z;

    const vec3 bp = v3(p.x-b.x, p.y-b.y, p.z-b.z);
    const float d3 = ab.x*bp.x + ab.y*bp.y + ab.z*bp.z;
    const float d4 = ac.x*bp.x + ac.y*bp.y + ac.z*bp.z;
    if(d3 >= 0.f && d4 <= d3) return bp.x*bp.x + bp.y*bp.y + bp.z*bp.z;

    const float vc = d1*d4 - d3*d2;
    if(vc <= 0.f && d1 >= 0.f && d3 <= 0.f){
        const float v = d1 / (d1 - d3);
        const vec3 proj = v3(a.x + ab.x*v, a.y + ab.y*v, a.z + ab.z*v);
        const vec3 d = v3(p.x-proj.x, p.y-proj.y, p.z-proj.z);
        return d.x*d.x + d.y*d.y + d.z*d.z;
    }

    const vec3 cp = v3(p.x-c.x, p.y-c.y, p.z-c.z);
    const float d5 = ab.x*cp.x + ab.y*cp.y + ab.z*cp.z;
    const float d6 = ac.x*cp.x + ac.y*cp.y + ac.z*cp.z;
    if(d6 >= 0.f && d5 <= d6) return cp.x*cp.x + cp.y*cp.y + cp.z*cp.z;

    const float vb = d5*d2 - d1*d6;
    if(vb <= 0.f && d2 >= 0.f && d6 <= 0.f){
        const float w = d2 / (d2 - d6);
        const vec3 proj = v3(a.x + ac.x*w, a.y + ac.y*w, a.z + ac.z*w);
        const vec3 d = v3(p.x-proj.x, p.y-proj.y, p.z-proj.z);
        return d.x*d.x + d.y*d.y + d.z*d.z;
    }

    const float va = d3*d6 - d5*d4;
    if(va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f){
        const vec3 bc = v3(c.x-b.x, c.y-b.y, c.z-b.z);
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const vec3 proj = v3(b.x + bc.x*w, b.y + bc.y*w, b.z + bc.z*w);
        const vec3 d = v3(p.x-proj.x, p.y-proj.y, p.z-proj.z);
        return d.x*d.x + d.y*d.y + d.z*d.z;
    }

    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    const vec3 proj = v3(a.x + ab.x*v + ac.x*w,
                         a.y + ab.y*v + ac.y*w,
                         a.z + ab.z*v + ac.z*w);
    const vec3 d = v3(p.x-proj.x, p.y-proj.y, p.z-proj.z);
    return d.x*d.x + d.y*d.y + d.z*d.z;
}

__device__ __forceinline__ vec3 d_cross(const vec3&a,const vec3&b){
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ __forceinline__ float d_dot(const vec3&a,const vec3&b){ return a.x*b.x + a.y*b.y + a.z*b.z; }

__device__ __forceinline__ bool d_ray_tri(const vec3& ro, const vec3& rd,
                                          const vec3& a, const vec3& b, const vec3& c,
                                          float tmin){
    const float EPS = 1e-8f;
    const vec3 e1 = v3(b.x-a.x, b.y-a.y, b.z-a.z);
    const vec3 e2 = v3(c.x-a.x, c.y-a.y, c.z-a.z);
    const vec3 pvec = d_cross(rd, e2);
    const float det = d_dot(e1, pvec);
    if(fabsf(det) < EPS) return false;
    const float invDet = 1.f / det;
    const vec3 tvec = v3(ro.x-a.x, ro.y-a.y, ro.z-a.z);
    const float u = d_dot(tvec, pvec) * invDet;
    if(u < 0.f || u > 1.f) return false;
    const vec3 qvec = d_cross(tvec, e1);
    const float v = d_dot(rd, qvec) * invDet;
    if(v < 0.f || u + v > 1.f) return false;
    const float t = d_dot(e2, qvec) * invDet;
    return (t > tmin);
}

__global__ void sdf3d_kernel_naive(float* __restrict__ sdf,
                                  Grid3D grid,
                                  dMesh3D mesh)
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

        mind2 = fminf(mind2, d_point_triangle_dist2(p,a,b,c));
        hits += d_ray_tri(p, rd, a,b,c, tmin) ? 1 : 0;
    }

    float sd = sqrtf(mind2);
    if(hits & 1) sd = -sd;
    sdf[(size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i] = sd;
}


// Upload helpers (host code, but compiled with nvcc)
inline static bool upload_mesh_to_device(const Mesh3D& hmesh, dMesh3D& dmesh,
                                        vec3*& d_vertices_out, Tri*& d_tris_out,
                                        std::string* err=nullptr)
{
    dmesh.nverts = (int)hmesh.vertices.size();
    dmesh.ntris  = (int)hmesh.tris.size();
    dmesh.shift  = hmesh.shift;
    dmesh.scale  = hmesh.scale;

    if(dmesh.nverts<=0 || dmesh.ntris<=0){
        if(err) *err = "Mesh is empty.";
        return false;
    }

    cudaError_t ce;
    ce = cudaMalloc((void**)&d_vertices_out, sizeof(vec3)*(size_t)dmesh.nverts);
    if(ce!=cudaSuccess){ if(err) *err = "cudaMalloc vertices failed"; return false; }
    ce = cudaMalloc((void**)&d_tris_out, sizeof(Tri)*(size_t)dmesh.ntris);
    if(ce!=cudaSuccess){ cudaFree(d_vertices_out); d_vertices_out=nullptr; if(err) *err="cudaMalloc tris failed"; return false; }

    ce = cudaMemcpy(d_vertices_out, hmesh.vertices.data(), sizeof(vec3)*(size_t)dmesh.nverts, cudaMemcpyHostToDevice);
    if(ce!=cudaSuccess){ cudaFree(d_vertices_out); cudaFree(d_tris_out); d_vertices_out=nullptr; d_tris_out=nullptr; if(err) *err="cudaMemcpy vertices failed"; return false; }
    ce = cudaMemcpy(d_tris_out, hmesh.tris.data(), sizeof(Tri)*(size_t)dmesh.ntris, cudaMemcpyHostToDevice);
    if(ce!=cudaSuccess){ cudaFree(d_vertices_out); cudaFree(d_tris_out); d_vertices_out=nullptr; d_tris_out=nullptr; if(err) *err="cudaMemcpy tris failed"; return false; }

    dmesh.d_vertices = d_vertices_out;
    dmesh.d_tris     = d_tris_out;
    return true;
}

inline static void free_mesh_device(vec3*& d_vertices, Tri*& d_tris){
    if(d_vertices){ cudaFree(d_vertices); d_vertices=nullptr; }
    if(d_tris){ cudaFree(d_tris); d_tris=nullptr; }
}

// ============================================================
// 2D-API compatibility layer (PUBLIC WRAPPERS for 3D)
// Paste this block after compute_sdf_cpu(...) and before #ifdef __CUDACC__
// ============================================================

static constexpr uint8_t OID_NONE_U8 = 0xFF;

// In 2D you expose Scene2D. For 3D, Mesh3D is already the "scene".
using Scene3D = Mesh3D;

// ---------------------------
// parse_obj3d() wrappers
// ---------------------------

inline static bool parse_obj_3d(
    const std::string& path,
    Scene3D& Sc)
{
    std::string err;
    if(!load_obj_mesh3d(path, Sc, &err)){
        std::fprintf(stderr, "[OBJ3D] %s\n", err.c_str());
        return false;
    }
    return true;
}

// Public wrapper with the exact "parse_objXd" pattern used in 2D
inline static int parse_obj3d(const char* path, Scene3D* sc){
    if(!path || !sc) return 0;
    return parse_obj_3d(std::string(path), *sc) ? 1 : 0;
}

// ---------------------------
// bbox in transformed space
// ---------------------------

inline static void mesh_bbox_transformed(const Mesh3D& mesh, vec3& out_min, vec3& out_max){
    if(mesh.vertices.empty()){
        out_min = v3(0,0,0);
        out_max = v3(0,0,0);
        return;
    }
    vec3 mn = mesh.vertex_transformed(0);
    vec3 mx = mn;
    for(size_t i=1; i<mesh.vertices.size(); ++i){
        const vec3 v = mesh.vertex_transformed((int)i);
        mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
        mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
    }
    out_min = mn; out_max = mx;
}

// ---------------------------
// build_sdf_cpu() (malloc buffer, like 2D)
// ---------------------------

inline static int build_sdf_cpu(
    const Scene3D& Sc,
    int Nx, int Ny, int Nz,
    float h,
    float** S_out)
{
    if(Nx<=0 || Ny<=0 || Nz<=0 || !S_out) return 0;

    const size_t N = (size_t)Nx * (size_t)Ny * (size_t)Nz;
    float* S = (float*)std::malloc(sizeof(float)*N);
    if(!S){
        std::fprintf(stderr, "[SDF3D] malloc failed\n");
        return 0;
    }

    vec3 bmin, bmax;
    mesh_bbox_transformed(Sc, bmin, bmax);

    Grid3D grid;
    grid.Nx = Nx; grid.Ny = Ny; grid.Nz = Nz;
    grid.h  = v3(h,h,h);
    // IMPORTANT: compute_sdf_cpu() uses origin + (i+0.5)*h => origin is "min corner" style.
    grid.origin = bmin;

    std::vector<float> tmp;
    compute_sdf_cpu(Sc, grid, tmp);
    std::memcpy(S, tmp.data(), sizeof(float)*N);

    *S_out = S;
    return 1;
}

// ---------------------------
// build_sdf() (computes grid dims + extents, uses CUDA if available)
// Same spirit as 2D build_sdf(Scene2D*,h,&Nx,&Ny,&Lx,&Ly,&S)
// ---------------------------

inline static void build_sdf(
    Scene3D* sc,
    float h,
    int* Nx, int* Ny, int* Nz,
    float* Lx, float* Ly, float* Lz,
    float** S)
{
    if(!sc || !Nx || !Ny || !Nz || !Lx || !Ly || !Lz || !S) return;

    vec3 bmin, bmax;
    mesh_bbox_transformed(*sc, bmin, bmax);

    *Lx = bmax.x - bmin.x;
    *Ly = bmax.y - bmin.y;
    *Lz = bmax.z - bmin.z;

    // Keep same convention as 2D: include both ends => +1
    *Nx = (int)std::ceil((*Lx)/h) + 1;
    *Ny = (int)std::ceil((*Ly)/h) + 1;
    *Nz = (int)std::ceil((*Lz)/h) + 1;

    // Default: CPU
    // If compiled with NVCC, we override with CUDA below.
    (void)build_sdf_cpu(*sc, *Nx, *Ny, *Nz, h, S);

#ifdef __CUDACC__
    // CUDA path (naive kernel + upload helpers already exist in this header)
    {
        const size_t N = (size_t)(*Nx) * (size_t)(*Ny) * (size_t)(*Nz);

        float* d_sdf = nullptr;
        vec3*  d_verts = nullptr;
        Tri*   d_tris  = nullptr;
        dMesh3D dmesh;
        std::string err;

        cudaError_t ce = cudaMalloc((void**)&d_sdf, sizeof(float)*N);
        if(ce != cudaSuccess){
            // Keep CPU result already computed.
            return;
        }

        if(!upload_mesh_to_device(*sc, dmesh, d_verts, d_tris, &err)){
            cudaFree(d_sdf);
            return; // keep CPU result
        }

        Grid3D grid;
        grid.Nx = *Nx; grid.Ny = *Ny; grid.Nz = *Nz;
        grid.h = v3(h,h,h);
        grid.origin = bmin;

        // Launch
        dim3 TP(8,8,8);
        dim3 BP((grid.Nx + TP.x - 1)/TP.x,
                (grid.Ny + TP.y - 1)/TP.y,
                (grid.Nz + TP.z - 1)/TP.z);

        sdf3d_kernel_naive<<<BP,TP>>>(d_sdf, grid, dmesh);
        ce = cudaDeviceSynchronize();
        if(ce == cudaSuccess){
            // Copy back into host buffer (replace CPU result)
            if(*S) std::free(*S);
            *S = (float*)std::malloc(sizeof(float)*N);
            if(*S){
                ce = cudaMemcpy(*S, d_sdf, sizeof(float)*N, cudaMemcpyDeviceToHost);
                if(ce != cudaSuccess){
                    std::free(*S);
                    *S = nullptr;
                    // fall back to CPU again
                    (void)build_sdf_cpu(*sc, *Nx, *Ny, *Nz, h, S);
                }
            }else{
                // fall back to CPU again
                (void)build_sdf_cpu(*sc, *Nx, *Ny, *Nz, h, S);
            }
        }

        cudaFree(d_sdf);
        free_mesh_device(d_verts, d_tris);
    }
#endif
}

// ---------------------------
// SDF gradient (central differences) for OID banding (3D)
// ---------------------------

inline static void grad_S_central3D(
    const float* S,
    int Nx, int Ny, int Nz,
    int i, int j, int k,
    float h,
    float& gx, float& gy, float& gz)
{
    const size_t idx = (size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i;

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

// ---------------------------
// Nearest triangle at a point (brute-force, CPU)
// returns tri index and group_id
// ---------------------------

struct NearestTriResult {
    int tri_id = -1;
    int group_id = -1;
    float d2 = FLT_MAX;
};

inline static NearestTriResult nearest_tri_at_point(const Mesh3D& mesh, const vec3& p){
    NearestTriResult best;
    for(size_t t=0; t<mesh.tris.size(); ++t){
        const Tri tri = mesh.tris[t];
        const vec3 a = mesh.vertex_transformed(tri.v0);
        const vec3 b = mesh.vertex_transformed(tri.v1);
        const vec3 c = mesh.vertex_transformed(tri.v2);
        const float d2 = point_triangle_dist2(p,a,b,c);
        if(d2 < best.d2){
            best.d2 = d2;
            best.tri_id = (int)t;
            best.group_id = tri.group_id;
        }
    }
    return best;
}

// ---------------------------
// build_oid_dense_cpu() (3D analogue of the 2D function)
// - Sets OID only for voxels with s>=0 and |s|<=tol
// - OID is a compact id in [0..254] mapped from mesh tri.group_id
// ---------------------------

inline static int build_oid_dense_cpu(
    const Scene3D& Sc,
    const float*   S,
    int            Nx,
    int            Ny,
    int            Nz,
    float          h,
    float          tol,
    uint8_t**      OID_out)
{
    if(!S || Nx<=0 || Ny<=0 || Nz<=0 || !OID_out) return 0;

    const size_t N = (size_t)Nx * (size_t)Ny * (size_t)Nz;
    uint8_t* oid = (uint8_t*)std::malloc(N);
    if(!oid){
        std::fprintf(stderr,"[OID3D] malloc failed\n");
        return 0;
    }
    std::memset(oid, OID_NONE_U8, N);

    // Map group_id -> compact [0..254]
    // group_id in this loader is 0..(ngroups-1) already, but we keep mapping robust.
    int max_gid = -1;
    for(const auto& t : Sc.tris) max_gid = std::max(max_gid, t.group_id);
    std::vector<int> gid2lid((size_t)(max_gid+1), -1);
    int lid_counter = 0;
    for(int gid=0; gid<=max_gid; ++gid){
        // only assign if appears
        bool appears=false;
        for(const auto& t : Sc.tris){ if(t.group_id==gid){ appears=true; break; } }
        if(!appears) continue;
        if(lid_counter <= 254){
            gid2lid[(size_t)gid] = lid_counter++;
        }
    }

    vec3 bmin, bmax;
    mesh_bbox_transformed(Sc, bmin, bmax);

    for(int k=1; k<Nz-1; ++k){
        for(int j=1; j<Ny-1; ++j){
            for(int i=1; i<Nx-1; ++i){
                const size_t idx = (size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i;
                const float s = S[idx];
                const float d = std::fabs(s);
                if(!(s >= 0.0f && d <= tol)) continue;

                float gx, gy, gz;
                grad_S_central3D(S, Nx, Ny, Nz, i, j, k, h, gx, gy, gz);
                const float gn2 = gx*gx + gy*gy + gz*gz;
                if(gn2 <= 1e-24f) continue;

                const float inv_gn = 1.0f/std::sqrt(gn2);
                const vec3 n = v3(gx*inv_gn, gy*inv_gn, gz*inv_gn);

                // voxel center in world coords (consistent with build_sdf grid.origin convention)
                const vec3 C = v3(bmin.x + (i + 0.5f)*h,
                                  bmin.y + (j + 0.5f)*h,
                                  bmin.z + (k + 0.5f)*h);

                // project to surface point W = C - s*n
                const vec3 W = v3(C.x - s*n.x, C.y - s*n.y, C.z - s*n.z);

                NearestTriResult nt = nearest_tri_at_point(Sc, W);
                if(nt.tri_id < 0) continue;
                if(nt.group_id < 0 || nt.group_id > max_gid) continue;

                const int lid = gid2lid[(size_t)nt.group_id];
                if(lid >= 0 && lid <= 254) oid[idx] = (uint8_t)lid;
            }
        }
    }

    *OID_out = oid;
    return 1;
}

// ---------------------------
// scene_free() parity with 2D
// ---------------------------

inline static int scene_layer_count_3d(const Scene3D& sc){
    // +1 because you include boundary layer
    return (int)sc.group_names.size() + 1;
}

inline static void scene_layer_names_3d(
    const Scene3D& sc,
    std::vector<std::string>& names)
{
    names.clear();

    // Boundary first (like 2D)
    names.push_back("Boundary");

    for(const auto& g : sc.group_names)
        names.push_back(g);
}

inline static void scene_free(Scene3D* sc){
    if(!sc) return;
    sc->vertices.clear();
    sc->tris.clear();
    sc->bmin = v3(0,0,0);
    sc->bmax = v3(0,0,0);
    sc->shift = v3(0,0,0);
    sc->scale = 1.f;
}

#endif // __CUDACC__
