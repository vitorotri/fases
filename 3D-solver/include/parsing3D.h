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

    // Default group
    //mesh.group_names.push_back("default");
    //int current_group = 0;
    int current_group = -1;

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
    		std::istringstream ss(line);
		
    		std::string tag, t1, t2;
    		ss >> tag >> t1 >> t2;   // read up to 2 tokens after tag
		
    		std::string name;
    		if(!t2.empty()) {
        		name = t2;           // take 2nd token (ceiling/walls/floor)
    		} else if(!t1.empty()) {
        		name = t1;           // fallback: only one token provided
    		} else {
        		name = "group_" + std::to_string(mesh.group_names.size());
    		}
		
    		current_group = (int)mesh.group_names.size();
    		mesh.group_names.push_back(name);
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
    			// Faces without any preceding 'g'/'o' go into an auto group (rare but legal OBJ).
    			mesh.group_names.push_back("unnamed");
    			current_group = 0;
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

inline static void compute_sdf_cpu(const Mesh3D& mesh, const Grid3D& grid, std::vector<float>& sdf_out){
    const int Nx=grid.Nx, Ny=grid.Ny, Nz=grid.Nz;
    sdf_out.assign((size_t)Nx*Ny*Nz, 0.f);

    const vec3 rd = v3(1.f,0.f,0.f);
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
                    if(ray_triangle_intersect(p, rd, a,b,c, tmin, tt)) hits++;
                }

                float sd = std::sqrt(mind2);
                if(hits & 1) sd = -sd;
                sdf_out[(size_t)k*Ny*Nx + (size_t)j*Nx + (size_t)i] = sd;
            }
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
    if(hits & 1) sd = -sd;
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

#endif // __CUDACC__

// ============================================================
// Public wrappers / “2D-like” API
// ============================================================

static constexpr uint8_t OID_NONE_U8 = 0xFF;
using Scene3D = Mesh3D;

inline static bool parse_obj_3d(const std::string& path, Scene3D& Sc){
    std::string err;
    if(!load_obj_mesh3d(path, Sc, &err)){
        std::fprintf(stderr, "[OBJ3D] %s\n", err.c_str());
        return false;
    }
    return true;
}
inline static int parse_obj3d(const char* path, Scene3D* sc){
    if(!path || !sc) return 0;
    return parse_obj_3d(std::string(path), *sc) ? 1 : 0;
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

    vec3 bmin,bmax;
    mesh_bbox_transformed(Sc, bmin, bmax);

    Grid3D grid;
    grid.Nx=Nx; grid.Ny=Ny; grid.Nz=Nz;
    grid.h=v3(h,h,h);
    grid.origin=bmin;

    std::vector<float> tmp;
    compute_sdf_cpu(Sc, grid, tmp);
    std::memcpy(S, tmp.data(), sizeof(float)*N);
    *S_out = S;
    return 1;
}

inline static void build_sdf(Scene3D* sc, float h,
                             int* Nx,int* Ny,int* Nz,
                             float* Lx,float* Ly,float* Lz,
                             float** S)
{
    if(!sc||!Nx||!Ny||!Nz||!Lx||!Ly||!Lz||!S) return;

    vec3 bmin,bmax;
    mesh_bbox_transformed(*sc, bmin, bmax);

    *Lx = bmax.x - bmin.x;
    *Ly = bmax.y - bmin.y;
    *Lz = bmax.z - bmin.z;

    *Nx = (int)std::ceil((*Lx)/h) + 1;
    *Ny = (int)std::ceil((*Ly)/h) + 1;
    *Nz = (int)std::ceil((*Lz)/h) + 1;

    // Always compute CPU first
    (void)build_sdf_cpu(*sc, *Nx,*Ny,*Nz, h, S);

#ifdef __CUDACC__
    // CUDA override (if it succeeds)
    const size_t N = (size_t)(*Nx)*(size_t)(*Ny)*(size_t)(*Nz);

    float* d_sdf = nullptr;
    cudaError_t ce = cudaMalloc((void**)&d_sdf, sizeof(float)*N);
    if(ce != cudaSuccess) return; // keep CPU

    vec3* d_verts=nullptr;
    Tri*  d_tris=nullptr;
    dMesh3D dmesh;
    std::string err;

    if(!upload_mesh_to_device(*sc, dmesh, d_verts, d_tris, &err)){
        cudaFree(d_sdf);
        return; // keep CPU
    }

    Grid3D grid;
    grid.Nx=*Nx; grid.Ny=*Ny; grid.Nz=*Nz;
    grid.h=v3(h,h,h);
    grid.origin=bmin;

    dim3 TP(8,8,8);
    dim3 BP((grid.Nx + TP.x - 1)/TP.x,
            (grid.Ny + TP.y - 1)/TP.y,
            (grid.Nz + TP.z - 1)/TP.z);

    sdf3d_kernel_naive<<<BP,TP>>>(d_sdf, grid, dmesh);
    ce = cudaDeviceSynchronize();
    if(ce == cudaSuccess){
        if(*S) std::free(*S);
        *S = (float*)std::malloc(sizeof(float)*N);
        if(*S){
            ce = cudaMemcpy(*S, d_sdf, sizeof(float)*N, cudaMemcpyDeviceToHost);
            if(ce != cudaSuccess){
                std::free(*S);
                *S = nullptr;
                (void)build_sdf_cpu(*sc, *Nx,*Ny,*Nz, h, S);
            }
        }else{
            (void)build_sdf_cpu(*sc, *Nx,*Ny,*Nz, h, S);
        }
    }

    cudaFree(d_sdf);
    free_mesh_device(d_verts, d_tris);
#endif
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

    vec3 bmin,bmax;
    mesh_bbox_transformed(Sc, bmin, bmax);

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

                const vec3 C = v3(bmin.x + (i + 0.5f)*h,
                                  bmin.y + (j + 0.5f)*h,
                                  bmin.z + (k + 0.5f)*h);
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
// Names / layers + free
// ============================================================

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

#undef HD
