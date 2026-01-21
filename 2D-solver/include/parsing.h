/* parsing.h — modernized C++17 header-only OBJ→SDF (CPU+CUDA) with PLY-ready design
   - Switched to std::vector/std::string (RAII) everywhere.
   - Kept public API names and signatures: parse_obj2d(...), build_sdf(...).
   - GPU path preserved with CUDA; CPU fallback retained.
*/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

/* ----------------------------- geometry (C++ types) ----------------------------- */
struct vec2 { float x=0.f, y=0.f; };

struct Poly {
    std::string name;                 /* "Boundary" or obstacle name */
    std::vector<int>   face;          /* indices into Scene.vertices (0-based) in winding order */
    std::vector<float> lines;         /* flat triples (a,b,c) per edge: ax+by+c=0 with unit (a,b), outward */
};

struct Scene2D {
    std::vector<vec2>  vertices;      /* global vertices (shifted to origin after parse) */
    std::vector<Poly>  polys;
    float  Lx=0.f, Ly=0.f;            /* extents after shifting to origin */
    int    idx_boundary=-1;           /* which poly is the domain boundary (-1 if none) */
};

/* ----------------------------- small helpers ----------------------------- */
inline static vec2 sub2(const vec2&a,const vec2&b){ return vec2{a.x-b.x,a.y-b.y}; }
inline static float dot2(const vec2&a,const vec2&b){ return a.x*b.x + a.y*b.y; }
inline static float clampf(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }

inline static float dist_point_seg(const vec2& p, const vec2& A, const vec2& B){
    vec2 AB = sub2(B,A);
    vec2 AP = sub2(p,A);
    float denom = AB.x*AB.x + AB.y*AB.y;
    float t = denom>0.0f ? dot2(AP,AB)/denom : 0.0f;
    t = clampf(t,0.0f,1.0f);
    vec2 C{ A.x + t*AB.x, A.y + t*AB.y };
    vec2 d = sub2(p,C);
    return std::sqrt(d.x*d.x + d.y*d.y);
}

inline static int point_in_poly(const vec2& p, const Scene2D& S, const Poly& P){
    int wn = 0, m = static_cast<int>(P.face.size());
    for(int e=0;e<m;e++){
        vec2 A = S.vertices[P.face[e]];
        vec2 B = S.vertices[P.face[(e+1==m)?0:e+1]];
        bool cond = ((A.y<=p.y)&&(B.y>p.y)) || ((A.y>p.y)&&(B.y<=p.y));
        if(cond){
            float xint = A.x + (p.y - A.y)*(B.x - A.x)/(B.y - A.y);
            wn ^= (xint > p.x);
        }
    }
    return wn;
}

inline static float dist_to_poly_edges(const vec2& p, const Scene2D& S, const Poly& P){
    float dmin = FLT_MAX; int m = static_cast<int>(P.face.size());
    for(int e=0;e<m;e++){
        vec2 A = S.vertices[P.face[e]];
        vec2 B = S.vertices[P.face[(e+1==m)?0:e+1]];
        float d = dist_point_seg(p,A,B);
        dmin = std::min(dmin,d);
    }
    return dmin;
}

/* ----------------------------- scene utilities ----------------------------- */
inline static Poly& scene_add_poly(Scene2D& S, std::string name){
    S.polys.emplace_back();
    S.polys.back().name = std::move(name);
    if(S.polys.back().name=="Boundary") S.idx_boundary = static_cast<int>(S.polys.size())-1;
    return S.polys.back();
}

/* ----------------------------- parsing (OBJ, C++17) ----------------------------- */
inline static int parse_obj2d(const char *path, Scene2D *Sout){
    if(!Sout) return 0;
    Scene2D S;
    std::ifstream in(path);
    if(!in){ std::fprintf(stderr,"[obj2d] cannot open '%s'\n", path); return 0; }

    Poly* cur=nullptr;
    std::string line;
    while(std::getline(in,line)){
        if(line.empty() || line[0]=='#') continue;
        std::istringstream iss(line);
        std::string tag; iss >> tag;
        if(tag=="o"){
            std::string name; iss >> name; if(name.empty()) name="unnamed";
            cur = &scene_add_poly(S, name);
        }else if(tag=="v"){
            double x=0,y=0; iss >> x >> y; if(iss){ S.vertices.push_back(vec2{(float)x,(float)y}); }
        }else if(tag=="f"){
            if(!cur) cur = &scene_add_poly(S,"unnamed");
            int idx; while(iss >> idx){ S.polys.back().face.push_back(idx-1); }
        }else if(tag=="vn"){ // compatibility: store (a,b,c)
            if(!cur) cur = &scene_add_poly(S,"unnamed");
            double a=0,b=0,c=0; iss >> a >> b >> c;
            double n = std::sqrt(a*a + b*b);
            if(n>0.0){ a/=n; b/=n; }
            cur->lines.push_back((float)a);
            cur->lines.push_back((float)b);
            cur->lines.push_back((float)c);
        }
    }
    if(S.vertices.empty() || S.polys.empty()){ std::fprintf(stderr,"[obj2d] empty scene\n"); return 0; }

    // shift to origin and update extents
    float xmin=S.vertices[0].x, xmax=xmin, ymin=S.vertices[0].y, ymax=ymin;
    for(size_t i=1;i<S.vertices.size();++i){
        auto v=S.vertices[i];
        xmin=std::min(xmin,v.x); xmax=std::max(xmax,v.x);
        ymin=std::min(ymin,v.y); ymax=std::max(ymax,v.y);
    }
    float dx=xmin, dy=ymin;
    for(auto& v: S.vertices){ v.x-=dx; v.y-=dy; }
    for(auto& P: S.polys){
        for(size_t t=0;t+2<P.lines.size(); t+=3){
            float a=P.lines[t+0], b=P.lines[t+1], c=P.lines[t+2];
            P.lines[t+2] = c + a*dx + b*dy;
        }
    }
    S.Lx = xmax-xmin; S.Ly = ymax-ymin;
    if(S.idx_boundary<0){
        std::fprintf(stderr,"[obj2d] Warning: no 'Boundary' object found. Using overall bbox as domain.\n");
    }
    *Sout = std::move(S);
    return 1;
}

/* -------------------- CPU fallback build (unchanged semantics) -------------------- */
inline static int build_sdf_cpu(const Scene2D *Sc, float h,
                                int *Nx, int *Ny, float *Lx, float *Ly, float **S_out)
{
    const float lx = Sc->Lx, ly = Sc->Ly;
    int nx = (int)std::floor(lx / h) + 1;
    int ny = (int)std::floor(ly / h) + 1;
    if(nx < 2) nx = 2; if(ny < 2) ny = 2;
    std::vector<float> S((size_t)nx*(size_t)ny);

    const Poly *B = (Sc->idx_boundary >= 0) ? &Sc->polys[Sc->idx_boundary] : NULL;
    for(int j=0;j<ny;j++){
        for(int i=0;i<nx;i++){
            vec2 p{ i * h, j * h };
            float dmin = FLT_MAX;
            int insideBoundary = 1;
            if(B){
                insideBoundary = point_in_poly(p,*Sc,*B);
                float db = dist_to_poly_edges(p,*Sc,*B);
                dmin = std::min(dmin, db);
            }
            int insideAnyObstacle = 0;
            for(size_t k=0;k<Sc->polys.size();++k){
                if((int)k==Sc->idx_boundary) continue;
                const Poly &P = Sc->polys[k];
                float dp = dist_to_poly_edges(p,*Sc,P);
                dmin = std::min(dmin, dp);
                insideAnyObstacle |= point_in_poly(p,*Sc,P);
            }
            float s = (insideBoundary==0) ? dmin : (insideAnyObstacle ? -dmin : dmin);
            S[(size_t)j*nx + i] = s;
        }
    }
    float *out = (float*)std::malloc(S.size()*sizeof(float));
    if(!out){ std::fprintf(stderr,"[SDF CPU] malloc failed\n"); return 0; }
    std::memcpy(out, S.data(), S.size()*sizeof(float));
    *Nx = nx; *Ny = ny; *Lx = lx; *Ly = ly; *S_out = out;
    return 1;
}

/* ----------------------------- GPU path (CUDA, RAII) ----------------------------- */
#ifdef __CUDACC__
template <class T>
struct DeviceBuffer {
    T* ptr=nullptr; size_t n=0;
    DeviceBuffer()=default;
    explicit DeviceBuffer(size_t n_): n(n_){ if(n){ auto st=cudaMalloc((void**)&ptr, n*sizeof(T)); if(st!=cudaSuccess) ptr=nullptr; } }
    ~DeviceBuffer(){ if(ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), n(o.n){ o.ptr=nullptr; o.n=0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept { if(this!=&o){ if(ptr) cudaFree(ptr); ptr=o.ptr; n=o.n; o.ptr=nullptr; o.n=0;} return *this; }
};

__device__ float clampf_d(float x, float lo, float hi){ return x<lo?lo:(x>hi?hi:x); }
__device__ float dist_point_seg_d(float px,float py,float ax,float ay,float bx,float by){
    float ABx=bx-ax, ABy=by-ay, APx=px-ax, APy=py-ay;
    float denom = ABx*ABx + ABy*ABy;
    float t = denom>0.0f ? (APx*ABx + APy*ABy)/denom : 0.0f;
    t = clampf_d(t,0.0f,1.0f);
    float cx=ax+t*ABx, cy=ay+t*ABy, dx=px-cx, dy=py-cy;
    return sqrtf(dx*dx+dy*dy);
}
__device__ int point_in_poly_d(float px,float py,
                               const float*Vx,const float*Vy,
                               const int*idx,int m){
    int wn=0;
    for(int e=0;e<m;e++){
        int i0=idx[e], i1=(e+1==m)?idx[0]:idx[e+1];
        float ax=Vx[i0], ay=Vy[i0], bx=Vx[i1], by=Vy[i1];
        int cond = ((ay<=py)&&(by>py)) || ((ay>py)&&(by<=py));
        if(cond){
            float xint = ax + (py - ay)*(bx - ax)/(by - ay);
            wn ^= (xint > px);
        }
    }
    return wn;
}
__device__ float dist_to_poly_edges_d(float px,float py,
                                      const float*Vx,const float*Vy,
                                      const int*idx,int m){
    float dmin = FLT_MAX;
    for(int e=0;e<m;e++){
        int i0=idx[e], i1=(e+1==m)?idx[0]:idx[e+1];
        float d = dist_point_seg_d(px,py,Vx[i0],Vy[i0],Vx[i1],Vy[i1]);
        dmin = fminf(dmin,d);
    }
    return dmin;
}

__global__ void sdf_kernel(float *S, int nx, int ny, float h,
                           const float *Vx, const float *Vy, int nV,
                           const int *polyOfs, const int *polyIdx, int nPoly,
                           const int idxB){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i>=nx || j>=ny) return;
    float px = i * h, py = j * h;
    float dmin = FLT_MAX;
    int insideBoundary = 1;
    if(idxB>=0){
        int ofs = polyOfs[idxB];
        int cnt = polyOfs[idxB+1]-ofs;
        insideBoundary = point_in_poly_d(px,py,Vx,Vy,&polyIdx[ofs],cnt);
        float db = dist_to_poly_edges_d(px,py,Vx,Vy,&polyIdx[ofs],cnt);
        dmin = fminf(dmin, db);
    }
    int insideAnyObstacle = 0;
    for(int k=0;k<nPoly;k++){
        if(k==idxB) continue;
        int ofs = polyOfs[k];
        int cnt = polyOfs[k+1]-ofs;
        float dp = dist_to_poly_edges_d(px,py,Vx,Vy,&polyIdx[ofs],cnt);
        dmin = fminf(dmin, dp);
        insideAnyObstacle |= point_in_poly_d(px,py,Vx,Vy,&polyIdx[ofs],cnt);
    }
    float s = (insideBoundary==0) ? dmin : (insideAnyObstacle ? -dmin : dmin);
    S[j*nx + i] = s;
}
#endif // __CUDACC__

/* ----------------------------- flatten to CSR (host) ----------------------------- */
struct CSRHost {
    std::vector<float> Vx, Vy;   // nV
    std::vector<int>   Ofs;      // nPoly+1
    std::vector<int>   Idx;      // nIdx
    int nV=0, nPoly=0, nIdx=0, idxB=-1;
    float Lx=0.f, Ly=0.f;
};

inline static CSRHost flatten_scene_host(const Scene2D& Sc){
    CSRHost H;
    H.nV = (int)Sc.vertices.size();
    H.nPoly = (int)Sc.polys.size();
    H.idxB = Sc.idx_boundary;
    H.Lx = Sc.Lx; H.Ly = Sc.Ly;
    H.Vx.resize(H.nV); H.Vy.resize(H.nV);
    for(int i=0;i<H.nV;i++){ H.Vx[i]=Sc.vertices[i].x; H.Vy[i]=Sc.vertices[i].y; }
    H.Ofs.resize(H.nPoly+1);
    int total=0; for(const auto& P: Sc.polys) total += (int)P.face.size();
    H.Idx.resize(total);
    int cur=0; H.Ofs[0]=0;
    for(int p=0;p<H.nPoly;p++){
        const auto& P = Sc.polys[p];
        for(int v: P.face) H.Idx[cur++]=v;
        H.Ofs[p+1]=cur;
    }
    H.nIdx = cur;
    return H;
}

/* ----------------------------- public entry ----------------------------- */
inline static int build_sdf(const Scene2D *Sc, float h,
                            int *Nx, int *Ny, float *Lx, float *Ly, float **S_out){
#ifndef __CUDACC__
    return build_sdf_cpu(Sc,h,Nx,Ny,Lx,Ly,S_out);
#else
    const float lx = Sc->Lx, ly = Sc->Ly;
    int nx = (int)std::floor(lx / h) + 1;
    int ny = (int)std::floor(ly / h) + 1;
    if(nx < 2) nx = 2; if(ny < 2) ny = 2;
    size_t nCells = (size_t)nx * (size_t)ny;
    CSRHost H = flatten_scene_host(*Sc);

    DeviceBuffer<float> dS(nCells);
    if(!dS.ptr){ // GPU not available → CPU fallback
        return build_sdf_cpu(Sc,h,Nx,Ny,Lx,Ly,S_out);
    }
    DeviceBuffer<float> dVx(H.nV), dVy(H.nV);
    DeviceBuffer<int>   dOfs(H.nPoly+1), dIdx(H.nIdx);

    cudaMemcpy(dVx.ptr, H.Vx.data(), sizeof(float)*H.nV, cudaMemcpyHostToDevice);
    cudaMemcpy(dVy.ptr, H.Vy.data(), sizeof(float)*H.nV, cudaMemcpyHostToDevice);
    cudaMemcpy(dOfs.ptr, H.Ofs.data(), sizeof(int)*(H.nPoly+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dIdx.ptr, H.Idx.data(), sizeof(int)*H.nIdx, cudaMemcpyHostToDevice);

    dim3 TP(8,8);
    dim3 BP( (nx + TP.x - 1)/TP.x, (ny + TP.y - 1)/TP.y );
    sdf_kernel<<<BP,TP>>>(dS.ptr, nx, ny, h, dVx.ptr, dVy.ptr, H.nV, dOfs.ptr, dIdx.ptr, H.nPoly, H.idxB);
    auto st = cudaGetLastError();
    if(st != cudaSuccess){
        return build_sdf_cpu(Sc,h,Nx,Ny,Lx,Ly,S_out);
    }
    cudaDeviceSynchronize();

    std::vector<float> hostS(nCells);
    cudaMemcpy(hostS.data(), dS.ptr, nCells*sizeof(float), cudaMemcpyDeviceToHost);
    float *out = (float*)std::malloc(nCells*sizeof(float));
    if(!out){ std::fprintf(stderr,"[SDF] host malloc failed\n"); return 0; }
    std::memcpy(out, hostS.data(), nCells*sizeof(float));

    *Nx = nx; *Ny = ny; *Lx = lx; *Ly = ly; *S_out = out;
    return 1;
#endif
}

inline void scene_free(Scene2D* S) {
    if (!S) return;
    *S = Scene2D{}; // RAII clears vectors/strings
}

/* =================== SERIAL TAGGING: cells & object IDs for |S|<=tol =================== */

/* Central-diff gradient of grid S at (i,j) with clamped neighbors */
inline static void grad_S_central(const float* S, int Nx, int Ny, int i, int j, float h,
                                  float& gx, float& gy)
{
    const int im1 = (i>0)     ? i-1 : i;
    const int ip1 = (i<Nx-1)  ? i+1 : i;
    const int jm1 = (j>0)     ? j-1 : j;
    const int jp1 = (j<Ny-1)  ? j+1 : j;
    gx = (S[j*Nx + ip1] - S[j*Nx + im1]) * (0.5f / h);
    gy = (S[jp1*Nx + i] - S[jm1*Nx + i]) * (0.5f / h);
}

struct NearestPolyResult {
    int   poly_id;   // object ID (index in Scene2D::polys)
    float dist;      // min distance from query point to that poly's boundary
};

/* Compute min distance from W to polygon boundary; fast accept by containment.
   Skips Sc.idx_boundary if set (to avoid attributing to the outer frame). */
inline static NearestPolyResult nearest_poly_at_point(const Scene2D& Sc, const vec2& W)
{
    NearestPolyResult out{ -1, FLT_MAX };

    // Prefer strict containment (interior test)
    for(int pid = 0; pid < (int)Sc.polys.size(); ++pid){
        if (pid == Sc.idx_boundary) continue;
        const Poly& P = Sc.polys[pid];
        if(!P.face.empty() && point_in_poly(W, Sc, P)){
            return NearestPolyResult{ pid, 0.0f };
        }
    }

    // Otherwise choose nearest boundary
    for(int pid = 0; pid < (int)Sc.polys.size(); ++pid){
        if (pid == Sc.idx_boundary) continue;
        const Poly& P = Sc.polys[pid];
        const int m = (int)P.face.size();
        if(m < 2) continue;
        for(int e=0; e<m; ++e){
            const vec2 A = Sc.vertices[P.face[e]];
            const vec2 B = Sc.vertices[P.face[(e+1==m)?0:e+1]];
            const float d = dist_point_seg(W, A, B);
            if(d < out.dist){ out = NearestPolyResult{ pid, d }; }
        }
    }
    return out;
}

/* Tag all *fluid-side* narrow-band cells with (cell index, object id). Returns Nb (count).
   IMPORTANT: Nx,Ny here MUST be the SAME Nx,Ny you pass into your kernel, and the S pointer
   MUST be laid out with this same (Nx,Ny) stride. */
inline static int tag_boundary_cells_serial(
        const Scene2D& Sc,          // parsed OBJ scene
        float*          S,          // Nx*Ny SDF (fluid >= 0, solid < 0), laid out with this Nx,Ny
        int             Nx,
        int             Ny,
        float           h,
        float           tol,        // e.g. tol = h .. 3*h
        std::vector<int>& cell,     // OUT: linear indices j*Nx + i (with this Nx!)
        std::vector<int>& obj_id)   // OUT: object (poly) id
{
    cell.clear();
    obj_id.clear();
    cell.reserve((size_t)(Nx*Ny*0.02));
    obj_id.reserve(cell.capacity());

    for(int j=0; j<Ny; ++j){
        for(int i=0; i<Nx; ++i){
            const int   Cij = j*Nx + i;    // <- stride matches your kernel Nx
            const float s   = S[Cij];
            const float d   = std::fabs(s);

            // Keep ONLY fluid-side narrow band
            if(!(s >= 0.0f && d <= tol)) continue;

            float gx, gy;
            grad_S_central(S, Nx, Ny, i, j, h, gx, gy);
            const float gn2 = gx*gx + gy*gy;
            if(gn2 <= 1e-24f) continue;

            const float inv_gn = 1.0f/std::sqrt(gn2);
            const float nx = gx * inv_gn;
            const float ny = gy * inv_gn;

            // TRUE cell center (S is at cell centers)
            const float Cx = (i + 0.5f) * h;
            const float Cy = (j + 0.5f) * h;

            // footpoint on boundary: W = C - s * n̂ (S>0 fluid, S<0 solid)
            const vec2 W{ Cx - s * nx, Cy - s * ny };

            NearestPolyResult np = nearest_poly_at_point(Sc, W);
            if(np.poly_id < 0) continue;

            cell.push_back(Cij);        // uses THIS Nx
            obj_id.push_back(np.poly_id);
        }
    }
    return (int)cell.size();
}

static constexpr uint8_t OID_NONE_U8 = 0xFF; // 255 = "not a boundary cell"

/* Build a dense per-cell object id map aligned with S:
   - SAME Nx,Ny stride as S (and as used by your kernel)
   - Marks only FLUID-SIDE narrow band cells: s>=0 && |s|<=tol
   - oid ∈ [0..nObjects-1], 255 means "not boundary"
   Returns 1 on success, 0 on failure. Caller must free(*OID_out). */
inline static int build_oid_dense_cpu(
    const Scene2D& Sc,
    const float*   S,     // SDF laid out with (Nx,Ny) you pass to the kernel
    int            Nx,
    int            Ny,
    float          h,
    float          tol,   // e.g., h .. 3*h
    uint8_t**      OID_out)
{
    if(!S || Nx<=0 || Ny<=0 || !OID_out) return 0;

    const size_t N = (size_t)Nx * (size_t)Ny;
    uint8_t* oid = (uint8_t*)std::malloc(N);
    if(!oid){ std::fprintf(stderr,"[OID] malloc failed\n"); return 0; }

    // init as "not boundary" (sets all to 255)
    std::memset(oid, OID_NONE_U8, N);

    for(int j=0; j<Ny; ++j){
        for(int i=0; i<Nx; ++i){
            const int Cij = j*Nx + i;
            const float s = S[Cij];
            const float d = std::fabs(s);

            // keep ONLY fluid-side narrow band so later zero(S<0) won't wipe our writes
            if(!(s >= 0.0f && d <= tol)) continue;

            // gradient of S (same scheme you use in CUDA)
            float gx, gy;
            grad_S_central(S, Nx, Ny, i, j, h, gx, gy);
            const float gn2 = gx*gx + gy*gy;
            if(gn2 <= 1e-24f) continue;

            const float inv_gn = 1.0f/std::sqrt(gn2);
            const float nx = gx * inv_gn;
            const float ny = gy * inv_gn;

            // geometry: your SDF is node-based (p = i*h, j*h) per your build_sdf_cpu
            // → keep consistent with your kernel (Cx = i*h, Cy = j*h)
            const float Cx = i * h;
            const float Cy = j * h;

            // footpoint on boundary: W = C - s * n̂  (S>0 fluid, S<0 solid)
            const vec2 W{ Cx - s * nx, Cy - s * ny };

            // attribute W to a polygon (skip outer boundary, if present)
            NearestPolyResult np = nearest_poly_at_point(Sc, W);
            if(np.poly_id < 0) continue;

            if(np.poly_id >= 0 && np.poly_id < 255){
                oid[(size_t)Cij] = (uint8_t)np.poly_id;
            } else {
                // too many objects for uint8_t; leave as NONE
            }
        }
    }

    *OID_out = oid;
    return 1;
}


/* =================== object meta: count, max_ID, and names =================== */

// Number of objects (polys) in the scene
inline static int scene_object_count(const Scene2D& Sc){
    return static_cast<int>(Sc.polys.size());
}

// Largest valid object ID (so you can sanity-check obj_id[b] ≤ max_ID)
inline static int scene_max_object_id(const Scene2D& Sc){
    int n = scene_object_count(Sc);
    return n>0 ? (n-1) : -1;
}

// Build a name table aligned with object IDs (index == obj_id from tagger)
inline static void scene_object_names(const Scene2D& Sc, std::vector<std::string>& names){
    names.clear();
    names.reserve(Sc.polys.size());
    for(const Poly& P : Sc.polys) names.push_back(P.name);
}
