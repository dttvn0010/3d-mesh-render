//nvcc -O3 -o librender.so -Xcompiler -fPIC --shared render.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

struct ProjectedVertex
{
    int x;
    int y;
    float z;
    float u;
    float v;
    float w;
};

struct ProjectedTriangle
{
    ProjectedVertex a;
    ProjectedVertex b;
    ProjectedVertex c;
};

struct  Point2i
{
    int x;
    int y;
};


struct BarycentricWeight
{
    float alpha;
    float beta;
    float gamma;
};

__device__ BarycentricWeight calculate_barycentric_weights(
    const ProjectedVertex* a,
    const ProjectedVertex* b,
    const ProjectedVertex* c,
    const Point2i* p
)
{
    Point2i ab{b->x - a->x, b->y - a->y};
    Point2i bc{c->x - b->x, c->y - b->y};
    Point2i ac{c->x - a->x, c->y - a->y};
    Point2i ap{p->x - a->x, p->y - a->y};
    Point2i bp{p->x - b->x, p->y - b->y};
    float triangle_area = ab.x * ac.y - ab.y * ac.x;
    if(triangle_area == 0)
    {
        return BarycentricWeight{1.0f/3, 1.0f/3, 1.0f/3};
    }
    float alpha = float(bc.x * bp.y - bc.y * bp.x) / triangle_area;      // bc x bp
    float beta =  float(ap.x * ac.y - ap.y * ac.x) / triangle_area;      // ap x ac
    float gamma = 1.0f - alpha - beta;
    return BarycentricWeight{alpha, beta, gamma};
}

__device__ void draw_line_interpolated(
    uint8_t* img,
    int img_width,
    int img_height,
    float* depth_buffer,
    const uint8_t* texture,
    int texture_width,
    int texture_height,
    const ProjectedVertex* a,
    const ProjectedVertex* b,
    const ProjectedVertex* c,
    int y,
    int x_start,
    int x_end
){
    if(x_end < x_start)
    {
        int tmp = x_start;
        x_start = x_end;
        x_end = tmp;   
    }
    
    if(x_start < 0) x_start = 0;
    if(x_end > img_width - 1) x_end = img_width - 1;
    
    for(int x = x_start; x <= x_end; x++)
    {
        Point2i pt{x, y};
        BarycentricWeight bw = calculate_barycentric_weights(a, b, c, &pt);
        float depth = (a->z * bw.alpha + b->z * bw.beta + c->z * bw.gamma);

        if(depth >= depth_buffer[y * img_width + x])
            continue;

        atomicExch(depth_buffer + y * img_width + x, depth);
            
        float w_a = bw.alpha / a->w, w_b = bw.beta / b->w, w_c = bw.gamma /  c->w;
        float u = (a->u * w_a + b->u * w_b + c->u * w_c) / (w_a + w_b + w_c);
        float v = (a->v * w_a + b->v * w_b + c->v * w_c) / (w_a + w_b + w_c);
        if(u < 0.0f) u = 0.0f;
        if(v < 0.0f) v = 0.0f;

        u = u - int(u);
        v = v - int(v);

        u = int(texture_width * u);
        v = int(texture_height * v);
        int src_idx = 3 * ((texture_height - 1 - int(v)) * texture_width +  int(u));
        int dst_idx = 3 * (y * img_width + x);
        img[dst_idx + 0] = texture[src_idx + 0]; // r
        img[dst_idx + 1] = texture[src_idx + 1]; // g
        img[dst_idx + 2] = texture[src_idx + 2]; // b
    }
}


__global__ void draw_textured_triangle(
    uint8_t* img,
    int img_width,
    int img_height,
    float* depth_buffer,
    const uint8_t* texture,
    int texture_width,
    int texture_height,
    const ProjectedTriangle* triangles,
    int n_triangle
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= n_triangle) return;

    const ProjectedVertex* a = &triangles[idx].a;
    const ProjectedVertex* b = &triangles[idx].b;
    const ProjectedVertex* c = &triangles[idx].c;

    float k_ca = (c->y != a->y) ? float(c->x - a->x) / (c->y - a->y) : 0.0f;
    float k_ba = (b->y != a->y) ? float(b->x - a->x) / (b->y - a->y) : 0.0f;
    float k_cb = (c->y != b->y) ? float(c->x - b->x) / (c->y - b->y) : 0.0f;

    if(b->y != a->y)
    {
        int ymin = a->y < 0 ? 0 : a->y;
        int ymax = b->y > img_height - 1? img_height - 1: b->y;
        for(int y=ymin; y <= ymax; y++)
        {
            int x_start = b->x + int((y - b->y) * k_ba);
            int x_end = a->x + int((y - a->y) * k_ca);
            
            draw_line_interpolated(
                img, 
                img_width,
                img_height,
                depth_buffer,
                texture,
                texture_width,
                texture_height,
                a, b, c,
                y,
                x_start,
                x_end
            );
        }
    }

    if(b->y != c->y)
    {
        int ymin = b->y < 0 ? 0 : b->y;
        int ymax = c->y > img_height - 1? img_height - 1: c->y;
        for(int y=ymin; y <= ymax; y++)
        {
            int x_start = b->x + int((y - b->y) * k_cb);
            int x_end = a->x + int((y - a->y) * k_ca);

            draw_line_interpolated(
                img, 
                img_width,
                img_height,
                depth_buffer,
                texture,
                texture_width,
                texture_height,
                a, b, c,
                y,
                x_start,
                x_end
            );
        }
    }
}

__global__ void init_depth_buffer(
    float* depth_buffer,
    int img_width,
    int img_height
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= img_width * img_height) return;
    depth_buffer[idx] = 1.0f;
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" void render(
    uint8_t* img,
    int img_width,
    int img_height,
    const uint8_t* texture,
    int texture_width,
    int texture_height,
    const ProjectedTriangle* triangles,
    int n_triangle
)
{
    uint8_t* img_device;
    cudaMalloc(&img_device, 3 * img_width * img_height);
    cudaMemset(img_device, 0, 3 * img_width * img_height);

    float* depth_buffer;
    cudaMalloc(&depth_buffer, img_width * img_height * sizeof(float));
    init_depth_buffer<<<(img_width * img_height + 255)/256, 256 >>>(depth_buffer, img_width, img_height);
    gpuErrchk( cudaDeviceSynchronize() );

    uint8_t* texture_device;
    cudaMalloc(&texture_device, 3 * texture_width * texture_height);
    cudaMemcpy(texture_device, texture, 3 * texture_width * texture_height, cudaMemcpyHostToDevice);

    ProjectedTriangle* triangles_device;
    cudaMalloc(&triangles_device, n_triangle * sizeof(ProjectedTriangle));
    cudaMemcpy(triangles_device, triangles, n_triangle * sizeof(ProjectedTriangle), cudaMemcpyHostToDevice);

    draw_textured_triangle<<<(n_triangle + 255)/256, 256>>>(
        img_device,
        img_width,
        img_height,
        depth_buffer,
        texture_device,
        texture_width,
        texture_height,
        triangles_device,
        n_triangle
    );

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //cudaDeviceSynchronize();
    cudaMemcpy(img, img_device, 3 * img_width * img_height, cudaMemcpyDeviceToHost);

    cudaFree(img_device);
    cudaFree(depth_buffer);
    cudaFree(texture_device);
    cudaFree(triangles_device);
}