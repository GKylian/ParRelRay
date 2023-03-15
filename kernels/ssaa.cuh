#pragma once

///
///@file ssaa.cuh
///@author Kylian G.
///@brief Creates the rays according to the anti-aliasing chosen (if any). Can also apply AA after the image has been rendered.
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

#include "../main/base.h"
#include "../cudamaths/tex2d.cuh"

/// @brief Initiates the GPU random number generator
/// @param seed The seed for the generator
/// @param res The resolution of the image
/// @param samples Samples per pixel
/// @param states CUDA states array
__global__ void randinit(unsigned int seed, i2 res, int samples, curandState_t *states)
{
    i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
    curand_init(seed, ij.y*res.x*samples + ij.x, 0, &states[ij.y*res.x + ij.x]);
}

/// @brief Create all the ray objects ('samples' rays for each pixel).
/// @param rays The ray array
/// @param res The resolution of the image
/// @param samples How many samples (rays) per pixel
__global__ void createRays_regular(ray *rays, i2 res, int samples)
{
    i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
    if ((ij.x >= res.x) || (ij.y >= res.y)) return; // Avoid out of bound and accessing other memory

    int n = (int)roundf(sqrtf(samples));
    f2 duv = cinv(res)/(n+1.0f);
    for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++){
        int pid = ij.y*res.x*samples + ij.x*samples + j*n+i;
        rays[pid].uv = ij/res + i2(i+1, j+1)*duv;
    }
}

/// @brief Creates 'samples' rays per pixel with fully random distribution
/// @param rays The ray array
/// @param res The resolution of the image
/// @param samples Samples (rays) per pixel
/// @param states CUDA states (used for RNG)
__global__ void createRays_random(ray *rays, i2 res, int samples, curandState_t *states)
{
    i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
    if ((ij.x >= res.x) || (ij.y >= res.y)) return; // Avoid out of bound and accessing other memory

    for (int i = 0; i < samples; i++) {
        f2 duv = f2(curand_uniform(&states[ij.y*res.x*samples + ij.x]), curand_uniform(&states[ij.y*res.x*samples + ij.x]));
        int pid = ij.y*res.x*samples + ij.x*samples + i;
        rays[pid].uv = ij/res + duv;
    }
}

/// @brief Creates 'samples' rays per pixel with a jittered distribution (faster Poisson approximation)
/// @param rays The ray array
/// @param res The resolution of the image
/// @param samples Samples (rays) per pixel
/// @param states CUDA states (used for RNG)
__global__ void createRays_jittered(ray *rays, i2 res, int samples, curandState_t *states)
{
    i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
    if ((ij.x >= res.x) || (ij.y >= res.y)) return; // Avoid out of bound and accessing other memory

    int n = (int)roundf(sqrtf(samples));
    f2 duv = cinv(res)/n;
    for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++){
        f2 dduv = f2(curand_uniform(&states[ij.y*res.x*samples + ij.x]), curand_uniform(&states[ij.y*res.x*samples + ij.x]));
        int pid = ij.y*res.x*samples + ij.x*samples + j*n+i;
        rays[pid].uv = ij/res + i2(i, j)*duv + dduv*duv; //Top left corner of the subcell + random float [0,1]*duv

    }
}

/// @brief Creates the rays with no multisample anti-aliasing (1 sample=ray per pixel)
/// @param rays The ray array
/// @param res The resolution of the image
/// @return 
__global__ void createRays_none(ray *rays, i2 res)
{
    i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
    if ((ij.x >= res.x) || (ij.y >= res.y)) return; // Avoid out of bound and accessing other memory
    int pid = ij.y*res.x + ij.x;

    rays[pid].uv = ij/res + 0.5f*cinv(res);
}



/// @brief For each pixel, combines the samples into one color
/// @param rays The final rays
/// @param image The image we're outputting
/// @param res The resolution of the image
/// @param samples How many samples (rays) per image
/// @return 
__global__ void combineSamples(ray *rays, tex2d image, i2 res, int samples)
{
    //Treat it as a (samples*res.x, res.y) image
    i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
    if ((ij.x >= res.x) || (ij.y >= res.y)) return; // Avoid out of bound and accessing other memory
    int pid = ij.y*res.x + ij.x;

    if (samples == 1)
    {
        image[pid] = clamp(rays[pid].u.xyz(),f3(0),f3(1));
        return;
    }

    f3 c;
    for (int i = 0; i < samples; i++)
    {
        c += clamp(sq(rays[pid*samples + i].u.xyz()),f3(0),f3(1));
    }
    c /= samples;
    c = sqrt(c);

    image[pid] = c;
}




//!==============================================
//!=============== FXAA ALGORITHM ===============
//!==============================================


__device__ float luma(f3 c) {
    return c.x*0.299*(1.0f/255.0f)+c.y*0.587*(1.0f/255.0f)+c.z*0.114*(1.0f/255.0f);
}

__device__ const float FXAA_SPAN_MAX = 8.0f;   __device__ const float FXAA_REDUCE_MUL = 1.0f/8.0f;   __device__ const float FXAA_REDUCE_MIN = (1.0f/128.0f);

/// @brief Applies the Nvidia's 'FXAA' anti-aliasing algorithm to the rendered image.
/// @param im The rendered image
/// @param fim The final image with FXAA applied
__global__ void fxaa(tex2d im, tex2d fim)
{
    i2 ij = i2(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
    if ((ij.x>=im.res.x)||(ij.y>=im.res.y)) return; // Avoid out of bound and accessing other memory
    ij = clamp(ij, i2(1, 1), i2(im.res.x-2, im.res.y-2));
    int pid = ij.y*im.res.x+ij.x;

   /* fim[pid] = im[pid];*/

    float lumaNW = luma(im[i2(ij.x-1, ij.y-1)]);   float lumaNE = luma(im[i2(ij.x+1, ij.y-1)]);
    float lumaSW = luma(im[i2(ij.x-1, ij.y+1)]);   float lumaSE = luma(im[i2(ij.x+1, ij.y+1)]);
    float lumaM = luma(im[ij]);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    f2 dir;   dir.x = -((lumaNW+lumaNE)-(lumaSW+lumaSE)); dir.y = ((lumaNW+lumaSW)-(lumaNE+lumaSE));
    float dirReduce = max((lumaNW+lumaNE+lumaSW+lumaSE)*(0.25f*FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0f/(min(abs(dir.x), abs(dir.y))+dirReduce);
    dir = clamp(dir*rcpDirMin, f2(-FXAA_SPAN_MAX), f2(FXAA_SPAN_MAX));

    f3 rgbA =               ( im[(ij+dir*(1.0f/3.0f-0.5f))/im.res]*0.5f  + im[(ij+dir*(2.0f/3.0f-0.5f))/im.res]*0.5f );
    f3 rgbB = 0.5f*rgbA +   ( im[(ij+dir*(-0.5f))/im.res]*0.25           + im[(ij+dir*(0.5f))/im.res]*0.25f );
    float lumaB = luma(rgbB);

    if ((lumaB<lumaMin)||(lumaB>lumaMax)) {
        fim[pid] = rgbA; return;
    }
    else {
        fim[pid] = rgbB; return;
    }

}