#pragma once

#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <cuda_runtime.h>

#include "if_interop.cuh"
#include "image.cuh"


//#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
//void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
//	if (result) {
//		fprintf(stderr, "CUDA error = %d at %s:%d '%s'\n", static_cast<unsigned int>(result), file, line, func);
//		// Make sure we call CUDA Device Reset before exiting
//		cudaDeviceReset();
//		exit(99);
//	}
//}



#define BILINEAR

class tex2d
{
public:

	__device__ __host__ void checkCuda(cudaError_t result)
	{
		if (result) {
			fprintf(stderr, "CUDA error = %d\n", static_cast<unsigned int>(result));
			cudaDeviceReset();
			exit(99);
		}
	}

	__device__ __host__ tex2d() {}

	__device__ __host__ tex2d(i2 resolution)
	{
		res = resolution; size = res.x*res.y*sizeof(f3);
		checkCuda(cudaMalloc((void **)&image, size));
	}

	__host__ __device__ size_t getSize() { return size; }

	//Loads the image and put it to GPU memory
	__host__ void load(std::string path)
	{
		f3 *image_host = 0;

		printf("\tLoading image from file %s\n", path.c_str());

		try {
			Images::get().loadRGBFromFile(path, &image_host, &res, &size);
			printf("\t\tLoaded image (%dx%d) at %s (array size: %f MB)\n", res.x, res.y, path.c_str(), size/1.0e6);
		}
		catch (const std::exception &e) {
			printf("\t\tCould not load image at %s to host memory. Error: %s\n", path.c_str(), e.what());
			return;
		}

		try {
			if(image_host == NULL) printf("a");
			checkCuda(cudaMalloc((void **)&image, size));
			checkCuda(cudaMemcpy(image, image_host, size, cudaMemcpyHostToDevice));
			printf("\t\tTransfered the image to device memory.\n");

			Images::get().freeMemory(image_host);
			printf("\t\tFreed host memory.\n");

		}
		catch (const std::exception &e)
		{
			printf("\t\tCould not transfer image to device memory at %s. Error: %s\n", path.c_str(), e.what());
			return;
		}
	}

	__host__ void unload() { checkCuda(cudaFree(image)); }
	__host__ void save(std::string path)
	{
		f3 *image_host;
		try {
			printf("\tSaving image to file %s\n", path.c_str());
			image_host = (f3 *)malloc(size);
			checkCuda(cudaMemcpy(image_host, image, size, cudaMemcpyDeviceToHost));
			Images::get().savePNGToFile(path, image_host, res);
			
			free(image_host);
			printf("\t\tImage saved !\n");
		}
		catch (const std::exception &e) {
			printf("\t\tCould not save image to %s\n", path.c_str());
			return;
		}
	}

//	uv = uv*res; i2 ij = floor(uv); f2 p = uv-ij;
//
//	/*if (ij.x>=res.x||ij.y>=res.y) return image[(ij.y-1)*res.x+(ij.x-1)];*/
//
//#ifndef BILINEAR
//	return image[ij.y*res.x+ij.x]*(1-p.x)*(1-p.y)+image[ij.y*res.x+ij.x+1]*p.x*(1-p.y)+image[(ij.y+1)*res.x+ij.x]*(1-p.x)*p.y+image[(ij.y+1)*res.x+ij.x+1]*p.x*p.y;
//#else
//	return image[ij.y*res.x+ij.x];
//#endif // BILINEAR

	//TODO: IMPLEMENT BILINEAR INTERPOLATION FOR TEXTURES
	__device__ inline f3 operator[](f2 uv) const {
		uv = uv*res; i2 ij = floor(uv); f2 p = uv-ij;   ij = clamp(ij, i2(0), res-i2(2));
#ifdef BILINEAR
		return image[ij.y*res.x+ij.x]*(1-p.x)*(1-p.y)+image[ij.y*res.x+ij.x+1]*p.x*(1-p.y)+image[(ij.y+1)*res.x+ij.x]*(1-p.x)*p.y+image[(ij.y+1)*res.x+ij.x+1]*p.x*p.y;
#else
		return image[ij.y*res.x+ij.x];
#endif // BILINEAR
	}

	__device__ inline f3 &operator[](f2 uv) {
		uv = uv*res; i2 ij = floor(uv); f2 p = uv-ij;   ij = clamp(ij, i2(0), res-i2(2));
#ifdef BILINEAR
		return image[ij.y*res.x+ij.x]*(1-p.x)*(1-p.y)+image[ij.y*res.x+ij.x+1]*p.x*(1-p.y)+image[(ij.y+1)*res.x+ij.x]*(1-p.x)*p.y+image[(ij.y+1)*res.x+ij.x+1]*p.x*p.y;
#else
		return image[ij.y*res.x+ij.x];
#endif // BILINEAR
	}

	__host__ __device__ inline f3 operator[](i2 ij) const	{ ij = clamp(ij, i2(0), res-i2(1)); return image[ij.y*res.x+ij.x]; }
	__host__ __device__ inline f3 &operator[](i2 ij)		{ ij = clamp(ij, i2(0), res-i2(1)); return image[ij.y*res.x+ij.x]; }

	__device__ inline f3 operator[](int i) const { return image[i]; }
	__device__ inline f3 &operator[] (int i) { return image[i]; }

	i2 res;
private:
	f3 *image; size_t size = 0;
	int channels = 3;


};
