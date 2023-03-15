#pragma once
///
///@file Camera.h
///@author Kylian G.
///@brief Definition of the camera class. Creates the rays' direction vectors based on its position and angle.
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///



#include "base.h"


class Camera
{
public:

	__device__ __host__ Camera();
	__device__ __host__ Camera(float vertFov, i2 resolution, SSAA_DISTR SSAAdistribution, int SSAAsamples);
	__device__ __host__ Camera(f3 position, float alpha, float beta, float vertFov, i2 resolution, SSAA_DISTR SSAAdistribution, int SSAAsamples);
	__device__ __host__ i2 getRes() { return res; }
	__device__ __host__ int getAAsamples() { return AAsamples; }
	__host__ SSAA_DISTR getSSAADistr() { return ssaa; }

	__host__ void computeVectors();

	__device__ inline f3 rayDir_PinHole(f2 uv) {
		return normalize(   forward + (2.0f*uv.x-1)*tanf(vfov/2.0f)*res.x/res.y*right + (1.0f-2.0f*uv.y)*tan(vfov/2.0f)*up   );

	}
	__device__ inline f3 rayDir_Panorama(f2 uv) {
		float mu = (0.5f-uv.y)*vfov;  float nu = (0.5f-uv.x)*res.x/res.y * vfov;
		return normalize(cosf(mu)*cosf(nu)*forward + cosf(mu)*sinf(nu)*right + sinf(mu)*mu*up);
	}

	__device__ inline f3 rayDir_Spherical(f2 uv) {
		float nu = (0.5f-uv.y)*vfov;  float mu = (0.5f-uv.x)*res.x/res.y * vfov;
        return normalize(f3( cos(b+nu)*cos(a+mu), cos(b+nu)*sin(a+mu), sin(b+nu) ));
	}

	float a = 0.0f, b = 0.0f; f3 pos, spos;

private:
	f3 forward, right, up;
	float vfov = 1.0471975512f; i2 res;
	SSAA_DISTR ssaa; int AAsamples;


};

