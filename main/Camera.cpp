#include "Camera.h"

__device__ __host__ Camera::Camera() : AAsamples(1), ssaa(SSAA_DISTR::NONE), vfov(1.0471975512f)
{ }


__device__ __host__ Camera::Camera(float vertFov, i2 resolution, SSAA_DISTR SSAAdistribution, int SSAAsamples)
{
	vfov = vertFov; res = resolution; ssaa = SSAAdistribution; AAsamples = SSAAsamples;
}


__device__ __host__ Camera::Camera(f3 position, float alpha, float beta, float vertFov, i2 resolution, SSAA_DISTR SSAAdistribution, int SSAAsamples)
{
	pos = position; a = alpha; b = beta; vfov = vertFov; res = resolution; ssaa = SSAAdistribution; AAsamples = SSAAsamples;
}


__host__ void Camera::computeVectors() {
    printf("\tComputing forward, down and right camera vectors from alpha and beta...\n");
    forward = normalize(f3(cosf(b)*cosf(a), cosf(b)*sinf(a), -sinf(b))); //a = 0 ~ forward to positive x
    up = normalize(f3(cosf(a)*sinf(b), sinf(a)*sinf(b), cosf(b)));
    right = normalize(cross(forward, up));
    printf("\t\ta = %f, b = %f\n", a, b);
    printf("\t\tForward vector: (%f, %f, %f)\n", forward.x, forward.y, forward.z);
    printf("\t\tUp vector: (%f, %f, %f)\n", up.x, up.y, up.z);
    printf("\t\tRight vector: (%f, %f, %f)\n", right.x, right.y, right.z);
}

