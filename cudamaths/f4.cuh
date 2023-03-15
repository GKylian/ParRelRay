#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "f3.cuh"

class f4
{
public:
	__host__ __device__ f4() { x = 0; y = 0; z = 0; w = 0; }
	__host__ __device__ f4(float k) { x = k; y = k; z = k; w = k; }
	__host__ __device__ f4(float _x, float _y, float _z, float _w) { x = _x; y = _y; z = _z; w = _w; }
	__host__ __device__ f4(const f3 &v, float _w) { x = v.x; y = v.y; z = v.z; w = _w; }

	__host__ __device__ inline const f4 &operator+() const { return *this; }
	__host__ __device__ inline f4 operator-() const { return f4(-x, -y, -z, -w); }

	__host__ __device__ inline f3 xyz() { return f3(x, y, z); }

	__host__ __device__ inline f4 &operator+=(const f4 &v2);
	__host__ __device__ inline f4 &operator-=(const f4 &v2);
	__host__ __device__ inline f4 &operator*=(const f4 &v2);
	__host__ __device__ inline f4 &operator/=(const f4 &v2);
	__host__ __device__ inline f4 &operator*=(const float t);
	__host__ __device__ inline f4 &operator/=(const float t);

	__host__ __device__ inline float norm() {	return sqrtf(x*x+y*y+z*z+w*w);	}
	__host__ __device__ inline float norm2() {	return x*x+y*y+z*z+w*w;		}
	__host__ __device__ inline void normalize() { float k = 1.0f/norm(); x *= k; y *= k; z *= k; w *= k; }
	__host__ __device__ inline void cinv() { x = 1.0f/x; y = 1.0f/y; z = 1.0f/z; w = 1.0f/w; }


	float x, y, z, w;
};







//--------------------------------------------------
//----------- Normal arithmetic operators ----------
//--------------------------------------------------

__host__ __device__ inline f4 operator+(const f4 &v1, const f4 &v2) { return f4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w); }
__host__ __device__ inline f4 operator-(const f4 &v1, const f4 &v2) { return f4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w); }
__host__ __device__ inline f4 operator*(const f4 &v1, const f4 &v2) { return f4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w); }
__host__ __device__ inline f4 operator/(const f4 &v1, const f4 &v2) { return f4(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z, v1.w/v2.w); }

__host__ __device__ inline f4 operator*(float t, const f4 &v) { return f4(v.x*t, v.y*t, v.z*t, v.w*t); }
__host__ __device__ inline f4 operator*(const f4 &v, float t) { return f4(v.x*t, v.y*t, v.z*t, v.w*t); }
__host__ __device__ inline f4 operator/(const f4 &v, float t) { return f4(v.x/t, v.y/t, v.z/t, v.w/t); }


__host__ __device__ inline f4 &f4::operator+=(const f4 &v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
__host__ __device__ inline f4 &f4::operator-=(const f4 &v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
__host__ __device__ inline f4 &f4::operator*=(const f4 &v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
__host__ __device__ inline f4 &f4::operator/=(const f4 &v) { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *this; }

__host__ __device__ inline f4 &f4::operator*=(const float t) { x *= t; y *= t; z *= t; w *= t; return *this; }
__host__ __device__ inline f4 &f4::operator/=(const float t) { x /= t; y /= t; z /= t; w /= t; return *this; }





//--------------------------------------------------
//------------------- Functions --------------------
//--------------------------------------------------

__host__ __device__ inline float dot(const f4 &a, const f4 &b) {	return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;										}
__host__ __device__ inline float norm(const f4 &a) {				return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);								}
__host__ __device__ inline float norm2(const f4 &a) {				return a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w;										}
__host__ __device__ inline f4 cinv(const f4 &a) {					return f4(1.0f/a.x, 1.0f/a.y, 1.0f/a.z, 1.0f/a.w); }
__host__ __device__ inline f4 normalize(const f4 &a) {				float k = 1.0f/norm(a); return k*a; }
__host__ __device__ inline f4 abs(const f4 &a) {					return f4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w));							}
__host__ __device__ inline f4 cmin(const f4 &a, const f4 &b) {		return f4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));	}
__host__ __device__ inline f4 cmax(const f4 &a, const f4 &b) {		return f4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));	}
__host__ __device__ inline f4 nmin(const f4 &a, const f4 &b) {		return norm2(a) < norm2(b) ? a : b;											}
__host__ __device__ inline f4 nmax(const f4 &a, const f4 &b) {		return norm2(a) > norm2(b) ? a : b;											}
__host__ __device__ inline float min(const f4 &a) {					return fminf(fminf(fminf(a.x, a.y), a.z), a.w);								}
__host__ __device__ inline float max(const f4 &a) {					return fmaxf(fmaxf(fmaxf(a.x, a.y), a.z), a.w);								}
__host__ __device__ inline bool isnan(const f4 &a) {				return (isnan(a.x) || isnan(a.y) || isnan(a.z) || isnan(a.w));				}




//--------------------------------------------------
//--------------- Special definitions --------------
//--------------------------------------------------

__host__ __device__ inline f4 f4_zero() { return f4(0, 0, 0, 0); }
__host__ __device__ inline f4 f4_id() { return f4(1, 1, 1, 1); }
