#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "i3.cuh"

class i4
{
public:
	__host__ __device__ i4() { x = 0; y = 0; z = 0; w = 0; }
	__host__ __device__ i4(int _x, int _y, int _z, int _w) { x = _x; y = _y; z = _z; w = _w; }
	__host__ __device__ i4(i3 v, int _w) { x = v.x; y = v.y; z = v.z; w = _w; }

	__host__ __device__ inline const i4 &operator+() const { return *this; }
	__host__ __device__ inline i4 operator-() const { return i4(-x, -y, -z, -w); }

	__host__ __device__ inline i3 xyz() { return i3(x, y, z); }

	__host__ __device__ inline i4 &operator+=(const i4 &v2);
	__host__ __device__ inline i4 &operator-=(const i4 &v2);
	__host__ __device__ inline i4 &operator*=(const i4 &v2);
	__host__ __device__ inline i4 &operator*=(const int t);

	__host__ __device__ inline float norm() {	return sqrtf(x*x+y*y+z*z+w*w);	}
	__host__ __device__ inline int norm2() {	return x*x+y*y+z*z+w*w;		}

	int x, y, z, w;
};







//--------------------------------------------------
//----------- Normal arithmetic operators ----------
//--------------------------------------------------

__host__ __device__ inline i4 operator+(const i4 &v1, const i4 &v2) { return i4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w); }
__host__ __device__ inline i4 operator-(const i4 &v1, const i4 &v2) { return i4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w); }
__host__ __device__ inline i4 operator*(const i4 &v1, const i4 &v2) { return i4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w); }

__host__ __device__ inline i4 operator*(int t, const i4 &v) { return i4(v.x*t, v.y*t, v.z*t, v.w*t); }
__host__ __device__ inline i4 operator*(const i4 &v, int t) { return i4(v.x*t, v.y*t, v.z*t, v.w*t); }


__host__ __device__ inline i4 &i4::operator+=(const i4 &v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
__host__ __device__ inline i4 &i4::operator-=(const i4 &v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
__host__ __device__ inline i4 &i4::operator*=(const i4 &v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }

__host__ __device__ inline i4 &i4::operator*=(const int t) { x *= t; y *= t; z *= t; w *= t; return *this; }





//--------------------------------------------------
//------------------- Functions --------------------
//--------------------------------------------------

__host__ __device__ inline int dot(const i4 &a, const i4 &b) {	return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;										}
__host__ __device__ inline float norm(const i4 &a) {			return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);								}
__host__ __device__ inline int norm2(const i4 &a) {				return a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w;										}
__host__ __device__ inline i4 abs(const i4 &a) {				return i4(abs(a.x), abs(a.y), abs(a.z), abs(a.w));							}
__host__ __device__ inline i4 cmin(const i4 &a, const i4 &b) {	return i4(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z), fmin(a.w, b.w));	}
__host__ __device__ inline i4 cmax(const i4 &a, const i4 &b) {	return i4(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z), fmax(a.w, b.w));	}
__host__ __device__ inline i4 nmin(const i4 &a, const i4 &b) {	return norm2(a) < norm2(b) ? a : b;											}
__host__ __device__ inline i4 nmax(const i4 &a, const i4 &b) {	return norm2(a) > norm2(b) ? a : b;											}





//--------------------------------------------------
//--------------- Special definitions --------------
//--------------------------------------------------

__host__ __device__ inline i4 i4_zero() { return i4(0, 0, 0, 0); }
__host__ __device__ inline i4 i4_id() { return i4(1, 1, 1, 1); }


