#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "i2.cuh"

class i3
{
public:
	__host__ __device__ i3() { x = 0; y = 0; z = 0; }
	__host__ __device__ i3(int _x, int _y, int _z) { x = _x; y = _y; z = _z; }

	__host__ __device__ inline const i3 &operator+() const { return *this; }
	__host__ __device__ inline i3 operator-() const { return i3(-x, -y, -z); }

	__host__ __device__ inline i2 xy() { return i2(x, y); }
	__host__ __device__ inline i2 xz() { return i2(x, z); }
	__host__ __device__ inline i2 yz() { return i2(y, z); }

	__host__ __device__ inline i3 &operator+=(const i3 &v2);
	__host__ __device__ inline i3 &operator-=(const i3 &v2);
	__host__ __device__ inline i3 &operator*=(const i3 &v2);
	__host__ __device__ inline i3 &operator*=(const int t);

	__host__ __device__ inline float norm() {	return sqrtf(x*x+y*y+z*z);	}
	__host__ __device__ inline int norm2() {	return x*x+y*y+z*z;		}

	int x, y, z;
};







//--------------------------------------------------
//----------- Normal arithmetic operators ----------
//--------------------------------------------------

__host__ __device__ inline i3 operator+(const i3 &v1, const i3 &v2) {       return i3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);    }
__host__ __device__ inline i3 operator-(const i3 &v1, const i3 &v2) {       return i3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);    }
__host__ __device__ inline i3 operator*(const i3 &v1, const i3 &v2) {       return i3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);    }

__host__ __device__ inline i3 operator*(int t, const i3 &v) {       return i3(v.x*t, v.y*t, v.z*t);    }
__host__ __device__ inline i3 operator*(const i3 &v, int t) {       return i3(v.x*t, v.y*t, v.z*t);    }


__host__ __device__ inline i3 &i3::operator+=(const i3 &v) {        x += v.x; y += v.y; z += v.z; return *this;   }
__host__ __device__ inline i3 &i3::operator-=(const i3 &v) {        x -= v.x; y -= v.y; z -= v.z; return *this;   }
__host__ __device__ inline i3 &i3::operator*=(const i3 &v) {        x *= v.x; y *= v.y; z *= v.z; return *this;   }

__host__ __device__ inline i3 &i3::operator*=(const int t) {        x *= t; y *= t; z *= t; return *this;   }





//--------------------------------------------------
//------------------- Functions --------------------
//--------------------------------------------------

__host__ __device__ inline int dot(const i3 &a, const i3 &b) {      return a.x*b.x+a.y*b.y+a.z*b.z;                     }
__host__ __device__ inline float norm(const i3 &a) {                return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);              }
__host__ __device__ inline int norm2(const i3 &a) {                 return a.x*a.x+a.y*a.y+a.z*a.z;                     }
__host__ __device__ inline i3 abs(const i3 &a) {                    return i3(abs(a.x), abs(a.y), abs(a.z));              }
__host__ __device__ inline i3 cmin(const i3 &a, const i3 &b) {      return i3(fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));  }
__host__ __device__ inline i3 cmax(const i3 &a, const i3 &b) {      return i3(fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));  }
__host__ __device__ inline i3 nmin(const i3 &a, const i3 &b) {      return norm2(a) < norm2(b) ? a : b;         }
__host__ __device__ inline i3 nmax(const i3 &a, const i3 &b) {      return norm2(a) > norm2(b) ? a : b;         }





//--------------------------------------------------
//--------------- Special definitions --------------
//--------------------------------------------------

__host__ __device__ inline i3 i3_zero() {   return i3(0,0,0);     }
__host__ __device__ inline i3 i3_id() {     return i3(1,1,1);     }





