#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>


class i2
{
public:
	__host__ __device__ i2() { x = 0; y = 0; }
	__host__ __device__ i2(int k) { x = k; y = k; }
	__host__ __device__ i2(int _x, int _y) { x = _x; y = _y; }

	__host__ __device__ inline const i2 &operator+() const { return *this; }
	__host__ __device__ inline i2 operator-() const { return i2(-x, -y); }


	__host__ __device__ inline i2 &operator+=(const i2 &v2);
	__host__ __device__ inline i2 &operator-=(const i2 &v2);
	__host__ __device__ inline i2 &operator*=(const i2 &v2);
	__host__ __device__ inline i2 &operator*=(const int t);

	__host__ __device__ inline float norm() {	return sqrtf(x*x+y*y);	}
	__host__ __device__ inline int norm2() {	return x*x+y*y;		}

	int x, y;
};







//--------------------------------------------------
//----------- Normal arithmetic operators ----------
//--------------------------------------------------

__host__ __device__ inline i2 operator+(const i2 &v1, const i2 &v2) {       return i2(v1.x+v2.x, v1.y+v2.y);    }
__host__ __device__ inline i2 operator-(const i2 &v1, const i2 &v2) {       return i2(v1.x-v2.x, v1.y-v2.y);    }
__host__ __device__ inline i2 operator*(const i2 &v1, const i2 &v2) {       return i2(v1.x*v2.x, v1.y*v2.y);    }

__host__ __device__ inline i2 operator*(int t, const i2 &v) {       return i2(v.x*t, v.y*t);    }
__host__ __device__ inline i2 operator*(const i2 &v, int t) {       return i2(v.x*t, v.y*t);    }


__host__ __device__ inline i2 &i2::operator+=(const i2 &v) {        x += v.x; y += v.y; return *this;   }
__host__ __device__ inline i2 &i2::operator-=(const i2 &v) {        x -= v.x; y -= v.y; return *this;   }
__host__ __device__ inline i2 &i2::operator*=(const i2 &v) {        x *= v.x; y *= v.y; return *this;   }

__host__ __device__ inline i2 &i2::operator*=(const int t) {        x *= t; y *= t; return *this;   }





//--------------------------------------------------
//------------------- Functions --------------------
//--------------------------------------------------

__host__ __device__ inline int dot(const i2 &a, const i2 &b) {      return a.x*b.x+a.y*b.y;                     }
__host__ __device__ inline float norm(const i2 &a) {                return sqrtf(a.x*a.x+a.y*a.y);              }
__host__ __device__ inline int norm2(const i2 &a) {                 return a.x*a.x+a.y*a.y;                     }
__host__ __device__ inline i2 abs(const i2 &a) {                    return i2(abs(a.x), abs(a.y));              }
__host__ __device__ inline i2 cmin(const i2 &a, const i2 &b) {      return i2(min(a.x, b.x), min(a.y, b.y));  }
__host__ __device__ inline i2 cmax(const i2 &a, const i2 &b) {      return i2(max(a.x, b.x), max(a.y, b.y));  }
__host__ __device__ inline i2 nmin(const i2 &a, const i2 &b) {      return norm2(a) < norm2(b) ? a : b;         }
__host__ __device__ inline i2 nmax(const i2 &a, const i2 &b) {      return norm2(a) > norm2(b) ? a : b;         }

__host__ __device__ inline i2 clamp(const i2 &v, const i2 &m, const i2 &M) { return cmin(cmax(v, m), M); }



//--------------------------------------------------
//--------------- Special definitions --------------
//--------------------------------------------------

__host__ __device__ inline i2 i2_zero() {   return i2(0,0);     }
__host__ __device__ inline i2 i2_id() {     return i2(1,1);     }





