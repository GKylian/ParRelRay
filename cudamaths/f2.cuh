#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string>


__host__ __device__ inline float sq(float x) { return x*x; }


class f2
{
public:
	__host__ __device__ f2() { x = 0; y = 0; }
	__host__ __device__ f2(float k) { x = k; y = k; }
	__host__ __device__ f2(float _x, float _y) { x = _x; y = _y; }

	__host__ __device__ inline const f2 &operator+() const { return *this; }
	__host__ __device__ inline f2 operator-() const { return f2(-x, -y); }


	__host__ __device__ inline f2 &operator+=(const f2 &v2);
	__host__ __device__ inline f2 &operator-=(const f2 &v2);
	__host__ __device__ inline f2 &operator*=(const f2 &v2);
	__host__ __device__ inline f2 &operator/=(const f2 &v2);
	__host__ __device__ inline f2 &operator*=(const float t);

	__host__ __device__ inline float norm() {	return sqrtf(x*x+y*y);	}
	__host__ __device__ inline float norm2() {	return x*x+y*y;		}
	__host__ __device__ inline void normalize() { float k = 1.0f/norm(); x *= k; y *= k; }
	__host__ __device__ inline void cinv() { x = 1.0f/x; y = 1.0f/y; }


	float x, y;
};







//--------------------------------------------------
//----------- Normal arithmetic operators ----------
//--------------------------------------------------

__host__ __device__ inline f2 operator+(const f2 &v1, const f2 &v2) {       return f2(v1.x+v2.x, v1.y+v2.y);    }
__host__ __device__ inline f2 operator-(const f2 &v1, const f2 &v2) {       return f2(v1.x-v2.x, v1.y-v2.y);    }
__host__ __device__ inline f2 operator*(const f2 &v1, const f2 &v2) {       return f2(v1.x*v2.x, v1.y*v2.y);    }
__host__ __device__ inline f2 operator/(const f2 &v1, const f2 &v2) {       return f2(v1.x/v2.x, v1.y/v2.y);    }

__host__ __device__ inline f2 operator*(float t, const f2 &v) {       return f2(v.x*t, v.y*t);    }
__host__ __device__ inline f2 operator*(const f2 &v, float t) {       return f2(v.x*t, v.y*t);    }
__host__ __device__ inline f2 operator/(const f2 &v, float t) {       return f2(v.x/t, v.y/t);    }


__host__ __device__ inline f2 &f2::operator+=(const f2 &v) {        x += v.x; y += v.y; return *this;   }
__host__ __device__ inline f2 &f2::operator-=(const f2 &v) {        x -= v.x; y -= v.y; return *this;   }
__host__ __device__ inline f2 &f2::operator*=(const f2 &v) {        x *= v.x; y *= v.y; return *this;   }
__host__ __device__ inline f2 &f2::operator/=(const f2 &v) {        x /= v.x; y /= v.y; return *this;   }

__host__ __device__ inline f2 &f2::operator*=(const float t) {        x *= t; y *= t; return *this;   }





//--------------------------------------------------
//------------------- Functions --------------------
//--------------------------------------------------

__host__ __device__ inline float dot(const f2 &a, const f2 &b) {    return a.x*b.x+a.y*b.y;							}
__host__ __device__ inline float norm(const f2 &a) {                return sqrtf(a.x*a.x+a.y*a.y);					}
__host__ __device__ inline float norm2(const f2 &a) {               return a.x*a.x+a.y*a.y;							}
__host__ __device__ inline f2 abs(const f2 &a) {                    return f2(fabsf(a.x), fabsf(a.y));				}
__host__ __device__ inline f2 cinv(const f2 &a) {                   return f2(1.0f/a.x, 1.0f/a.y);					}
__host__ __device__ inline f2 normalize(const f2 &a) {				float k = 1.0f/norm(a); return k*a;}
__host__ __device__ inline f2 cmin(const f2 &a, const f2 &b) {      return f2(fminf(a.x, b.x), fminf(a.y, b.y));	}
__host__ __device__ inline f2 cmax(const f2 &a, const f2 &b) {      return f2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));	}
__host__ __device__ inline f2 nmin(const f2 &a, const f2 &b) {      return norm2(a) < norm2(b) ? a : b;				}
__host__ __device__ inline f2 nmax(const f2 &a, const f2 &b) {      return norm2(a) > norm2(b) ? a : b;				}
__host__ __device__ inline bool isnan(const f2 &a) {				return (isnan(a.x) || isnan(a.y));				}

__host__ __device__ inline f2 clamp(const f2 &v, const f2 &m, const f2 &M) { return cmin(cmax(v, m), M); }




//--------------------------------------------------
//--------------- Special definitions --------------
//--------------------------------------------------

__host__ __device__ inline f2 f2_zero() {   return f2(0,0);     }
__host__ __device__ inline f2 f2_id() {     return f2(1,1);     }





