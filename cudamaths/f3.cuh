#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "f2.cuh"

class f3
{
public:
	__host__ __device__ f3() { x = 0; y = 0; z = 0; }
	__host__ __device__ f3(float k) { x = k; y = k; z = k; }
	__host__ __device__ f3(float _x, float _y, float _z) { x = _x; y = _y; z = _z; }

	__host__ __device__ inline const f3 &operator+() const { return *this; }
	__host__ __device__ inline f3 operator-() const { return f3(-x, -y, -z); }

	__host__ __device__ inline f2 xy() { return f2(x, y); }
	__host__ __device__ inline f2 xz() { return f2(x, z); }
	__host__ __device__ inline f2 yz() { return f2(y, z); }

	__host__ __device__ inline f3 &operator+=(const f3 &v2);
	__host__ __device__ inline f3 &operator-=(const f3 &v2);
	__host__ __device__ inline f3 &operator*=(const f3 &v2);
	__host__ __device__ inline f3 &operator/=(const f3 &v2);
	__host__ __device__ inline f3 &operator*=(const float t);
	__host__ __device__ inline f3 &operator/=(const float t);

	__host__ __device__ inline float norm() {	return sqrtf(x*x+y*y+z*z);	}
	__host__ __device__ inline float norm2() {	return x*x+y*y+z*z;		}
	__host__ __device__ inline void normalize() { float k = 1.0f/norm(); x *= k; y *= k; z *= k; }
	__host__ __device__ inline void cinv() { x = 1.0f/x; y = 1.0f/y; z = 1.0f/z; }


	float x, y, z;
};







//--------------------------------------------------
//----------- Normal arithmetic operators ----------
//--------------------------------------------------

__host__ __device__ inline f3 operator+(const f3 &v1, const f3 &v2) {       return f3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);    }
__host__ __device__ inline f3 operator-(const f3 &v1, const f3 &v2) {       return f3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);    }
__host__ __device__ inline f3 operator*(const f3 &v1, const f3 &v2) {       return f3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);    }
__host__ __device__ inline f3 operator/(const f3 &v1, const f3 &v2) {       return f3(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z);    }

__host__ __device__ inline f3 operator*(float t, const f3 &v) {       return f3(v.x*t, v.y*t, v.z*t);    }
__host__ __device__ inline f3 operator*(const f3 &v, float t) {       return f3(v.x*t, v.y*t, v.z*t);    }
__host__ __device__ inline f3 operator/(const f3 &v, float t) {       return f3(v.x/t, v.y/t, v.z/t);    }


__host__ __device__ inline f3 &f3::operator+=(const f3 &v) {        x += v.x; y += v.y; z += v.z; return *this;   }
__host__ __device__ inline f3 &f3::operator-=(const f3 &v) {        x -= v.x; y -= v.y; z -= v.z; return *this;   }
__host__ __device__ inline f3 &f3::operator*=(const f3 &v) {        x *= v.x; y *= v.y; z *= v.z; return *this;   }
__host__ __device__ inline f3 &f3::operator/=(const f3 &v) {        x /= v.x; y /= v.y; z /= v.z; return *this;   }

__host__ __device__ inline f3 &f3::operator*=(const float t) {        x *= t; y *= t; z *= t; return *this;   }
__host__ __device__ inline f3 &f3::operator/=(const float t) {        x /= t; y /= t; z /= t; return *this;   }





//--------------------------------------------------
//------------------- Functions --------------------
//--------------------------------------------------

__host__ __device__ inline float dot(const f3 &a, const f3 &b) {    return a.x*b.x+a.y*b.y+a.z*b.z;                     }
__host__ __device__ inline f3 cross(const f3 &a, const f3 &b) {		return f3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);	}
__host__ __device__ inline float norm(const f3 &a) {                return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);              }
__host__ __device__ inline float norm2(const f3 &a) {               return a.x*a.x+a.y*a.y+a.z*a.z;                     }
__host__ __device__ inline f3 cinv(const f3 &a) {					return f3(1.0f/a.x, 1.0f/a.y, 1.0f/a.z);			}
__host__ __device__ inline f3 normalize(const f3 &a) {				float k = 1.0f/norm(a); return k*a;					}
__host__ __device__ inline f3 abs(const f3 &a) {                    return f3(fabsf(a.x), fabsf(a.y), fabsf(a.z));		}
__host__ __device__ inline f3 cmin(const f3 &a, const f3 &b) {      return f3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));	}
__host__ __device__ inline f3 cmax(const f3 &a, const f3 &b) {      return f3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));	}
__host__ __device__ inline f3 nmin(const f3 &a, const f3 &b) {      return norm2(a) < norm2(b) ? a : b;					}
__host__ __device__ inline f3 nmax(const f3 &a, const f3 &b) {      return norm2(a) > norm2(b) ? a : b;					}
__host__ __device__ inline f3 sqrt(const f3 &a) {					return f3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));		}
__host__ __device__ inline f3 sq(const f3 &a) {						return f3(a.x*a.x, a.y*a.y, a.z*a.z);				}
__host__ __device__ inline bool isnan(const f3 &a) {				return (isnan(a.x) || isnan(a.y) || isnan(a.z));	}

__host__ __device__ inline f3 clamp(const f3 &v, const f3 &m, const f3 &M) { return cmin(cmax(v, m), M); }

__host__ __device__ inline f3 avg(const f3 &a, const f3 &b) {	return f3(0.5f*(a.x+b.x), 0.5f*(a.y+b.y), 0.5f*(a.z+b.z));	}
__host__ __device__ inline f3 avg(const f3 &a, const f3 &b, const f3 &c) {	return f3(0.333333333333f*(a.x+b.x+c.x), 0.333333333333f*(a.y+b.y+c.y), 0.333333333333f*(a.z+b.z+c.z));	}
__host__ __device__ inline f3 avg(const f3 &a, const f3 &b, const f3 &c, const f3 &d) {	return f3(0.25f*(a.x+b.x+c.x+d.x), 0.25f*(a.y+b.y+c.y+d.y), 0.25f*(a.z+b.z+c.z+d.z));	}

__host__ __device__ inline f3 sqavg(const f3 &a, const f3 &b) {
	return f3(sqrtf(0.5f*(a.x*a.x+b.x*b.x)), sqrtf(0.5f*(a.y*a.y+b.y*b.y)), sqrtf(0.5f*(a.z*a.z+b.z*b.z)));
}
__host__ __device__ inline f3 sqavg(const f3 &a, const f3 &b, const f3 &c) {
	return f3(sqrtf(0.333333333f*(a.x*a.x+b.x*b.x+c.x*c.x)), sqrtf(0.333333333f*(a.y*a.y+b.y*b.y+c.y*c.y)), sqrtf(0.333333333f*(a.z*a.z+b.z*b.z+c.z*c.z)));
}
__host__ __device__ inline f3 sqavg(const f3 &a, const f3 &b, const f3 &c, const f3 &d) {
	return f3(sqrtf(0.25f*(a.x*a.x+b.x*b.x+c.x*c.x+d.x*d.x)), sqrtf(0.25f*(a.y*a.y+b.y*b.y+c.y*c.y+d.y*d.y)), sqrtf(0.25f*(a.z*a.z+b.z*b.z+c.z*c.z+d.z*d.z)));
}



//--------------------------------------------------
//--------------- Special definitions --------------
//--------------------------------------------------

__host__ __device__ inline f3 f3_zero() {   return f3(0,0,0);     }
__host__ __device__ inline f3 f3_id() {     return f3(1,1,1);     }





