#pragma once

#include "f4.cuh"
#include "i4.cuh"





// --------------------------------------------------------
// ---------- OPERATORS: int input, float output ----------
// --------------------------------------------------------

// Dividing two int vectors -> float vector
__host__ __device__ inline f2 operator/(const i2 &v1, const i2 &v2) { return f2((float)v1.x/v2.x, (float)v1.y/v2.y); }
__host__ __device__ inline f3 operator/(const i3 &v1, const i3 &v2) { return f3((float)v1.x/v2.x, (float)v1.y/v2.y, (float)v1.z/v2.z); }
__host__ __device__ inline f4 operator/(const i4 &v1, const i4 &v2) { return f4((float)v1.x/v2.x, (float)v1.y/v2.y, (float)v1.z/v2.z, (float)v1.w/v2.w); }



// Scaling an int vector with a float -> float vector
__host__ __device__ inline f2 operator*(float t, const i2 &v) { return f2(v.x*t, v.y*t); }
__host__ __device__ inline f2 operator*(const i2 &v, float t) { return f2(v.x*t, v.y*t); }
__host__ __device__ inline f2 operator/(const i2 &v, float t) { return f2(v.x/t, v.y/t); }
__host__ __device__ inline f3 operator*(float t, const i3 &v) { return f3(v.x*t, v.y*t, v.z*t); }
__host__ __device__ inline f3 operator*(const i3 &v, float t) { return f3(v.x*t, v.y*t, v.z*t); }
__host__ __device__ inline f3 operator/(const i3 &v, float t) { return f3(v.x/t, v.y/t, v.z/t); }
__host__ __device__ inline f4 operator*(float t, const i4 &v) { return f4(v.x*t, v.y*t, v.z*t, v.w*t); }
__host__ __device__ inline f4 operator*(const i4 &v, float t) { return f4(v.x*t, v.y*t, v.z*t, v.w*t); }
__host__ __device__ inline f4 operator/(const i4 &v, float t) { return f4(v.x/t, v.y/t, v.z/t, v.w/t); }



// Operation between int and float vectors -> float vector
__host__ __device__ inline f2 operator+(const f2 &v1, const i2 &v2) { return f2(v1.x+v2.x, v1.y+v2.y); }
__host__ __device__ inline f2 operator-(const f2 &v1, const i2 &v2) { return f2(v1.x-v2.x, v1.y-v2.y); }
__host__ __device__ inline f2 operator*(const f2 &v1, const i2 &v2) { return f2(v1.x*v2.x, v1.y*v2.y); }
__host__ __device__ inline f2 operator/(const f2 &v1, const i2 &v2) { return f2(v1.x/v2.x, v1.y/v2.y); }
__host__ __device__ inline f2 operator+(const i2 &v1, const f2 &v2) { return f2(v1.x+v2.x, v1.y+v2.y); }
__host__ __device__ inline f2 operator-(const i2 &v1, const f2 &v2) { return f2(v1.x-v2.x, v1.y-v2.y); }
__host__ __device__ inline f2 operator*(const i2 &v1, const f2 &v2) { return f2(v1.x*v2.x, v1.y*v2.y); }
__host__ __device__ inline f2 operator/(const i2 &v1, const f2 &v2) { return f2(v1.x/v2.x, v1.y/v2.y); }

__host__ __device__ inline f3 operator+(const f3 &v1, const i3 &v2) { return f3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z); }
__host__ __device__ inline f3 operator-(const f3 &v1, const i3 &v2) { return f3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z); }
__host__ __device__ inline f3 operator*(const f3 &v1, const i3 &v2) { return f3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z); }
__host__ __device__ inline f3 operator/(const f3 &v1, const i3 &v2) { return f3(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z); }
__host__ __device__ inline f3 operator+(const i3 &v1, const f3 &v2) { return f3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z); }
__host__ __device__ inline f3 operator-(const i3 &v1, const f3 &v2) { return f3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z); }
__host__ __device__ inline f3 operator*(const i3 &v1, const f3 &v2) { return f3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z); }
__host__ __device__ inline f3 operator/(const i3 &v1, const f3 &v2) { return f3(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z); }

__host__ __device__ inline f4 operator+(const f4 &v1, const i4 &v2) { return f4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w); }
__host__ __device__ inline f4 operator-(const f4 &v1, const i4 &v2) { return f4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w); }
__host__ __device__ inline f4 operator*(const f4 &v1, const i4 &v2) { return f4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w); }
__host__ __device__ inline f4 operator/(const f4 &v1, const i4 &v2) { return f4(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z, v1.w/v2.w); }
__host__ __device__ inline f4 operator+(const i4 &v1, const f4 &v2) { return f4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w); }
__host__ __device__ inline f4 operator-(const i4 &v1, const f4 &v2) { return f4(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w); }
__host__ __device__ inline f4 operator*(const i4 &v1, const f4 &v2) { return f4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w); }
__host__ __device__ inline f4 operator/(const i4 &v1, const f4 &v2) { return f4(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z, v1.w/v2.w); }




// --------------------------------------------------------
// ---------- FUNCTIONS: int input, float output ----------
// --------------------------------------------------------


__host__ __device__ inline f2 cinv(const i2 &a) { return f2(1.0f/a.x, 1.0f/a.y); }
__host__ __device__ inline f2 normalize(const i2 &a) { float k = 1.0f/norm(a); return k*a; }
__host__ __device__ inline f3 cinv(const i3 &a) { return f3(1.0f/a.x, 1.0f/a.y, 1.0f/a.z); }
__host__ __device__ inline f3 normalize(const i3 &a) { float k = 1.0f/norm(a); return k*a; }
__host__ __device__ inline f4 cinv(const i4 &a) { return f4(1.0f/a.x, 1.0f/a.y, 1.0f/a.z, 1.0f/a.w); }
__host__ __device__ inline f4 normalize(const i4 &a) { float k = 1.0f/norm(a); return k*a; }

__host__ __device__ inline i2 round(const f2 &a) { return i2(round(a.x), round(a.y)); }
__host__ __device__ inline i3 round(const f3 &a) { return i3(round(a.x), round(a.y), round(a.z)); }
__host__ __device__ inline i4 round(const f4 &a) { return i4(round(a.x), round(a.y), round(a.z), round(a.w)); }
__host__ __device__ inline i2 ceil(const f2 &a) {  return i2(ceilf(a.x), ceilf(a.y)); }
__host__ __device__ inline i3 ceil(const f3 &a) {  return i3(ceilf(a.x), ceilf(a.y), ceilf(a.z)); }
__host__ __device__ inline i4 ceil(const f4 &a) {  return i4(ceilf(a.x), ceilf(a.y), ceilf(a.z), ceilf(a.w)); }
__host__ __device__ inline i2 floor(const f2 &a) { return i2(floorf(a.x), floorf(a.y)); }
__host__ __device__ inline i3 floor(const f3 &a) { return i3(floorf(a.x), floorf(a.y), floorf(a.z)); }
__host__ __device__ inline i4 floor(const f4 &a) { return i4(floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w)); }