#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "f3.cuh"



// for(int i = 0; i < 16; i++)
class mat4
{
public:
	__host__ __device__ mat4() { for (int i = 0; i < 16; i++) e[i] = 0; }

	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float &operator[](int i) { return e[i]; };

	__host__ __device__ inline const mat4 &operator+() const { return *this; }
	__host__ __device__ inline mat4 operator-() const { mat4 m; for (int i = 0; i < 16; i++) m[i] = -e[i]; return m; }

	//__host__ __device__ inline mat4 &operator+=(const mat4 &v2);
	//__host__ __device__ inline mat4 &operator-=(const mat4 &v2);
	//__host__ __device__ inline mat4 &operator*=(const mat4 &v2);
	//__host__ __device__ inline mat4 &operator/=(const mat4 &v2);
	//__host__ __device__ inline mat4 &operator*=(const float t);

	//Fills the lower triangle with the upper triangle (gives a symmetric matrix)
	__host__ __device__ inline void sym() {
		e[4] = e[1]; e[8] = e[2]; e[9] = e[6]; e[12] = e[3]; e[13] = e[7]; e[14] = e[8];
	}

	__host__ __device__ inline void cinv() { for (int i = 0; i < 16; i++) e[i] = 1.0f/e[i]; }
	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float m(int i, int j) { return e[i*4+j]; }

	float e[16];
};







//--------------------------------------------------
//----------- Normal arithmetic operators ----------
//--------------------------------------------------

//__host__ __device__ inline f4 operator+(const f4 &v1, const f4 &v2) { return f4(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w); }
//__host__ __device__ inline f4 &f4::operator+=(const f4 &v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }







//--------------------------------------------------
//------------------- Functions --------------------
//--------------------------------------------------

//__host__ __device__ inline float dot(const f4 &a, const f4 &b) {	return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;										}
//__host__ __device__ inline mat4 inv(const f4 &m) {
//	
//}
__host__ __device__ inline float dot(const mat4 &a, const mat4 &b)
{
	float sum = 0.0f; for (int i = 0; i<16; i++) sum += a[i]*b[i];
	return sum;
}




//--------------------------------------------------
//--------------- Special definitions --------------
//--------------------------------------------------

//Identity matrix
__host__ __device__ inline mat4 mat4_id() { mat4 m; m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1; return m; }
//Matrix filled with k
__host__ __device__ inline mat4 mat4_full(float k) { mat4 m; for (int i = 0; i < 16; i++) m[i] = k; return m; }
//Matrix filled with all combinations of components of give four-vector
__host__ __device__ inline mat4 mat4_comb(const f4 &v) {
	mat4 m;
	m[0] = v.x*v.x;		m[1] = v.x*v.y;		m[2] = v.x*v.z;		m[3] = v.x*v.w;
						m[5] = v.y*v.y;		m[6] = v.y*v.z;		m[7] = v.y*v.w;
											m[10] = v.z*v.z;	m[11] = v.z*v.w;
																m[15] = v.w*v.w;
	m.sym();
	return m;
}
