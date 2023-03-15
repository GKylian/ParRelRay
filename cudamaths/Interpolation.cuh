#pragma once

#include "f4.cuh"
#include "mat4.cuh"

namespace bilinear
{
	/// <summary> Perform bilinear interpolation using function values at four points (square) </summary>
	/// <param name="p"> The position of the point </param>
	/// <param name="x1"> The position of the bottom left point </param>
	/// <param name="x2"> The position of the top right point </param>
	/// <param name="f"> The function values at the four points. </param>
	__host__ __device__ inline float bilinear(f2 p, f2 x1, f2 x2, f4 f)
	{
		float d = 1.0f/((x1.x-x2.x)*(x1.y-x2.y));
		float a0 = f.x*x2.x*x2.y*d	- f.y*x2.x*x1.y*d	- f.z*x1.x*x2.y*d	+ f.w*x1.x*x1.y*d;
		float a1 = -f.x*x2.y*d		+ f.y*x1.y*d		+ f.z*x2.y*d		- f.w*x1.y*d;
		float a2 = -f.x*x2.x*d		+ f.y*x2.x*d		+ f.z*x1.x*d		- f.w*x1.x*d;
		float a3 = f.x*d			- f.y*d				- f.z*d				+ f.w*d;

		return a0 + a1*p.x + a2*p.y + a3*p.x*p.y;
	}


	/// <summary> Perform bilinear interpolation using function values at the corners of a UNIT square </summary>
	/// <param name="p"> The position of the point </param>
	/// <param name="f"> The function values at the four points. </param>
	__host__ __device__ inline float bilinear(f2 p, f4 f)
	{
		float a0 = f.x;
		float a1 = -f.x			+ f.z;
		float a2 = -f.x	+ f.y;
		float a3 = f.x	- f.y	- f.z	+ f.w;

		return a0 + a1*p.x + a2*p.y + a3*p.x*p.y;
	}

	/// <summary> Perform bilinear interpolation using function values 3d vectors at the corners of a UNIT square </summary>
	/// <param name="p"> The position of the point </param>
	/// <param name="f"> The function values 3d vectors at the four points. </param>
	__host__ __device__ inline f3 bilinear(f2 p, f3 f00, f3 f01, f3 f10, f3 f11)
	{
		f3 a0 = f00;
		f3 a1 = -f00			+ f10;
		f3 a2 = -f00	+ f01;
		f3 a3 = f00		- f01	- f10	+ f11;

		return a0 + a1*p.x + a2*p.y + a3*p.x*p.y;
	}
}