#pragma once
///
///@file Spacetime.h
///@author Kylian G.
///@brief Contains all parameters and data relating to spacetime (black hole parameters and plasma data)
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///
#include "base.h"
#include <vector>
#include "../scene/sphere.h"


class Spacetime
{
public:
	Spacetime() { printf("Created spacetime class !\n"); }

	BlackHole &getBlackHole() { return bh; }
	bool hasBh() { return hasBlackhole; }   void setHasBh(bool hasBh) { hasBlackhole = true; }
	__host__ __device__ float getBhMass() { return bh.M; }		__host__ __device__ float getBhRs() { return bh.rs; }   void setBhMass(float mass) { bh.M = mass; bh.rs = 2.0f*mass; }
	__host__ __device__ float getBhSpin() { return bh.a; }		void setBhSpin(float spin) { bh.a = spin; }
	__host__ __device__ float getBhCharge() { return bh.q; }	void setBhCharge(float charge) { bh.q = charge; }
	__host__ __device__ float getBhP(int i) { return i >= 16 ? 0.0f : bh.p[i]; }   void setBhP(int i, float p) { if (i >= 16) return; bh.p[i] = p; }

	void setDataDimensions(i3 dimensions) { dataDim = dimensions; hasData = true; }


	void setupScene();
	void unloadScene();
	void updateScene(int frame);

	

	//If there is an intersection, return the color, otherwise, return a vector with negative values
	__device__ f3 IntersectsScene(f3 p, f3 u, float h) {
		f4 c = f4(-1, -1, -1, 100000000000); f4 _c;
		for (int i = 0; i<m_nbrspheres; i++) {
			_c = m_spheres[i].intersects(p, u, h);
			if (_c.x>=0.0f&&_c.w<=c.w) c = _c; //pick the closest one to the camera
		}
		
		return c.xyz();
	}

private:
	BlackHole bh; bool hasBlackhole;
	float *density; float *pressure; float *temperature; f3 *velocity; f3 *magnField;
	bool hasData = false;  i3 dataDim;
	Sphere *m_spheres; int m_nbrspheres;



};