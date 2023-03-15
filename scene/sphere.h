#pragma once
///
///@file sphere.h
///@author Kylian G.
///@brief A sphere object you can place in the scene. Ray-object intersection will be performed during the numerical integration, meaning the sphere will be rendered.
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///

#include "../main/base.h"
#include "../cudamaths/tex2d.cuh"
#include "../cudamaths/vecs.h"

#define PI 3.14159265359

/// @brief A sphere than can be placed and rendered in the scene. You can apply a texture to it (e.g. planet's surface)
struct Sphere
{
public:
	__host__ __device__ Sphere() { printf("Created sphere object\n"); }
	Sphere(f3 position, float radius, std::string texture) : c(position), r(radius) { printf("\tCreated sphere at (%f, %f, %f) with radius %f.\n", c.x, c.y, c.z, r); tex.load(texture); }
	__host__ __device__ ~Sphere() { printf("Destroying sphere !\n"); }
	__host__ __device__ void unload() { if (tex.getSize()!=0) tex.unload(); }

	__host__ __device__ void update(f3 position, float radius) { c = position; r = radius; printf("\tSphere now at (%f, %f, %f) with radius %f.\n", c.x, c.y, c.z, r); }

	 /// @brief Performs the ray-object intersection and returns the color at that point.
	 /// @param p The position of the ray
	 /// @param v The velocity of the ray (direction)
	 /// @param h The last step size used in the numerical integration
	 /// @return The color of the sphere at the intersection point. If none, returns f4(-1,-1,-1,-1)
	 __device__ f4 intersects(f3 p, f3 v, float h)
	{
		 v.normalize(); f3 pc = (c-p); pc.normalize();
		//The distance between the ray and the center of the sphere has to be smaller than the radius + how much the ray travelled in the time step
		
		float v2 = v.norm2();
		float disc = sq(dot(v, p-c)) - v2*((p-c).norm2()-r*r);

		if (disc<0) return f4(-1);
		
		float d = (-2.0f*dot(v, p-c)-sqrtf(disc))/v2;
		if (d*d>v2*h) return f4(-1);
		if (d < 0.0f) return f4(-1);


		return f4(getColor((p+v*d)-c),d);
	}

	f3 c; float r;
private:
	tex2d tex;

	/// @brief Gets the color of the textured sphere at the given point
	/// @param inter The point
	/// @return The color of the textured sphere at the given point as f3(r,g,b)
	__device__ f3 getColor(f3 inter)
	{
		float theta = acosf(inter.z/r), phi = atan2f(inter.y, inter.x)+PI;
		f2 uv = f2(phi/(2.0f*PI), theta/PI); f3 sun = (0,0,-1);
		return tex[clamp(uv, f2(), f2(0.999999999f, 0.999999999f))]*fmaxf(0.0f, fminf(0.5f+0.5f*inter.z/r,1));
	}
};


