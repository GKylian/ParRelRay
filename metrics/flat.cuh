#pragma once
#include "../main/base.h"
#include "../main/Camera.h"
#include "../main/Spacetime.h"
#include "../cudamaths/mats.h"




//Calls from the main code (both CPU and GPU calls):
#pragma region MainCalls


// Called before starting the loop to render all frames.
__host__ void User_INIT(Ray *rays, Camera *cam, Spacetime *space)
{

}


// Called before ray initialization. Return false to skip default ray initialization.
__host__ bool User_RAYINIT(ray *rays, Camera *cam, Spacetime *space)
{
	return true;
}

// Called before ray tracing. Return false to skip default ray tracing.
__host__ bool User_TRACING(ray *rays, Camera *cam, Spacetime *space)
{
	return true;
}

// Called before post-processing (AA, ...). Return false to skip default post-processing.
__host__ bool User_POST(ray *rays, Camera *cam, Spacetime *space)
{
	return true;
}

// Called when the image has been fully rendered (just before saving it).
__host__ void User_END(ray *rays, Camera *cam, Spacetime *space)
{
	return;
}




// Called before the integration loop starts in the solver kernel. Return false to skip the loop.
__device__ bool User_LOOP(ray *iray, Camera cam, Spacetime space)
{
	return true;
}

// Called before updating the 4-velocity and 4-position in the solver kernel. Return false to skip the update.
__device__ bool User_UPDATE(ray *iray, Camera cam, Spacetime space, float h)
{

	f3 c = space.IntersectsScene(iray->r.xyz(), iray->u.xyz(), h);
	if (c.x>=0.0f) { /*printf("Intersection mode should active\n");*/ iray->result = RESULT::INTERSECTION; return false; }


	return true;
}

// Called at the end of the integration iteration in the solver kernel. Return false to stop the integration (stop the solver kernel).
__device__ bool User_ENDLOOP(ray *iray, Camera cam, Spacetime space, tracer trace)
{
	float r = iray->r.xyz().norm();

	if (r >= trace.sph1.rout) {
		iray->result = RESULT::SUCCESS;
		return false;
	}

	f3 c = space.IntersectsScene(iray->r.xyz(), iray->u.xyz(), trace.h);
	if (c.x>=0.0f) { iray->u = f4(c, 0.0f); iray->result = RESULT::INTERSECTION; return false; }

	return true;
}
#pragma endregion


//Required (GPU) functions for the integration: getGeod, getMetric
#pragma region Physics


//Get the geodesics (du/dlambda) to update the ray's 4-velocity (give an expression, or get it from SpaceTime).
__device__ f4 getGeod(f4 p, f4 u, Spacetime space) {
	return f4();
}


//Get the metric (give an expression, or get it from SpaceTime).
__device__ mat4 getMetric(ray *Ray, Spacetime space) {
	mat4 g;
	g[0] = 1; g[5] = 1; g[10] = 1; g[15] = -1;

	return g;
}

#pragma endregion

//We store the color of each ray in the 4-velocity to save space (saves 12*width*height bytes, so 25MB for 1080p and 100MB for 4k)
__global__ void rayToRGB(ray *rays, Camera cam, Spacetime space, tracer trace, int samples)
{
	i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
	if ((ij.x >= cam.getRes().x*samples) || (ij.y >= cam.getRes().y)) return; // Avoid out of bound and accessing other memory
	int pid = ij.y*cam.getRes().x*samples + ij.x;

	ray R = rays[pid];
	float r = R.r.xyz().norm();
	if (R.result == RESULT::SUCCESS) {
		if (r >= trace.sph1.rout) {
			rays[pid].u = f4(trace.sph1.getColorDefault2(rays[pid].u), 0.0f); return;
		}
	}
	else if (R.result == RESULT::INFINITE_GEO) {
		rays[pid].u = f4(0.0f, 0.0f, 0.0f, 0.0f); return;
		//rays[pid].u = f4(0.0f, 0.5f+0.5f*dot(rays[pid].u.xyz(), f3(-1, 0, 0))/rays[pid].u.xyz().norm(), 0.5f, 0.0f); return;
		printf("Infinite geodesics at (%d,%d), p=(%f,%f,%f), u=(%f,%f,%f)\n", ij.x, ij.y, R.r.x, R.r.y, R.r.z, R.u.x, R.u.y, R.u.z);
	}
	else if (R.result==RESULT::INTERSECTION) {
		//rays[pid].u = f4(1.0f, 0.0f, 0.0f, 0.0f); return;
		return;
	}
	else {
		rays[pid].u = f4(0.0f, 0.0f, 0.0f, 0.0f); return;
	}


}