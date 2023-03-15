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
	return true;
}

// Called at the end of the integration iteration in the solver kernel. Return false to stop the integration (stop the solver kernel).
__device__ bool User_ENDLOOP(ray *iray, Camera cam, Spacetime space, tracer trace)
{
	//If the angle is smaller than 2 degrees (0.035 radians), we tp the ray on the other side
	/*if (iray->r.y >= M_PI || iray->r.y <= 0) {
		iray->u.y = -iray->u.y;
		iray->r.z += M_PI;
	}*/

	if (fabsf(iray->r.y) <= 0.04 || fabsf(iray->r.y-M_PI) <= 0.04) {
		iray->r.z += M_PI;
		iray->u.y = -iray->u.y;
	}
	

	/* Keep phi in the [0; 2pi] range, and theta in [0, pi] */
	iray->r.z += 2.0f*M_PI; iray->r.z = fmodf(iray->r.z, 2.0f*M_PI);
	iray->r.y = fabsf(iray->r.y); iray->r.y = fmodf(iray->r.y, 2.0f*M_PI);
	if (iray->r.y > M_PI && iray->r.y <= 2.0f*M_PI) iray->r.y = 2.0f*M_PI - iray->r.y;


	if (iray->r.x >= trace.sph1.rout || iray->r.x <= space.getBhRs()+0.001f) {
		iray->result = RESULT::SUCCESS;
		return false;
	}
	return true;
}
#pragma endregion


//Required (GPU) functions for the integration: getGeod, getMetric
#pragma region Physics

//Get the geodesics (du/dlambda) to update the ray's 4-velocity (give an expression, or get it from SpaceTime).
__device__ f4 getGeod(f4 x, f4 u, Spacetime space) {
	f4 du;
	float rs = space.getBhRs(); float r = x.x;

	float tanth = tanf(x.y); float sinth = sinf(x.y);
	if (fabsf(tanth) < 1e-4) {
		tanth = 1e-4; sinth = 1e-4;
	}


	du.x =	-0.5f*rs*(r - rs)/(r*r*r)			* u.w*u.w
			+ 0.5f*rs/(r*r - r*rs)				* u.x*u.x
			+ (r-rs)*(u.y*u.y + sinth*sinth		* u.z*u.z);
	du.y =	-2.0f/r								* u.x*u.y
			+ sinth*cosf(x.y)					* u.z*u.z;
	du.z =	-2.0f/r								* u.x*u.z
			- 2.0f/tanth						* u.y*u.z;
	du.w =	-rs/(r*r - r*rs)					* u.x*u.w;


	return du;
}

//Get the metric (give an expression, or get it from SpaceTime).
__device__ mat4 getMetric(ray *Ray, Spacetime space) {
	mat4 g;
	float rs = space.getBhRs();
	f4 p = Ray->r; float r = p.x;

	// rr	rth		rph		rt
	// thr	thth	thph	tht
	// phr	phth	phph	pht
	// tr	tth		tph		tt
	/*
	g[0] = 1.0f/(1.0f-rs/r.x);		g[1] = 0.0f;	g[2] = 0.0f;						g[3] = 0.0f;
	g[4] = 0.0f;					g[5] = r*r;		g[6] = 0.0f;						g[7] = 0.0f;
	g[8] = 0.0f;					g[9] = 0.0f;	g[10] = r*r*sinf(p.y)*sinf(p.y);	g[11] = 0.0f;
	g[12] = 0.0f;					g[13] = 0.0f;	g[14] = 0.0f;						g[15] = -(1.0f-rs/r);
	*/
	g[0] = 1.0f/(1.0f-rs/r);   g[5] = r*r;   g[10] = r*r*sinf(p.y)*sinf(p.y);	g[15] = -(1.0f-rs/r);

	return g;
}


//__device__ f3 getColorDefault(f3 *bimage, f4 p, i2 size) {
//
//	float theta = p.y; float phi = p.z;
//
//	phi = -phi; phi += 2.0f * PI;
//
//	if (phi == 2.0f * PI)
//		phi = 0;
//	if (theta == PI)
//		theta = 0;
//
//	if (fabsf(phi) > 2.0f * PI) {
//		return f3(0.0f, 0.0f, 0.0f);
//	}
//
//
//	int x = (int)round(fmodf(phi / (2.0 * PI) * size.x, size.x)); if (x == size.x) x -= 1;
//	int y = (int)round(fmodf(theta / PI * size.y, size.y));   if (y == size.y) y -= 1;
//
//	if (x < 0 || x >= size.x) {
//		return f3(0.0f, 0.0f, 0.0f);
//	}
//
//	if (y < 0 || y >= size.y) {
//		return f3(0.0f, 0.0f, 0.0f);
//	}
//
//	return f3(bimage[y*size.x + x].x, bimage[y*size.x + x].y, bimage[y*size.x + x].z);
//}





//We store the color of each ray in the 4-velocity to save space (saves 12*width*height bytes, so 25MB for 1080p and 100MB for 4k)
__global__ void rayToRGB(ray *rays, Camera cam, Spacetime space, tracer trace, int samples)
{
	i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
	if ((ij.x >= cam.getRes().x*samples) || (ij.y >= cam.getRes().y)) return; // Avoid out of bound and accessing other memory
	int pid = ij.y*cam.getRes().x*samples + ij.x;

	ray R = rays[pid];
	if (R.result == RESULT::SUCCESS) {
		if (R.r.x <= space.getBhRs()+0.01f) {
			rays[pid].u = f4(0.0f, 0.0f, 0.0f, 0.0f); return;
		}
		if (R.r.x >= trace.sph1.rout) {
			//rays[pid].u = f4(getColorDefault(trace.sph1.bimage, rays[pid].r, trace.sph1.res), 0.0f); return;
			rays[pid].u = f4(trace.sph1.getColorDefault(rays[pid].r), 0.0f); return;

			rays[pid].u = f4(0.0f, 0.0f, 0.5f*R.r.z/3.1416f, 0.0f); return;
		}
	}
	else if (R.result == RESULT::INFINITE_GEO) {
		rays[pid].u = f4(0.0f, 0.5f*R.r.z/3.1416f, 0.0f, 0.0f); return;
	}
	else {
		rays[pid].u = f4(0.0f, 0.0f, 0.0f, 0.0f); return;
	}
	

}

#pragma endregion

