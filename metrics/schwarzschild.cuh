#pragma once
#include "../main/base.h"
#include "../main/Camera.h"
#include "../main/Spacetime.h"
#include "../cudamaths/mats.h"




//Calls from the main code (both CPU and GPU calls):
#pragma region MainCalls


// Called before starting the loop to render all frames.
__host__ void User_INIT(ray *rays, Camera *cam, Spacetime *space)
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

	if (r >= trace.sph1.rout || r <= space.getBhRs()+0.001f) {
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
	f4 du;
	float rs = space.getBhRs(); 
	float r = p.xyz().norm();
	float x2 = p.x*p.x, y2 = p.y*p.y, z2 = p.z*p.z;


	//xTODO: attempting to fix errors in the equations
	//xFIXME: 27/04:21h25:: This should be correct now

	float Phi	=	(r*(x2-2.0f*y2-2.0f*z2)  + rs*x2)	* u.x*u.x	
				+	(r*(-2.0f*x2+y2-2.0f*z2) + rs*y2)	* u.y*u.y	
				+	(r*(-2.0f*x2-2.0f*y2+z2) + rs*z2)	* u.z*u.z	
				+	2.0f*(3.0f*r+rs)*(p.x*p.y*u.x*u.y + p.x*p.z*u.x*u.z + p.y*p.z*u.y*u.z)	
				+	r*r*(rs-r)*u.w*u.w + 2*r*rs*u.w*(p.x*u.x+p.y*u.y+p.z*u.z);
	Phi *= rs/(2.0f*powf(r,6));

	du.x = p.x*Phi;
	du.y = p.y*Phi;
	du.z = p.z*Phi;
	du.w =		(2.0f*r*(-x2+y2+z2) -rs*x2)		*	u.x*u.x
			+	(2.0f*r*(x2-y2+z2)  -rs*y2)		*	u.y*u.y
			+	(2.0f*r*(x2+y2-z2)  -rs*z2)		*	u.z*u.z
			-	2.0f*(4.0f*r+rs)*(p.x*p.y*u.x*u.y + p.x*p.z*u.x*u.z + p.y*p.z*u.y*u.z)
			-	r*r*rs*u.w*u.w - 2.0f*r*(r+rs)*u.w*(p.x*u.x+p.y*u.y+p.z*u.z);
	du.w *= rs/(2.0f*powf(r, 5));


	return du;
}


//Get the metric (give an expression, or get it from SpaceTime).
__device__ mat4 getMetric(ray *Ray, Spacetime space) {
	mat4 g;
	float rs = space.getBhRs();
	f4 p = Ray->r; float ir = 1.0f/p.xyz().norm(); float ir2 = ir*ir; float ir3 = ir2*ir;

	// xx	xy	xz	xt
	// yx	yy	yz	yt
	// zx	zy	zz	zt
	// tx	ty	tz	tt
	
	g[0] =  1.0f+rs*p.x*p.x*ir3;	g[1] =  rs*p.x*p.y*ir3;			g[2] =  rs*p.x*p.z*ir3;			g[3] =  rs*p.x*ir2;
	g[4] =  g[1];					g[5] =  1.0f+rs*p.y*p.y*ir3;	g[6] =  rs*p.y*p.z*ir3;			g[7] =  rs*p.y*ir2;
	g[8] =  g[2];					g[9] =  g[6];					g[10] = 1+rs*p.z*p.z*ir3;		g[11] = rs*p.z*ir2;
	g[12] = g[3];					g[13] = g[7];					g[14] = g[11];					g[15] = -1+rs*ir;
	


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
		if (r <= space.getBhRs()*1.01f) {
			rays[pid].u = f4(0.0f, 0.0f, 0.0f, 0.0f); return;
		}
		if (r >= trace.sph1.rout) {
			//rays[pid].u = f4(0.0f, 0.0f, 1.0f, 0.0f); return;

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