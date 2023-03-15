#pragma once
#include "../main/base.h"
#include "../main/Camera.h"
#include "../main/Spacetime.h"
#include "../cudamaths/mats.h"




//Calls from the main code (both CPU and GPU calls):
#pragma region MainCalls


// Called before starting the loop to render all frames.
__host__ void User_INIT(Ray *rays, Camera cam, Spacetime *space)
{

}


// Called before ray initialization. Return false to skip default ray initialization.
__host__ bool User_RAYINIT(ray *rays, Camera cam, Spacetime *space)
{
	return true;
}

// Called before ray tracing. Return false to skip default ray tracing.
__host__ bool User_TRACING(ray *rays, Camera cam, Spacetime *space)
{
	return true;
}

// Called before post-processing (AA, ...). Return false to skip default post-processing.
__host__ bool User_POST(ray *rays, Camera cam, Spacetime *space)
{
	return true;
}

// Called when the image has been fully rendered (just before saving it).
__host__ void User_END(ray *rays, Camera cam, Spacetime *space)
{

}




// Called before the integration loop starts in the solver kernel. Return false to skip the loop.
__device__ bool User_LOOP(ray *iray, Camera cam, Spacetime *space)
{
	return true;
}

// Called before updating the 4-velocity and 4-position in the solver kernel. Return false to skip the update.
__device__ bool User_UPDATE(ray *iray, Camera cam, Spacetime *space)
{
	return true;
}

// Called at the end of the integration iteration in the solver kernel. Return false to stop the integration (stop the solver kernel).
__device__ bool User_ENDLOOP(ray *iray, Camera cam, Spacetime *space)
{
	return true;
}
#pragma endregion


//Required (GPU) functions for the integration: getGeod, getMetric
#pragma region Physics

//Get the geodesics (du/dlambda) to update the ray's 4-velocity (give an expression, or get it from SpaceTime).
__device__ f4 getGeod(ray *r, Spacetime *space) {
	return f4_zero();
}

//Get the metric (give an expression, or get it from SpaceTime).
__device__ mat4 getMetric(ray *Ray, Spacetime *space) {
	mat4 g;
	float rs = space->getBhRs(); float a = space->getBhSpin();
	f4 r = Ray->r; float r2 = r.x*r.x; float sin2 = sinf(r.y)*sinf(r.y);

	float Si = r2 + a*a*cosf(r.y)*cosf(r.y);
	float Dl = r2 - rs*r.x + a*a;


	// rr	rth		rph		rt
	// thr	thth	thph	tht
	// phr	phth	phph	pht
	// tr	tth		tph		tt
	/*
	g[0] = Si/Dl;					g[1] = 0.0f;	g[2] = 0.0f;								g[3] = 0.0f;
	g[4] = 0.0f;					g[5] = Si;		g[6] = 0.0f;								g[7] = 0.0f;
	g[8] = 0.0f;					g[9] = 0.0f;	g[10] = (r2+a*a+rs*r.x*a*a/Si*sin2)*sin2;	g[11] = -rs*r.x*a*sin2/Si;
	g[12] = 0.0f;					g[13] = 0.0f;	g[14] = g[11];								g[15] = -(1.0f-rs*r.x/Si);
	*/
	g[0] = Si/Dl; g[5] = Si; g[10] = (r2+a*a+rs*r.x*a*a/Si*sin2)*sin2; g[15] = -(1.0f-rs*r.x/Si);
	g[11] = -rs*r.x*a*sin2/Si; g[14] = g[11];

	return g;
}



//__device__ 
#pragma endregion

