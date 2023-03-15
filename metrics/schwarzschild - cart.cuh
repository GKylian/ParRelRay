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
	float r = iray->r.xyz().norm();


	if (r >= trace.sph1.rout || r <= space.getBhRs()+0.001f) {
		iray->result = RESULT::SUCCESS;
		return false;
	}
	return true;
}
#pragma endregion


//Required (GPU) functions for the integration: getGeod, getMetric
#pragma region Physics


//Get the Christoffel symbols for x
__device__ void getChristoffels(f4 p, Spacetime space, mat4 *cx, mat4 *cy, mat4 *cz, mat4 *ct)
{
	float rs = space.getBhRs(), rs2 = rs*rs, rs4 = rs2*rs2; float r = p.xyz().norm(); float r2 = r*r, r4 = r2*r2;
	float x2 = p.x*p.x, y2 = p.y*p.y, z2 = p.z*p.z;	float x4 = x2*x2, y4 = y2*y2, z4 = z2*z2;
	float xy = x2+y2, xz = x2+z2, yz = y2+z2;

	float gamma = 2.0f*r4*r2*powf(r-rs, 4);
	float alpha = -rs*p.x*(-2*rs4*r+9*rs2*rs*r2-15*rs2*r2*r+11*rs*r4-3*r4*r)/gamma;

	//MATHEMATICA GENERATED
	//Diagonal, then upper triangle
	cx->e[0]	= -rs*p.x*(2*rs4*yz*r+rs*r4*(3*x2-8*yz)-r4*r*(x2-2*yz)+rs2*rs*(x4-7*x2*yz-8*yz*yz)+3*rs2*r*(-x4+3*x2*yz+4*yz*yz))/gamma;
	cx->e[5]	= -rs*p.x*(2*rs4*xz*r+r4*r*(2*x2-y2+2*z2)-rs*r4*(8*x2-3*y2+8*z2)+3*rs2*r*(4*x4-y4+3*y2*z2+4*z4+x2*(3*y2+8*z2))-rs2*rs*(8*x4-y4+7*y2*z2+8*z4+x2*(7*y2+16*z2)))/gamma;
	cx->e[10]	= -rs*p.x*(2*rs4*xy*r-rs*(8*x2+8*y2-3*z2)*r4+(2*x2+2*y2-z2)*r4*r+3*rs2*r*(4*x4+4*y4+3*y2*z2-z4+x2*(8*y2+3*z2))-rs2*rs*(8*x4+8*y4+7*y2*z2-z4+x2*(16*y2+7*z2)))/gamma;
	cx->e[15]	= -rs*p.x*(-rs2*rs+3*rs2*r2-3*rs*r2*r+r4)/(2*r4*r*sq(r-rs));
	cx->e[1]	= p.x*p.y*alpha; cx->e[2]	= p.x*p.z*alpha; cx->e[6]	= p.y*p.z*alpha;
	cx->sym();

	gamma = 2.0f*r4*r*powf(r-rs, 3);
	alpha = rs*p.y*(-2*rs2*rs+7*rs2*r-8*rs*r2+3*r2*r)/gamma;
	cy->e[0]	= rs*p.y*(2*rs2*rs*yz+rs2*r*(x2-6*yz)+r*(x4-x2*yz-2*yz*yz)+rs*(-2*x4+4*x2*yz+6*yz*yz))/gamma;
	cy->e[5]	= -rs*p.y*(-2*rs2*rs*xz*r+r4*(2*x2-y2+2*z2)-2*rs*r*(3*x4-y4+2*y2*z2+3*z4+2*x2*(y2+3*z2))+rs2*(6*x4-y4+5*y2*z2+6*z4+x2*(5*y2+12*z2)))/gamma;
	cy->e[10]	= rs*p.y*(2*rs2*rs*xy+rs2*(-6*x2-6*y2+z2)*r+2*rs*(3*x4+3*y4+2*y2*z2-z4+2*x2*(3*y2+z2))-r*(2*x4+2*y4+y2*z2-z4+x2*(4*y2+z2)))/gamma;
	cy->e[15]	= -rs*p.y*(rs2*r-2*rs*r2+r2*r)/(2*r4*r*(r-rs));
	cy->e[1]	= p.x*p.y*alpha; cy->e[2] = p.x*p.z*alpha; cy->e[6] = p.y*p.z*alpha;
	cy->sym();

	gamma = 2.0f*r4*r2*powf(r-rs, 4);
	alpha = -rs*p.x*(-2*rs4*r+9*rs2*rs*r2-15*rs2*r2*r+11*rs*r4-3*r4*r)/gamma;
	cz->e[0] = rs*p.z*(2*rs4*yz*r+rs*r4*(3*x2-8*yz)-r4*r*(x2-2*yz)+rs2*r*(x4-7*x2*yz-8*yz*yz)+3*rs2*r*(-x4+3*x2*yz+4*yz*yz))/gamma;
	cz->e[5] = -rs*p.z*(2*rs4*xz*r+r4*r*(2*x2-y2+2*z2)-rs*r4*(8*x2-3*y2+8*z2)+3*rs2*r*(4*x4-y4+3*y2*z2+4*z4+x2*(3*y2+8*z2))-rs2*rs*(8*x4-y4+7*y2*z2+8*z4+x2*(7*y2+16*z2)))/gamma;
	cz->e[10] = -rs*p.z*(2*rs4*xy*r-rs*(8*x2+8*y2-3*z2)*r4+(2*x2+2*y2-z2)*r4*r+3*rs2*r*(4*x4+4*y4+3*y2*z2-z4+x2*(8*y2+3*z2))-rs2*rs*(8*x4+8*y4+7*y2*z2-z4+x2*(16*y2+7*z2)))/gamma;
	cz->e[15] = -rs*p.z*(-rs2*rs*r+3*rs2*r2-3*rs*r2*r+r4)/(2.0f*r4*r*sq(r-rs));
	cz->e[1] = p.x*p.y*alpha; cz->e[2] = p.x*p.z*alpha; cz->e[6] = p.y*p.z*alpha;
	cz->sym();

	alpha = -rs/(2.0f*r2*(r-rs));
	ct->e[3] = p.x*alpha; ct->e[7] = p.y*alpha; ct->e[11] = p.z*alpha;
	ct->sym();

}



//Get the geodesics (du/dlambda) to update the ray's 4-velocity (give an expression, or get it from SpaceTime).
//__device__ f4 getGeod(f4 p, f4 u, Spacetime space) {
//	f4 du;
//	float rs = space.getBhRs(); 
//	float r = p.xyz().norm();
//
//	float Phi	=	(r*(p.x*p.x-2.0f*p.y*p.y-2.0f*p.z*p.z)  - rs*p.x*p.x)	* u.x*u.x
//				+	(r*(-2.0f*p.x*p.x+p.y*p.y-2.0f*p.z*p.z) - rs*p.y*p.y)	* u.y*u.y
//				+	(r*(-2.0f*p.x*p.x-2.0f*p.y*p.y+p.z*p.z) - rs*p.z*p.z)	* u.z*u.z
//				+	2.0f*(3.0f*r-rs)*(p.x*p.y*u.x*u.y + p.x*p.z*u.x*u.z + p.y*p.z*u.y*u.z)
//				+	r*r*(rs-1)*u.w*u.w + 2*r*rs*u.w*dot(p.xyz(), u.xyz());
//	Phi *= rs/(2.0f*powf(r,6));
//
//	du.x = p.x*Phi;
//	du.y = p.y*Phi;
//	du.z = p.z*Phi;
//	du.w =		(2.0f*r*(-p.x*p.x+p.y*p.y+p.z*p.z) -rs*p.x*p.x)		*	u.x*u.x
//			+	(2.0f*r*(p.x*p.x-p.y*p.y+p.z*p.z)  -rs*p.y*p.y)		*	u.y*u.y
//			+	(2.0f*r*(p.x*p.x+p.y*p.y-p.z*p.z)  -rs*p.z*p.z)		*	u.z*u.z
//			-	2.0f*(4.0f*r+rs)*(p.x*p.y*u.x*u.y + p.x*p.z*u.x*u.z + p.y*p.z*u.y*u.z)
//			-	r*r*rs*u.w*u.w - 2.0f*r*(r+rs)*u.w*dot(p.xyz(), u.xyz());
//	du.w *= rs/(2.0f*powf(r, 5));
//
//
//	return du;
//}

__device__ f4 getGeod(f4 p, f4 u, Spacetime space) {
	float rs = space.getBhRs();
	float r = p.xyz().norm();

	mat4 cx, cy, cz, ct; getChristoffels(p, space, &cx, &cy, &cz, &ct);
	mat4 us = mat4_comb(u);

	return f4(dot(cx, us), dot(cy, us), dot(cz, us), dot(ct, us));
}


//Get the metric (give an expression, or get it from SpaceTime).
__device__ mat4 getMetric(ray *Ray, Spacetime space) {
	mat4 g;
	float rs = space.getBhRs();
	f4 p = Ray->r; float r = p.xyz().norm(), r2 = r*r;
	float x2 = p.x*p.x, y2 = p.y*p.y, z2 = p.z*p.z;

	// xx	xy	xz	xt
	// yx	yy	yz	yt
	// zx	zy	zz	zt
	// tx	ty	tz	tt
	float alpha = rs/(r2*(r-rs));
	
	/*g[0] =  1.0f+rs*p.x*p.x*ir3;	g[1] =  rs*p.x*p.y*ir3;			g[2] =  rs*p.x*p.z*ir3;	g[3] =  rs*p.x*ir2;
	g[4] =  g[1];					g[5] =  1.0f+rs*p.y*p.y*ir3;	g[6] =  rs*p.y*p.z*ir3;	g[7] =  rs*p.y*ir2;
	g[8] =  g[2];					g[9] =  g[6];					g[10] = 1+rs*p.z*p.z*ir3;	g[11] = rs*p.z*ir2;
	g[12] = g[3];					g[13] = g[7];					g[14] = g[11];				g[15] = -1+rs*ir;*/
	g[0] = -(x2*x2+x2*(y2+z2)-(y2+z2)*r*(r-rs))/(r2*r*(r-rs));		g[1] = -p.x*p.y*alpha;				g[2] = -p.x*p.z*alpha;
																	g[5] = (rs*(x2+z2)-r2*r)*alpha/rs;	g[6] = -p.y*p.z*alpha;
																										g[10] = ((x2+y2)+z2*r/(r-rs))/r2;
																																			g[15] = -1+rs/r;
	g.sym();

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
		if (r <= space.getBhRs()+0.01f) {
			rays[pid].u = f4(0.0f, 0.0f, 0.0f, 0.0f); return;
		}
		if (r >= trace.sph1.rout) {
			//rays[pid].u = f4(getColorDefault(trace.sph1.bimage, rays[pid].r, trace.sph1.res), 0.0f); return;
			rays[pid].u = f4(trace.sph1.getColorDefault(rays[pid].r), 0.0f); return;

			rays[pid].u = f4(0.0f, 0.0f, 0.5f*R.r.z/3.1416f, 0.0f); return;
		}
	}
	else if (R.result == RESULT::INFINITE_GEO) {
		rays[pid].u = f4(0.0f, 0.5f*R.r.z/3.1416f, 0.0f, 0.0f); return;
	}
	else {
		rays[pid].u = f4(0.0f, 1.0f, 0.0f, 0.0f); return;
	}


}