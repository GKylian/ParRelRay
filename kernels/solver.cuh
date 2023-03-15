#pragma once
///
///@file solver.cuh
///@author Kylian G.
///@brief Initializes the rays and solves the geodesics equation with numerical integration using Euler's method (2nd order) or Runge-Kutta-Fehlberg (4th order with adaptive step size)
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///

#include <device_launch_parameters.h>

#include "../main/base.h"
#include "../main/Camera.h"
#include "../main/Spacetime.h"
#include "../cudamaths/mats.h"
#include "../main/metric.h"


/// @brief Initializes the rays' position and velocity based on the camera and black hole.
/// @param rays The ray array
/// @param cam The camera
/// @param space The spacetime containing the black hole parameters and plasma data
/// @param samples How many samples per pixel
__global__ void initRays(ray *rays, Camera cam, Spacetime space, int samples)
{
	i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
    if ((ij.x >= cam.getRes().x*samples) || (ij.y >= cam.getRes().y)) return; // Avoid out of bound and accessing other memory
    int pid = ij.y*cam.getRes().x*samples + ij.x;


	
#ifdef SPHERICAL
	u = f4(vel_cartTOspher(cam.pos, cam.rayDir_PinHole(rays[pid].uv)), 1.0f);
	rays[pid].r = f4(cam.spos, 0);

#else
	f4 u = f4(cam.rayDir_Spherical(rays[pid].uv), 1.0f);
	rays[pid].r = f4(cam.pos, 0);
#endif // SPHERICAL

	mat4 g = getMetric(&rays[pid], space);

	//tex:We have a polynomial of 2nd order $ax^2+bx+c$, with $a = g_{tt}$, $b=-2(g_{tr}\dot{r} + g_{t\theta}\dot{\theta}+g_{t\phi}\dot{\phi})$ and
	// $c = g_{rr}\dot{r}^2 + g_{\theta\theta}\dot{\theta}^2+g_{\phi\phi}\dot{\phi}^2+2(g_{r\theta}\dot{r}\dot{\theta} + g_{r\phi}\dot{r}\dot{\phi}+g_{\theta\phi}\dot{\theta}\dot{\phi})$

	float a = g.m(3, 3);  float b = 2.0f*(g.m(3, 0)*u.x + g.m(3, 1)*u.y + g.m(3, 2)*u.z);
	float c = g.m(0, 0)*u.x*u.x + g.m(1, 1)*u.y*u.y + g.m(2, 2)*u.z*u.z + 2.0f*(g.m(0, 1)*u.x*u.y + g.m(0, 2)*u.x*u.z + g.m(1, 2)*u.y*u.z);
	float rtdisc = sqrtf(b*b-4.0f*a*c);

	float u0 = fmaxf((-b+rtdisc)/(2.0f*a), (-b-rtdisc)/(2.0f*a));
	if (u0 < 0.0f) u0 = 1.0f;
	u *= u0;
	rays[pid].u = u;
}



/// @brief Performs ray tracing by numerically integrating the geodesics equation using Euler's method (2nd order)
/// @param rays The rays array
/// @param cam The camera
/// @param space The spacetime containing the black hole parameters and plasma data
/// @param trace Contains the numerical integrator's parameters
/// @param layer For images done in multiple batches, the current layer
/// @param nlines 
/// @param samples How many samples per pixel
__global__ void trace_Euler(ray *rays, Camera cam, Spacetime space, tracer trace, int layer, int nlines, int samples)
{
	//Treat it as a (samples*res.x, res.y) image
	i2 res = cam.getRes();
	i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y + layer*nlines);
	//Should be between (0,layer*nlines) and (samples*res.x, min(res.y, (layer+1)*nlines))
	// -> we need to stop it before the next layer (not just before the end of the image)
	if (ij.x >= samples*res.x || ij.y >= res.y || ij.y >= (layer+1)*nlines) return;
	int pid = ij.y*samples*res.x + ij.x;

	bool intersect = false;
	float h = trace.h;
	ray R = rays[pid]; f4 x, u;
	if (User_LOOP(&R, cam, space))
	{
		int i = 0;
		for (float l = 0.0f; l < 5000.0f; l += h)
		{

			f4 du = getGeod(R.r, R.u, space);
			if (isnan(du)) {
				R.result = RESULT::MATH_ERROR;
				rays[pid] = R; return;
			}

			x = R.r; u = R.u;
			R.r += R.u*h; R.u += du*h;

			if (User_UPDATE(&R, cam, space, h)) {
				
			}
			else {
				if (R.result==RESULT::INTERSECTION && !intersect) {
					intersect = true; h = trace.hmin; continue;
					R.r = x; R.u = u;
				}
			}

			if (!User_ENDLOOP(&R, cam, space, trace)) {
				/*printf("Stop");*/
				rays[pid] = R;
				return;
			}
			i++;
		}
	}
	R.result = RESULT::INFINITE_GEO;
	rays[pid] = R;
}







//Weights for RKF
__device__ const float A21 = 1.0/4.0;
__device__ const float A31 = 3.0/32.0;			__device__ const float A32 = 9.0/32.0;
__device__ const float A41 = 1932.0/2197.0;	__device__ const float A42 = -7200.0/2197.0;	__device__ const float A43 = 7296.0/2197.0;
__device__ const float A51 = 439.0/216.0;		__device__ const float A52 = -8.0;				__device__ const float A53 = 3680.0/513.0;		__device__ const float A54 = -845.0/4104.0;
__device__ const float A61 = -8.0/27.0;		__device__ const float A62 = 2.0;				__device__ const float A63 = -3544.0/2565.0;	__device__ const float A64 = 1859.0/4104.0;	__device__ const float A65 = -11.0/40.0;
__device__ const float K41 = 25.0/216.0;	__device__ const float K42 = 0.0;	__device__ const float K43 = 1408.0/2565.0;	__device__ const float K44 = 2197.0/4104.0;	__device__ const float K45 = -1.0/5.0;		__device__ const float K46 = 0;
__device__ const float K51 = 16.0/135.0;	__device__ const float K52 = 0.0;	__device__ const float K53 = 6656.0/12825.0;	__device__ const float K54 = 28561.0/56430.0;	__device__ const float K55 = -9.0/50.0;	__device__ const float K56 = 2.0/55.0;


/// @brief Performs ray tracing by numerically integrating the geodesics equation using Runge-Kutta-Fehlberg (4th order with adaptive step size)
/// @param rays The rays array
/// @param cam The camera
/// @param space The spacetime containing the black hole parameters and plasma data
/// @param trace Contains the numerical integrator's parameters
/// @param layer For images done in multiple batches, the current layer
/// @param nlines 
/// @param samples How many samples per pixel
__global__ void trace_RKF(Ray *rays, Camera cam, Spacetime space, Tracer trace, int layer, int nlines, int samples)
{
	//Treat it as a (samples*res.x, res.y) image
	i2 res = cam.getRes();
	i2 ij = i2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y + layer*nlines);
	if (ij.x >= samples*res.x || ij.y >= res.y || ij.y >= (layer+1)*nlines) return;
	int pid = ij.y*samples*res.x + ij.x;


	float h = trace.h, hmin = trace.hmin, hmax = trace.hmax, _h = h;
	float eps = trace.eps;
	Ray R = rays[pid]; f4 x = R.r; f4 u = R.u; f4 _x = x, _u = u, __x, __u;
	if (User_LOOP(&R, cam, space))
	{
		bool intersect = false;
		int i = 0; int attempts = 0;
		for (float l = 0.0f; l < 1000.0f;)
		{
			i += 1; attempts += 1;
			//c.f. wikipedia and https://math.berkeley.edu/~mgu/MA128AFall2017/MA128ALectureWeek11.pdf
			//Solves the system of PDEs (x'=u; u'=getGeod(x, u))
			f4 kx1 = h*fx(x, u);		f4 ku1 = h*getGeod(x, u, space);
			f4 kx2 = h*fx(x+A21*kx1, u+A21*kx1);		f4 ku2 = h*getGeod(x+A21*kx1, u+A21*kx1, space);
			f4 kx3 = h*fx(x+A31*kx1+A32*kx2, u+A31*kx1+A32*ku2);		f4 ku3 = h*getGeod(x+A31*kx1+A32*kx2, u+A31*kx1+A32*ku2, space);
			f4 kx4 = h*fx(x+A41*kx1+A42*kx2+A43*kx3, u+A41*kx1+A42*ku2+A43*ku3);		f4 ku4 = h*getGeod(x+A41*kx1+A42*kx2+A43*kx3, u+A41*kx1+A42*ku2+A43*ku3, space);
			f4 kx5 = h*fx(x+A51*kx1+A52*kx2+A53*kx3+A54*kx4, u+A51*kx1+A52*ku2+A53*ku3+A54*ku4);		f4 ku5 = h*getGeod(x+A51*kx1+A52*kx2+A53*kx3+A54*kx4, u+A51*kx1+A52*ku2+A53*ku3+A54*ku4, space);
			f4 kx6 = h*fx(x+A61*kx1+A62*kx2+A63*kx3+A64*kx4+A65*kx5, u+A61*kx1+A62*ku2+A63*ku3+A64*ku4+A65*ku5);		f4 ku6 = h*getGeod(x+A61*kx1+A62*kx2+A63*kx3+A64*kx4+A65*kx5, u+A61*kx1+A62*ku2+A63*ku3+A64*ku4+A65*ku5, space);

			_x = x + K41*kx1+K42*kx2+K43*kx3+K44*kx4+K45*kx5+K46*kx6;		_u = u + K41*ku1+K42*ku2+K43*ku3+K44*ku4+K45*ku5+K46*ku6;
			__x = x + K51*kx1+K52*kx2+K53*kx3+K54*kx4+K55*kx5+K56*kx6;		__u = u + K51*ku1+K52*ku2+K53*ku3+K54*ku4+K55*ku5+K56*ku6;
			
			//float q = powf(max(abs(eps*h*cinv(2.0f*(__x-_x)))),0.25f); //Maximum error across
			float maxerr = max(max(abs(__x-_x)),max(abs(__u-_u)));
			_h = 0.9f*h*powf(eps/maxerr, 0.2f); //Minimum step size (maximum error on x)
			_h = fminf(fmaxf(_h, hmin), hmax);


			if (intersect&&i>10000) intersect = false;

			if (i > 1000000) {
				printf("Took too many iterations... maxerr = %f, h = %f, _h = %f\n", maxerr, h, _h);
				break;
			}
			if (maxerr > eps && h != hmin && !intersect) { //If error too big AND not already at the smallest step size possible (otherwise infinite loop), solve again with smaller step
				h = _h;
				continue;
			}
			
			if (isnan(x) || isnan(u)) {
				R.result = RESULT::MATH_ERROR;
				rays[pid] = R; return;
			}

			R.r = _x; R.u = _u;

			if (User_UPDATE(&R, cam, space, h)) {
				if(!intersect) h = _h;
				x = _x; u = _u; R.r = x; R.u = u; trace.h = h;
			}
			else {
				if (R.result==RESULT::INTERSECTION && !intersect) {
					/*printf("Intersection mode\n");*/
					intersect = true; h = hmin; continue;
					R.r = x; R.u = u; trace.h = h;
				}
			}

			if (!User_ENDLOOP(&R, cam, space, trace)) {
				rays[pid] = R;
				return;
			} x = R.r; u = R.u;
			

			l += h; attempts = 0;
		}
	}
	R.result = RESULT::INFINITE_GEO;
	rays[pid] = R;
}