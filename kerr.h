#pragma once
#include "utils.h"

using namespace std;

/*
*
*		Contains all the functions needed for ray tracing in the metric
*		of Kerr (rotating black hole of mass M and spin parameter a)
* 
*/


/* Get the time-component of the four-velocity (not w.r.t proper time because of m=0 for light) */
ld getu0(blackhole bh, ld4 p, ld4 v) {
	ld g00 = (1.0-bh.rs/p.x); /* Does NOT contain the metric signature (-+++) */
	ld g11 = 1.0/g00;
	
	ld g22 = p.x*p.x;
	ld g33 = g22*sinl(p.y)*sinl(p.y);

	ld u02 = (g11*v.x*v.x + g22*v.y*v.y + g33*v.z*v.z)/g00;
	if (u02 <= 0.0) cout << "ERROR:::getu0::u02 <= 0.0" << endl;
	return sqrtl( u02 );
}


/* Expressions for d^2x/dl^2*/
ld4 d2x(blackhole *bh, ld4 *p, ld4 *u) {
	ld4 g = { 0.0,0.0,0.0,0.0 };
	ld rs = bh->rs;
	ld r = p->x;

	ld tanth = tan(p->y); if (fabs(tanth) < 1e-4) tanth = 1e-4;
	ld sinth = sinl(p->y); ld costh = cosl(p->y);
	if (fabsl(sinth) < 1e-4) sinth = 1e-4;

	g.t =	- rs/(r*r-r*rs) * u->x*u->t;
	g.x =	- 0.5*rs*(r-rs)/(r*r*r) * u->t*u->t
			+ 0.5*rs/(r*r-r*rs)*u->x*u->x
			+ (r-rs)*( u->y*u->y + sinth*sinth*u->z*u->z );
	g.y =	- 2.0/r * u->x*u->y + sinth*costh*u->z*u->z;
	g.z =	- 2.0/r * u->x*u->z - 2.0/tanth * u->y*u->z; 
		 

	return g;
}


//TODO: Implement a more efficient tracing for the Schwarzschild metric using the spherical symmetry

/* Perform the ray tracing using the Schwarzschild metric */
int trace(blackhole bh, ld4 *_p, ld4 *_u, ld dlM, ld rout) {
	ld dl = dlM; 

	ld4 p = *_p; ld4 u = *_u;

	/* Four-vectors for RG4 (faster to initialize once outside of loop rather than every iteration (I think ?) */
	ld4 k2, p2, u2;    ld4 k3, p3, u3;    ld4 k4, p4, u4;


	for (ld lambda = 0.0; lambda <= 1000.0; lambda += dl) {
		if (dl == 0.0) {
			cout << "ERROR:::trace::dlambda is null !" << endl;
			return -1;
		}


#ifdef EULER
		ld4 du = d2x(&bh, &p, &u);
		p = p + u*dl; u = u + du*dl;
#endif // EULER


#ifdef RG4
		ld4 k1 = d2x(&bh, &p, &u);

		p2 = p + dl*u/2.0;    u2 = u + dl*k1/2.0;
		k2 = d2x(&bh, &p2, &u2);
		if (p2.x <= bh.rs) {
			_p->t = p2.t; _p->x = p2.x; _p->y = p2.y; _p->z = p2.z;    _u->t = u2.t; _u->x = u2.x; _u->y = u2.y; _u->z = u2.z;
			return 0;
		}

		p3 = p + dl*u/2.0 + dl*dl*k1/4.0;    u3 = u + dl*k2/2.0;
		k3 = d2x(&bh, &p3, &u3);
		if (p3.x <= bh.rs) {
			_p->t = p3.t; _p->x = p3.x; _p->y = p3.y; _p->z = p3.z;    _u->t = u3.t; _u->x = u3.x; _u->y = u3.y; _u->z = u3.z;
			return 0;
		}

		p4 = p + dl*u + dl*dl*k2/2.0;    u4 = u + dl*k3;
		k4 = d2x(&bh, &p4, &u4);
		if (p4.x <= bh.rs) {
			_p->t = p4.t; _p->x = p4.x; _p->y = p4.y; _p->z = p4.z;    _u->t = u4.t; _u->x = u4.x; _u->y = u4.y; _u->z = u4.z;
			return 0;
		}

		p = p + dl*u + dl*dl/6.0 * (k1+k2+k3);
		u = u + dl/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);

#endif // RG4
		


		/* This solved the domain problem at the poles */
		if (p.y >= M_PI || p.y <= 0) {
			u.y = -u.y;
			p.z += M_PI;
		}

		/* Keep phi in the [0; 2pi] range, and theta in [0, pi] */
		p.z = fmod(p.z, 2.0*M_PI); if (p.z < 0) p.z += 2.0*M_PI;
		p.y = fabs(p.y); p.y = fmod(p.y, 2.0*M_PI);
		if (p.y > M_PI && p.y <= 2.0*M_PI) p.y = 2.0*M_PI - p.y;


		if (p.x <= bh.rs || p.x+u.x*dl <= bh.rs) {
			_p->t = p.t; _p->x = p.x; _p->y = p.y; _p->z = p.z;    _u->t = u.t; _u->x = u.x; _u->y = u.y; _u->z = u.z;
			return 0;
		}
		if (p.x >= rout) {
			_p->t = p.t; _p->x = p.x; _p->y = p.y; _p->z = p.z;    _u->t = u.t; _u->x = u.x; _u->y = u.y; _u->z = u.z;
			return 1;
		}




		dl = dlM; /* Reset dlambda */

		/* Smaller dlambda at the poles to get good result with the solution of the domain problem */
		dl = (-fabs(p.y-M_PI_2)+M_PI_2)*dlM;

		/* Limit overall integration step to a fraction of the event horizon of the black hole */
		if (dl*fabsl(u.x) >= 0.1*bh.rs) {
			dl = 0.1*bh.rs/fabsl(u.x);
		}

		/* Limit step so that it can't do a full */

		/* Limit integration step close to the BH to a fraction of its event horizon */
		if (dl*fabsl(u.x) >= 0.01*bh.rs && p.x <= 5.0*bh.rs) {
			dl = 0.01*bh.rs/fabsl(u.x);
		}

		if (dl < dlM/1e4) dl = dlM/1e4;


	}

	_p->t = p.t; _p->x = p.x; _p->y = p.y; _p->z = p.z;    _u->t = u.t; _u->x = u.x; _u->y = u.y; _u->z = u.z;
	return -1;
}