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
	ld r = p.x; ld r2 = r*r; ld a2 = bh.a*bh.a;
	ld sin2 = sin(p.y)*sin(p.y);

	ld Dl = r2 - bh.rs*r + a2;
	ld Sg = r2 + a2*cos(p.y)*cos(p.y);

	ld g00 = (1.0-bh.rs*r/Sg); /* Does NOT contain the metric signature (-+++) */
	ld g03 = bh.rs*r*bh.a*sin2/Sg;
	ld g11 = Sg/Dl;
	ld g22 = Sg;
	ld g33 = (r2 + a2 + bh.rs*r*a2/Sg*sin2)*sin2;

	ld disc = 4.0*g03*g03*v.z*v.z + 4.0*g00 * ( g11*v.x*v.x + g22*v.y*v.y + g33*v.z*v.z );

	ld u0_1 = (2.0*g03*v.z + sqrtl(disc))/(-2.0*g00); ld u0_2 = (2.0*g03*v.z - sqrtl(disc))/(-2.0*g00);
	ld u02 = fmaxl(u0_1, u0_2);
	if (u02 <= 0.0) cout << "ERROR:::getu0::u02 <= 0.0 !" << endl;

	return sqrtl( u02 );
}


/* Expressions for d^2x/dl^2*/
ld4 d2x(blackhole *bh, ld4 *p, ld4 *u) {
	ld4 g = { 0.0,0.0,0.0,0.0 };
	ld rs = bh->rs; ld a = bh->a; ld a2 = a*a;
	ld r = p->x; ld r2 = r*r; ld r3 = r2*r;

	double theta = p->y; if (fabs(theta) < 1e-4) theta = 1e-4; if (theta > M_PI-1e-4) theta = M_PI-1e-4;
	double cos_2 = cos(2*theta); double cos2 = cos(theta)*cos(theta); double sin_2 = sin(2*theta);
	double sin2 = sin(theta)*sin(theta);// if(fabs(sin2) < 1e-8) sin2 = 1e-8;
	double sinth = sin(theta);// if(fabs(theta) < 1e-4){ sinth = 1e-4; sin2 = 1e-8; cos2 = cos(1e-4)*cos(1e-4); cos_2 = cos(2e-4); sin_2 = sin(2e-4);}

	double Sg = r2 +a2*cos2; double Dl = r2 - rs*r + a2;


	g.t = rs/(2*Dl*Sg*Sg)* (-a*(a2*a2-3*a2*r2-6*r2*r2 + a2*(a2-r2)*cos_2)*sin2 *u->x*u->z //
                                   -2*a*a2*r*Dl*sin2*sin_2 *u->y*u->z
                                   +(a2*a2-a2*r2-2*r2*r2) *u->t*u->x //
                                   +a2*(a2+r2)*cos_2 *u->t*u->x //
                                   +2*a2*r*Dl*sin_2 *u->x*u->y);

    g.x = 1/(2*Sg*Sg*Sg)* (1/Dl *Sg*Sg*(-2*a2*r+r2*rs+a2*(2*r-rs)*cos2) *u->x*u->x //ok.
                                  +2*a2*Sg*Sg*sin_2 *u->x*u->y
                                  +2*r*Dl*Sg*Sg *u->y*u->y
                                  +Dl*sin2*(2*r2*r2*r+2*a2*a2*r*cos2*cos2-a2*r2*rs*sin2+cos2*(4*a2*r*r2+a2*a2*rs*sin2)) *u->z*u->z //
                                  -2*a*Dl*rs*(-r2+a2*cos2)*sin2 *u->z*u->t //
                                  +Dl*rs*(-r2+a2*cos2) *u->t*u->t);

    g.y = 1/(Sg*Sg*Sg)* (-a2/Dl*Sg*Sg*sin_2/2 *u->x*u->x
                                    -2*r*Sg*Sg *u->x*u->y //
                                    +a2*Sg*Sg*sin_2/2 *u->y*u->y
                                    +(a2*r2*r2+r2*r2*r2+a2*a2*(a2+r2)*cos2*cos2+2*a2*r*r2*rs*sin2+a2*a2*r*rs*sin2*sin2+2*a2*r*cos2*(a2*r+r*r2+a2*rs*sin2))*sin_2/2 * u->z*u->z
                                    -a*r*(a2+r2)*rs*sin_2 *u->z*u->t
                                    +a2*r*rs*sin_2/2 *u->t*u->t);

    g.z = 1/(4*Sg*Sg)* (-4/Dl*(2*r2*r2*(r-rs)+2*a2*a2*r*cos2*cos2-a2*r2*rs*sin2+a2*cos2*(4*r*r2-2*r2*rs+a2*rs*sin2)) *u->x*u->z //
                                 -(3*a2*a2+8*a2*r2+8*r2*r2+4*a2*r*rs+4*a2*(a2+2*r2-r*rs)*cos_2+a2*a2*cos(4*theta))*cos(theta)/sinth *u->z*u->y
                                 +4*a*rs/Dl*(-r2+a2*cos2) *u->x*u->t //
                                 +8*a*r*rs*cos(theta)/sinth *u->y*u->t);

		 

	return g;
}


//TODO: Implement a more efficient tracing for the Schwarzschild metric using the spherical symmetry

/* Perform the ray tracing using the Schwarzschild metric */
int trace(blackhole bh, ld4 *_p, ld4 *_u, ld dlM, ld rout) {
	ld dl = dlM; 

	ld4 p = *_p; ld4 u = *_u;
	ld rhe_in = 0.5*(bh.rs - sqrtl(bh.rs*bh.rs) - 4.0*bh.a*bh.a);
	ld rhe_out = 0.5*(bh.rs + sqrtl(bh.rs*bh.rs) - 4.0*bh.a*bh.a);

	ld rhe = bh.rs;

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
		if (p2.x <= rhe) {
			_p->t = p2.t; _p->x = p2.x; _p->y = p2.y; _p->z = p2.z;    _u->t = u2.t; _u->x = u2.x; _u->y = u2.y; _u->z = u2.z;
			return 0;
		}

		p3 = p + dl*u/2.0 + dl*dl*k1/4.0;    u3 = u + dl*k2/2.0;
		k3 = d2x(&bh, &p3, &u3);
		if (p3.x <= rhe) {
			_p->t = p3.t; _p->x = p3.x; _p->y = p3.y; _p->z = p3.z;    _u->t = u3.t; _u->x = u3.x; _u->y = u3.y; _u->z = u3.z;
			return 0;
		}

		p4 = p + dl*u + dl*dl*k2/2.0;    u4 = u + dl*k3;
		k4 = d2x(&bh, &p4, &u4);
		if (p4.x <= rhe) {
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


		if (p.x <= rhe || p.x+u.x*dl <= rhe) {
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