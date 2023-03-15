#pragma once
///
///@file base.h
///@author Kylian G.
///@brief Contains a bunch of enum and struct definitions
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///

#include <iostream>
#include "../cudamaths/vecs.h"
#include <map>
#include <string>

#include "../cudamaths/Interpolation.cuh"
#include "../cudamaths/tex2d.cuh"

#define PI     3.14159265358979323846
#define RAYPAR 1

typedef std::map<std::string, std::map<std::string, std::string>> map2d;

/// @brief Which solver we're using
enum class SOLVER {
    EULER = 0,
    RUNGE_KUTTA = 1,
    RUNGE_KUTTA_FEHLBERG = 2,
    ADAMS = 3,
    GRAGG = 4,
    DORMAND = 5,
    TAYLOR = 6
};

/// @brief The result of the numerical integration
enum class RESULT {
    SUCCESS = 0,
    MATH_ERROR = 1,
    IO_ERROR = 2,
    INFINITE_GEO = 3,
    INTERSECTION = 4
};

/// @brief Whick SSAA sample distribution should be used (if any)
enum class SSAA_DISTR {
    NONE = 0,
    REGULAR = 1,
    RANDOM = 2,
    POISSON = 3,
    JITTERED = 4,
    ROTATED = 5
};

/// @brief The black hole's parameters
struct BlackHole {
    float M; float rs; float a; float q; float p[16];
};

/// @brief The outer sphere the 'sky' is projected on
struct OutSphere {
    tex2d tex;
    float rout;

    /// @brief Get the background color at the correct angles
    /// @param p The final position of the ray
    /// @return The RGB color f3(r,g,b)
    __device__ f3 getColorDefault(f4 p) {
        
        float theta = p.y; float phi = p.z;
#ifndef SPHERICAL
        theta = acosf(p.z/p.xyz().norm());  phi = atan2f(p.y, p.x);
        phi = (phi >= 0 ? phi : phi+2.0f*PI);
#endif // !SPHERICAL

#ifdef SPHERICAL
        phi = -phi; phi += 2.0f * PI;

        if (phi == 2.0f * PI)
            phi = 0;
        if (theta == PI)
            theta = 0;
#endif // !SPHERICAL


        if (phi > 2.0f*PI || phi < 0.0f) {
            return f3(0.5f, 0.5f, 0.0f);
        }
        if (theta < 0.0f || theta > PI) {
            return f3(0.0f, 0.5f, 0.8f);
        }


        //int x = (int)round(fmodf(phi / (2.0 * PI) * res.x, res.x)); if (x == res.x) x -= 1;
        //int y = (int)round(fmodf(theta / PI * res.y, res.y));   if (y == res.y) y -= 1;

        f2 uv = f2(phi/(2.0f*PI)*tex.res.x, theta/PI*tex.res.y);
        i2 pi = floor(uv); if (pi.x ==tex.res.x) pi.x -= 1;   if (pi.y ==tex.res.y) pi.y -= 1; //Location of bottom left pixel (0,0)
        uv = uv - pi; //Local (cell) coordinates
        if (uv.x > 1 || uv.y > 1) printf("uv = (%f, %f), pi = (%d, %d)", uv.x, uv.y, pi.x, pi.y);

        if (pi.x < 0 || pi.x >= tex.res.x) { printf("pi.x; (theta,phi) = (%.2f, %.2f), pi = (%d, %d)\n", theta, phi, pi.x, pi.y); return f3(0.0f, 0.5f, 0.0f); }
        if (pi.y < 0 || pi.y >= tex.res.y) { printf("pi.y; (theta,phi) = (%.2f, %.2f), pi = (%d, %d)\n", theta, phi, pi.x, pi.y); return f3(0.0f, 0.5f, 0.0f); }

        if (pi.x == tex.res.x-1 || pi.y == tex.res.y -1) return tex[pi];

        //Bottom left, top left, bottom right and top right respectively
        f3 c_00 = tex[pi];           f3 c_01 = tex[pi+i2(0, 1)];
        f3 c_10 = tex[pi+i2(1,0)];   f3 c_11 = tex[pi+i2(1, 1)];

        return sqavg(c_00, c_01, c_10, c_11);

        //f3 color = bilinear::bilinear(uv/cinv(res), c_00, c_01, c_10, c_11); //sqavg(c_00, c_01, c_10, c_11)
        //color.normalize(); return color;
        //return clamp(bilinear::bilinear(uv/cinv(res), c_00, c_01, c_10, c_11), f3(0,0,0), f3(1,1,1));

    }

    __device__ f3 getColorDefault2(f4 u) {

#ifdef SPHERICAL
        printf("Spherical coordinates not implemented for getColorDefault2 !!!\n");
        return f3(0.0f, 1.0f, 0.0f);
#endif // SPHERICAL

        f3 n = normalize(u.xyz());
        float theta = acosf(n.z);  float phi = atan2f(n.y, n.x);
        phi = (phi>=0 ? phi : phi+2.0f*PI);


        if (phi>2.0f*PI||phi<0.0f) {
            return f3(0.5f, 0.5f, 0.0f);
        }
        if (theta < 0.0f||theta > PI) {
            return f3(0.0f, 0.5f, 0.8f);
        }


        //int x = (int)round(fmodf(phi / (2.0 * PI) * res.x, res.x)); if (x == res.x) x -= 1;
        //int y = (int)round(fmodf(theta / PI * res.y, res.y));   if (y == res.y) y -= 1;

        f2 uv = f2(phi/(2.0f*PI)*tex.res.x, theta/PI*tex.res.y);
        i2 pi = floor(uv); if (pi.x==tex.res.x) pi.x -= 1;   if (pi.y==tex.res.y) pi.y -= 1; //Location of bottom left pixel (0,0)
        uv = uv-pi; //Local (cell) coordinates
        if (uv.x>1||uv.y>1) printf("uv = (%f, %f), pi = (%d, %d)", uv.x, uv.y, pi.x, pi.y);

        if (pi.x<0||pi.x>=tex.res.x) { printf("pi.x; (theta,phi) = (%.2f, %.2f), pi = (%d, %d)\n", theta, phi, pi.x, pi.y); return f3(0.0f, 0.5f, 0.0f); }
        if (pi.y<0||pi.y>=tex.res.y) { printf("pi.y; (theta,phi) = (%.2f, %.2f), pi = (%d, %d)\n", theta, phi, pi.x, pi.y); return f3(0.0f, 0.5f, 0.0f); }

        if (pi.x==tex.res.x-1||pi.y==tex.res.y-1) return tex[pi];

        //Bottom left, top left, bottom right and top right respectively
        f3 c_00 = tex[pi];           f3 c_01 = tex[pi+i2(0, 1)];
        f3 c_10 = tex[pi+i2(1, 0)];   f3 c_11 = tex[pi+i2(1, 1)];

        return sqavg(c_00, c_01, c_10, c_11);

        //f3 color = bilinear::bilinear(uv/cinv(res), c_00, c_01, c_10, c_11); //sqavg(c_00, c_01, c_10, c_11)
        //color.normalize(); return color;
        //return clamp(bilinear::bilinear(uv/cinv(res), c_00, c_01, c_10, c_11), f3(0,0,0), f3(1,1,1));

    }
};

/// @brief The numerical integrator's 
struct Tracer {
	float h, hmin, hmax, eps;
    OutSphere sph1, sph2;
    SOLVER solver;
};

struct Ray {
    f2 uv; RESULT result;
    f4 r, u;
    float I = 0.0f; //Luminosity (integrated along the path)
}; //2*4 + 4 + 4*4*4 + 4 + RAYPAR*4 = 92 bytes if 16, or 44 bytes if 4





