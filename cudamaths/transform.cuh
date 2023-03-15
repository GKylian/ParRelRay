#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "f4.cuh"
#include "mat4.cuh"



// -----------------------------------------
// ---------- CARTESIAN-SPHERICAL ----------
// -----------------------------------------

__device__ __host__ inline f3 pos_cartTOspher(f3 pc) {
    float r = pc.norm();
    return f3(r, acosf(pc.z/r), atan2f(pc.y, pc.x));
}

__device__ __host__ inline f3 pos_spherTOcart(f3 ps) {
    return f3(ps.x*cosf(ps.z)*sinf(ps.y), ps.x*sinf(ps.z)*sinf(ps.y), ps.x*cosf(ps.y));
}


__device__ __host__ inline f3 vel_cartTOspher(f3 pc, f3 vc) {
    f3 vs; float x2y2 = pc.xy().norm2();   float r2 = x2y2 + pc.z*pc.z;
    vs.x = dot(pc, vc)/sqrtf(r2);
    vs.y = (pc.z*(pc.x*vc.x+pc.y*vc.y) - x2y2*vc.z) / (r2*sqrtf(x2y2));
    vs.z = (vc.x*pc.y-pc.x*vc.y)/x2y2;
    return vs;
}


__device__ __host__ inline f4 fourvec_spherTOcart(f4 p, f4 u)
{
    mat4 d;
    d[0] = sinf(p.y)*cosf(p.z);     d[1] = p.x*cosf(p.y)*cosf(p.z);     d[2] = -p.x*sinf(p.y)*sinf(p.z);
    d[4] = sinf(p.y)*sinf(p.z);     d[5] = p.x*cosf(p.y)*sinf(p.z);     d[6] = p.x*sinf(p.y)*cosf(p.z);
    d[8] = cosf(p.y);               d[9] = -p.x*sinf(p.y);              d[10] = 0.0;

    f4 v;
    for (int i = 0; i<4; i++) v.x += d[0*4+i];  for (int i = 0; i<4; i++) v.y += d[1*4+i];
    for (int i = 0; i<4; i++) v.z += d[2*4+i];  for (int i = 0; i<4; i++) v.w += d[3*4+i];
    return v;
}