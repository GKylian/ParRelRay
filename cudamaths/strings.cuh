#pragma once

#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "f4.cuh"
#include <iostream>
#include <string>


__host__ inline std::string str(const f2 &v) { return "("+std::to_string(v.x)+", "+std::to_string(v.y)+")"; }
__host__ inline std::string str(const f3 &v) { return "("+std::to_string(v.x)+", "+std::to_string(v.y)+", "+std::to_string(v.z)+")"; }
__host__ inline std::string str(const f4 &v) { return "("+std::to_string(v.x)+", "+std::to_string(v.y)+", "+std::to_string(v.z)+", "+std::to_string(v.w)+")"; }


