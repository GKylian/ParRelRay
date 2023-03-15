///
///@file prr.cpp
///@author Kylian G.
///@brief Main file. Reads the parameters, initializes and performs the numerical integration, applies any post processing and saves all the images to renders/*.png
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///

#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudamaths/vecs.h"

#include "main/io.h"
#include "main/base.h"
#include "main/Spacetime.h"
#include "main/Camera.h"
#include "main/Parameters.h"

#include "kernels/ssaa.cuh"
#include "kernels/solver.cuh"

#include "cudamaths/Interpolation.cuh"
#include "cudamaths/strings.cuh"

using namespace std;

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error = %d at %s:%d '%s'\n", static_cast<unsigned int>(result), file, line, func);
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// TODO: move so that it's not global variables
Tracer trace;
Spacetime spacetime;
Camera camera;

/// @brief Render an image based on spacetime & camera parameters as well as scene content.
/// @param frame Image number
/// @param rays Rays array
/// @param image The image we're creating
/// @param final_image The final image that will be saved
/// @param states CUDA states (mainly for RNG)
bool renderframe(int frame, Ray *rays, tex2d image, tex2d final_image, curandState_t *states);

#pragma region checks

/// @brief CMD helper
/// @param name Program name
static void show_usage(string name)
{
    cerr << "Usage: " << name << " <option(s)>\n"
         << "Options:\n"
         << "\t-t, --trace\t\tStart the ray tracing with the given file as parameters\n"
         << "\t-h, --help\t\tShow this help message\n"
         << endl
         << endl;
}

/// @brief Parses and verifies the command line input
/// @param argc
/// @param argv
/// @return
string cmdinput(int argc, char *argv[])
{
    string fname = ""; // The .params file path

    if (argc == 0)
    {
        show_usage(argv[0]);
        return "";
    }
    std::vector<std::string> sources;
    std::string destination;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            show_usage(argv[0]);
            return "";
        }
        else if ((arg == "-t") || (arg == "--trace"))
        {
            if (i + 1 < argc)
                destination = argv[i++];
            else
            {
                cerr << "ERROR::: -t and --trace require one argument (the .params file path)." << endl
                     << endl;
                return "";
            }
            fname = argv[i];
        }
        else
        {
            show_usage(argv[0]);
            return "";
        }
    }
    cout << endl;

    if (fname == "")
    {
        cerr << "ERROR::: No .params file was given. You need to specify a problem file when running PRR (e.g. ./prr -t params/sod.h)." << endl
             << endl;
        return "";
    }
    ifstream f(fname);
    if (!f.good())
    {
        cerr << "ERROR::: Could not find/open the specified .params file (" << fname << ")." << endl
             << endl;
        return "";
    }
    f.close();

    return fname;
}

/// @brief Checks that frame parameters are all valid
/// @return The last valid frame
int checkFrame()
{
    for (int i = 0; i < 100000; i++)
    {
        string header = "frame" + to_string(i); // If parameters for frame don't exist -> don't render it
        if (!(parExists(header, "x") && parExists(header, "y") && parExists(header, "z") && parExists(header, "a") && parExists(header, "b")))
        {
            if (headerExists(header))
                printf("\tHeader exists for frame %d, but not all the parameters are specified !", i);
            return i;
        }
    }

    return 0;
}
#pragma endregion



/// @brief Creates & initialized the camera, the numerical integrator and the spacetime (scene)
void createScene()
{

    // ---------- Spacetime ---------- //
    if (pari("spacetime", "hasBlackHole") == 1)
    {
        spacetime.setHasBh(true);
        spacetime.setBhMass(parf("spacetime", "mass"));
        spacetime.setBhSpin(parf("spacetime", "spin"));
        spacetime.setBhCharge(parf("spacetime", "charge"));
    }
    else
    {
        // TODO: implement precomputing the metric
        printf("ERROR: Precomputing the metric from the matter distribution has not been implemented yet !\n");
    }
    int n = 0;
    for (int i = 0; i < 16; i++)
    {
        spacetime.setBhP(i, parf("spacetime", "p" + to_string(i)));
        n += 1;
    }
    printf("\tLoaded the black hole with %d additional parameters !\n", n);

    // ---------- Tracer ---------- //
    trace.solver = Parameters::get().getSolver(par("tracer", "solver"));
    trace.hmin = parf("tracer", "hmin");
    trace.hmax = parf("tracer", "hmax");
    trace.h = parf("tracer", "h");
    trace.eps = parf("tracer", "eps");
    trace.sph1.rout = parf("tracer", "rout1");
    trace.sph2.rout = parf("tracer", "rout2");
    printf("\tLoaded the tracer !\n");

    // ---------- Camera ---------- //
    SSAA_DISTR distr = Parameters::get().getSSAA(par("camera", "SSAA"));
    camera = Camera(parf("camera", "vfov") * M_PI / 180.0f, i2(pari("camera", "width"), pari("camera", "height")), distr, pari("camera", "SSAAsamples"));

    printf("--> Loading sucessful !\n");
}

/// @brief Initialized the scene and camera based on the parameter files
/// @param paramsPath The path of the simulation's .params file
/// @return 
int init(string paramsPath)
{
    Parameters::get();
    printf("1. Reading ray tracing parameters from .params file ...\n");
    readParams(paramsPath, &Parameters::get().parameters);
    printf("Reading trajectory from .params file at %s\n", par("tracer", "traj").c_str());
    readParams(par("tracer", "traj"), &Parameters::get().parameters);
    readParams(par("tracer", "scene"), &Parameters::get().parameters);

    printf("2. Checking how many frames need to be rendered...\n");
    int frames = checkFrame();
    if (frames == 0)
    {
        printf("There are no frames to be rendered in the .params file !\n");
        return -1;
    }

    printf("\tDetected %d frames to be rendered in the .params file. They will be exported to %simage[i].ppm.\n", frames, par("tracer", "out").c_str());

    printf("3. Loading the scene with the parameters provided in the .params file...\n");
    createScene();
    spacetime.setupScene();

    return frames;
}


/// @brief Render a frame
/// @param frame The index of the frame we're rendering
/// @param rays The rays array
/// @param image The image we're creating
/// @param final_image The final image
/// @param states CUDA states (mainly for RNG)
/// @return Has frame been successfully rendered and written to file ?
bool renderframe(int frame, Ray *rays, tex2d image, tex2d final_image, curandState_t *states)
{
    i2 res = camera.getRes();
    int samples = camera.getAAsamples(); // Treat this as a (samples*res.x, res.y) image
    std::string header = "frame" + to_string(frame);

    //!========== FRAME INIT ==========
    printf("0. Getting position and orientation for frame.\n");
    camera.pos = f3(parf(header, "x"), parf(header, "y"), parf(header, "z"));
    camera.a = parf(header, "a") * M_PI / 180.0f;
    camera.b = parf(header, "b") * M_PI / 180.0f;
    camera.computeVectors();
    spacetime.updateScene(frame);
#ifdef SPHERICAL
    camera.spos = pos_cartTOspher(camera.pos);
#endif                                                                // SPHERICAL
    spacetime.setBhMass(parf(header, "mass", spacetime.getBhMass())); // Change black hole's mass if specified.

    //!========== RAY CREATION ==========
    printf("1. Creating all the pixels with the chosen SSAA distribution (if any).\n");

    dim3 blocks(res.x / 8 + 1, res.y / 8 + 1);
    dim3 threads(8, 8);

    switch (camera.getSSAADistr())
    {
    case SSAA_DISTR::NONE:
        printf("\tLaunching kernel with no SSAA...\n");
        createRays_none<<<blocks, threads>>>(rays, res);
        break;
    case SSAA_DISTR::REGULAR:
        printf("\tLaunching kernel with regular sampling distribution...\n");
        createRays_regular<<<blocks, threads>>>(rays, res, samples);
        break;
    default:
        break;
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("-->All rays have been given a screen position !\n");

    blocks = dim3(samples * res.x / 8 + 1, res.y / 8 + 1);
    threads = dim3(8, 8);
    printf("2. Giving every ray its initial 4-position and 4-momentum !\n");
    initRays<<<blocks, threads>>>(rays, camera, spacetime, samples);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("-->All rays have been initialized !\n");

    //!========== RAY TRACING ==========
    printf("3. Starting to ray trace the image !\n");
    int nlines = 64; // Number of lines per block
    int nlayers = ceil((double)res.y / nlines);
    blocks = dim3(samples * res.x / 8 + 1, nlines / 8 + 1);
    threads = dim3(8, 8);
    for (int l = 0; l < nlayers; l++)
    {
        switch (trace.solver)
        {
        case SOLVER::EULER:
            trace_Euler<<<blocks, threads>>>(rays, camera, spacetime, trace, l, nlines, samples);
            break;
        case SOLVER::RUNGE_KUTTA_FEHLBERG:
            trace_RKF<<<blocks, threads>>>(rays, camera, spacetime, trace, l, nlines, samples);
            break;
        default:
            break;
        }
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\tLayer %d / %d (%d - %d) has been ray traced !\n", l + 1, nlayers, l * nlines, (l + 1) * nlines - 1);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("--> Ray tracing done !\n");

    //!========== POST PROCESSING ==========
    printf("4. Converting rays into RGB float colors !\n");
    blocks = dim3(samples * res.x / 8 + 1, res.y / 8 + 1);
    threads = dim3(8, 8);
    rayToRGB<<<blocks, threads>>>(rays, camera, spacetime, trace, samples);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("--> All rays have been converted to a color !\n");

    blocks = dim3(res.x / 8 + 1, res.y / 8 + 1);
    threads = dim3(8, 8);
    printf("5. Combining samples into the final image !\n");
    combineSamples<<<blocks, threads>>>(rays, image, res, samples);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("--> All samples have been combined into the final image !\n");

    //!========== WRITE RENDER TO FILE ==========
    printf("7. Saving the final image to a .png file !\n");
    image.save(par("tracer", "out") + "image_" + to_string(frame) + ".png");

    printf("\n\n\n");

    return true;
}



int main(int argc, char *argv[])
{

    checkCudaErrors(cudaDeviceReset());
    string fname = cmdinput(argc, argv);
    if (fname == "")
        return 0;

    //!========== INITIALIZATION ==========
    frames = init(fname);
    if (frames < 0)
        return -1;

    //!========== MEMORY ALLOCATION ==========
    printf("4. Creating and allocating memory for the background image(s), the pixel array, random states and the final image!\n");
    std::string bpath1 = par("tracer", "bimage1");
    std::string bpath2 = par("tracer", "bimage2");
    if (bpath1 != "")
        trace.sph1.tex.load(bpath1);
    if (bpath2 != "")
        trace.sph2.tex.load(bpath2);

    i2 res = camera.getRes();
    size_t raysBufferSize = res.x * res.y * camera.getAAsamples() * sizeof(Ray);
    Ray *rays;
    checkCudaErrors(cudaMallocManaged((void **)&rays, raysBufferSize));
    printf("\tAllocated %f MB for ray array with image resolution (%d, %d) and %d samples per pixel.\n", raysBufferSize / 1e6, res.x, res.y, camera.getAAsamples());

    tex2d image(res);
    tex2d final_image(res);
    printf("\tAllocated 2x%f MB for two output images with resolution (%d, %d).\n", image.getSize() / 1e6, res.x, res.y);

    curandState_t *states;
    checkCudaErrors(cudaMalloc((void **)&states, res.x * res.y * sizeof(curandState)));

    //!========== RAY TRACING ==========
    printf("5. Ray tracing all frames.\n\n\n\n\n");
    for (int i = 0; i < frames; i++)
    {
        printf("---------- Rendering frame %d/%d ----------\n\n", i + 1, frames);
        renderframe(i, rays, image, final_image, states);
    }

    printf("\n\n\n\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n");

    //!========== FREE MEMORY & EXIT ==========
    printf("6. Freeing CUDA memory of images...\n");
    spacetime.unloadScene();
    if (bpath1 != "")
        trace.sph1.tex.unload();
    if (bpath2 != "")
        trace.sph2.tex.unload();
    checkCudaErrors(cudaFree(rays));
    checkCudaErrors(cudaFree(states));
    printf("7. Reseting device(s)...\n");
    checkCudaErrors(cudaDeviceReset());

    printf("\n\n\n\n");
    return 0;
}
