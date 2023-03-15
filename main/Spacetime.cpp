#include "Spacetime.h"
#include "Parameters.h"


void check_cuda(cudaError_t result) {
	if (result) {
		fprintf(stderr, "CUDA error = %d\n", static_cast<unsigned int>(result));
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


void Spacetime::setupScene()
{
	for (int i = 0; i<10000; i++)
		if (!headerExists("sphere"+std::to_string(i))) { m_nbrspheres = i; break; }

	check_cuda(cudaMallocManaged((void **)&m_spheres, m_nbrspheres*sizeof(Sphere)));
	
	for (int i = 0; i<m_nbrspheres; i++) {
		std::string hd = "sphere"+std::to_string(i);
		m_spheres[i] = Sphere(f3(parf(hd, "x0"), parf(hd, "y0"), parf(hd, "z0")), parf(hd, "r0"), par(hd, "tex"));
	}

	printf("\tSetup scene with %d sphere(s) !.\n", m_nbrspheres);
	
}

void Spacetime::unloadScene()
{
	check_cuda(cudaFree(m_spheres));
	
}


void Spacetime::updateScene(int f)
{
	printf("Updating position and size for %d spheres.\n", m_nbrspheres);
	for (int i = 0; i<m_nbrspheres; i++) {
		std::string hd = "sphere"+std::to_string(i); f3 p = m_spheres[i].c; float r = m_spheres[i].r;
		m_spheres[i].update(f3(parf(hd, "x"+std::to_string(f), p.x), parf(hd, "y"+std::to_string(f), p.y), parf(hd, "z"+std::to_string(f), p.z)), parf(hd, "r"+std::to_string(f), r));
	}
}