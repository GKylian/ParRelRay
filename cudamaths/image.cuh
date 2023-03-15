#pragma once

#include <math.h>
#include <stdlib.h>
#include <fstream>

#include "if_interop.cuh"

class Images
{
public:
	static Images &get() { static Images p; return p; }

	bool loadRGBFromFile(std::string path, f3 **image, i2 *resolution, size_t *size);
	void freeMemory(f3 *image);
	bool savePNGToFile(std::string path, f3 *image, i2 res);

private:
	Images() {}
	int loaded = 0;
};


