#include "image.cuh"

#include <cuda_runtime.h>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"



bool Images::loadRGBFromFile(std::string path, f3 **image, i2 *res, size_t *size)
{
	stbi_ldr_to_hdr_gamma(1.0f);
	*image = NULL;
	*image = (f3 *)stbi_loadf(path.c_str(), &(*res).x, &(*res).y, NULL, 3); *size = (*res).x*(*res).y*sizeof(f3);


	return true;
}

void Images::freeMemory(f3 *image)
{
	stbi_image_free(image);
}

bool Images::savePNGToFile(std::string path, f3 *image, i2 res)
{
	unsigned char *img = (unsigned char *)malloc(3*res.x*res.y);
	for (int i = 0; i<res.x*res.y; i++) { img[3*i] = roundf(255.0f*image[i].x); img[3*i+1] = roundf(255.0f*image[i].y); img[3*i+2] = roundf(255.0f*image[i].z); }

	stbi_write_png(path.c_str(), res.x, res.y, 3, img, res.x*3); //Last argument = Number of bytes per line ?
	free(img);



	return true;
}