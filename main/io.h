#pragma once
///
///@file io.h
///@author Kylian G.
///@brief Contains utility functions to read parameters and read/write images (.ppm format)
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///


#include "../cudamaths/vecs.h"
#include "base.h"
#include <iostream>
#include <fstream>
#include <map>
#include <string>
using namespace std;

void readParams(string fname, map2d *params) {
	ifstream input(fname);
	int count = 0;

    bool printBlocks = !(fname.find("traj")!=std::string::npos||fname.find("scene")!=std::string::npos);

	string block = ""; /* What block are we in (if any) ? */
	for (string line; getline(input, line);) {
		if (!(line.find('_') != string::npos) && !(line.find('<') != string::npos)) continue; /* Line without any parameter or block name -> skip */
		if (line.find('_') != string::npos) {
			unsigned first = line.find('_');
			unsigned last = line.find_last_of('_');
			block = line.substr(first + 1, last - first - 1);
            if(printBlocks)
			    cout << "\tEntering block named " << block << endl; //Don't print if position
			continue;
		}
		/* Split the line in two (before and after the equal */
		string left = line.substr(0, line.find('='));
		string right = line.substr(line.find('=') + 1, line.length() - 1);

		string key = ""; string val = "";
		if (left != "") {
			unsigned first = left.find('<'); unsigned last = left.find('>');
			key = left.substr(first + 1, last - first - 1);
		}
		if (right != "") {
			unsigned first = right.find('<'); unsigned last = right.find('>');
			val = right.substr(first + 1, last - first - 1);
			count++;
		}



		(*params)[block][key] = val;
	}
    printf("-->Read %d parameters\n", count);

}


/* Get the size of a given .ppm image */
i2 getSize(string dir) {
    ifstream image; image.open(dir);
    if (!image.is_open()) {
        printf("\tCan't open the image at %s.", dir.c_str());
        return i2(0,0);
    }

    string word;
    image >> word;
    image >> word; int w = stoi(word);
    image >> word;

    return i2(w, stoi(word));
}



#define PPMREADBUFLEN 256
typedef unsigned char pixel[3];
bool readBinPPM(string fname, f3 *image) {
    printf("\tReading file at %s.\n", fname.c_str());
    char buf[PPMREADBUFLEN], *t;
    unsigned int w, h, d;
    int r;

    FILE *fp; fp = fopen(fname.c_str(), "rb");
    if (fp == NULL) return false;
    t = fgets(buf, PPMREADBUFLEN, fp);
    if ((t==NULL) || (strncmp(buf, "P6\n", 3) != 0)) return false;
    printf("\t\tP6 header checked.\n");
    do // Go to after comments (there can be comments after the first line)
    {
        t = fgets(buf, PPMREADBUFLEN, fp);
        if (t==NULL) return false;
    } while (strncmp(buf, "#", 1) == 0);
    r = sscanf(buf, "%u %u", &w, &h); // Get width and height
    if (r<2) return false; // Both should have been found
    printf("\t\tImage size checked.\n");

    r = fscanf(fp, "%u", &d);
    if ((r<1) || (d!=255)) return false; //Should have found d, and d should be 255 (-> RGB in [0-255])
    fseek(fp, 1, SEEK_CUR); //Skip one byte, should now be on the new line...

    printf("\t\tReading binary PPM file...\n");
    if (image != NULL) {
        pixel *fb = (pixel *)malloc(w*h*sizeof(pixel)); if (fb == NULL) return false;
        size_t rd = fread(fb, sizeof(pixel), w*h, fp);
        if (rd < w*h) {
            free(fb);
            printf("\t\tExpected %d pixels, but only got %d.\n", w*h, (int)rd);
            return false;
        }
        for (int i = 0; i < w*h; i++) {
            image[i] = f3(fb[i][0]/255.0f, fb[i][1]/255.0f, fb[i][2]/255.0f);
        }
        free(fb);

        return true;
    }

    return false;
}


void writeBinPPM(string fname, f3 *image, i2 size)
{
    FILE *fp = fopen(fname.c_str(), "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", size.x, size.y);
    for(int j = 0; j < size.y; j++)
    for(int i = 0; i < size.x; i++)
    {
        size_t pid = j*size.x+i;
        static unsigned char color[3];
        color[0] = int(255.0f*image[pid].x);   color[1] = int(255.0f*image[pid].y);   color[2] = int(255.0f*image[pid].z);
        (void)fwrite(color, 1, 3, fp);
    }
    (void)fclose(fp);
    
}


