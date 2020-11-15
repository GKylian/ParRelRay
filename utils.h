#pragma once
#include <map>
#include <string>
#include <iostream>
#include <fstream>

#define RG4




typedef std::map<std::string, std::map<std::string, std::string>> map2d;
typedef long double ld;
using namespace std;


struct blackhole {
    ld M = 1.0; ld a = 0.0; ld rs;
};



/* ---------- 2d vector and its operations ---------- */
typedef struct ld2 {
    ld x;
    ld y;
} ld2;

ld2 operator +(const ld2 &a, const ld2 &b) {    return ld2{ a.x+b.x, a.y+b.y }; }

ld2 operator -(const ld2 &a, const ld2 &b) {    return ld2{ a.x-b.x, a.y-b.y }; }

/* Element-wise multiplication */
ld2 operator *(const ld2 &a, const ld2 &b) {    return ld2{ a.x*b.x, a.y*b.y }; }

ld2 operator /(const ld2 &a, const ld2 &b) {    return ld2{ a.x/b.x, a.y/b.y }; }

ld2 operator *(const ld t, const ld2 &a) {  return ld2{ t*a.x, t*a.y }; }
ld2 operator *(const ld2 &a, const ld t) {  return ld2{ t*a.x, t*a.y }; }

ld2 operator /(const ld2 &a, const ld t) { return ld2{ t/a.x, t/a.y }; }





/* ---------- 3d vector and its operations ---------- */
typedef struct ld3 {
    ld x;
    ld y;
    ld z;

    void normalize() {
        ld r = sqrtl(x*x+y*y+z*z);
        x /= r; y /= r; z /= r;
    }
    string str() {
        return "("+to_string(x)+", "+to_string(y)+", "+to_string(z)+")";
    }
    
} ld3;



ld3 operator +(const ld3 &a, const ld3 &b) { return ld3{ a.x+b.x, a.y+b.y, a.z+b.z}; }

ld3 operator -(const ld3 &a, const ld3 &b) { return ld3{ a.x-b.x, a.y-b.y, a.z-b.z }; }

/* Element-wise multiplication */
ld3 operator *(const ld3 &a, const ld3 &b) { return ld3{ a.x*b.x, a.y*b.y, a.z*b.z }; }

ld3 operator /(const ld3 &a, const ld3 &b) { return ld3{ a.x/b.x, a.y/b.y, a.z/b.z }; }

ld3 operator *(const ld t, const ld3 &a) { return ld3{ t*a.x, t*a.y, t*a.z }; }
ld3 operator *(const ld3 &a, const ld t) { return ld3{ t*a.x, t*a.y, t*a.z }; }

ld3 operator /(const ld3 &a, const ld t) { return ld3{ a.x/t, a.y/t, a.z/t }; }







/* ---------- four-vector and its operations ---------- */
typedef struct ld4 {
    ld t;
    ld x;
    ld y;
    ld z;

    void normalize_xyz() {
        ld r = sqrtl(x*x+y*y+z*z);
        x /= r; y /= r; z /= r;
    }

    void scale(ld a) {
        t *= a; x *= a; y *= a; z *= a;
    }

    string str() {
        return "("+to_string(t)+", "+to_string(x)+", "+to_string(y)+", "+to_string(z)+")";
    }

} ld4;


ld4 operator +(const ld4 &a, const ld4 &b) { return ld4{ a.t+b.t, a.x+b.x, a.y+b.y, a.z+b.z }; }

ld4 operator -(const ld4 &a, const ld4 &b) { return ld4{ a.t-b.t, a.x-b.x, a.y-b.y, a.z-b.z }; }

/* Element-wise multiplication */
ld4 operator *(const ld4 &a, const ld4 &b) { return ld4{ a.t*b.t, a.x*b.x, a.y*b.y, a.z*b.z }; }

ld4 operator /(const ld4 &a, const ld4 &b) { return ld4{ a.t/b.t, a.x/b.x, a.y/b.y, a.z/b.z }; }

ld4 operator *(const ld t, const ld4 &a) { return ld4{ t*a.t, t*a.x, t*a.y, t*a.z }; }
ld4 operator *(const ld4 &a, const ld t) { return ld4{ t*a.t, t*a.x, t*a.y, t*a.z }; }

ld4 operator /(const ld4 &a, const ld t) { return ld4{ a.t/t, a.x/t, a.y/t, a.z/t }; }






typedef struct color {
    int r;
    int g;
    int b;

    string str() {
        return "("+to_string(r)+", "+to_string(g)+", "+to_string(b)+")";
    }
} color;

color black = { 0, 0, 0 };
color white = { 255, 255, 255 };
color red = { 255, 0, 0 };
color green = { 0, 255, 0 };
color blue = { 0, 0, 255 };


ld3 cross(ld3 a, ld3 b) {
    ld3 c = { 0.0, 0.0, 0.0 };
    c.x = a.y*b.z - a.z*b.y;
    c.y = -(a.x*b.z - a.z*b.x);
    c.z = a.x*b.y - a.y*b.x;

    return c;
}



bool exists(map<string, string> *p, string a) {
    return (p->find(a) != p->end());
}


ld4 pos_cartTOspher(ld4 pc) {
    ld4 ps = { pc.t, 0.0, 0.0, 0.0 }; /* t, r, theta, phi*/
    ps.x = sqrtl(pc.x*pc.x + pc.y*pc.y + pc.z*pc.z);
    ps.y = acosl(pc.z/ps.x); /* acos(z/r) */
    ps.z = atan2l(pc.y, pc.x);

    return ps;
}

ld4 vel_cartTOspher(ld4 pc, ld4 vc) {
    ld4 vs = { vc.t, 0.0, 0.0, 0.0 }; /*dt, dr, dtheta, dphi */

    ld x2y2 = pc.x*pc.x + pc.y*pc.y;
    ld r2 = x2y2 + pc.z*pc.z;

    vs.x = (pc.x*vc.x + pc.y*vc.y + pc.z*vc.z) / sqrtl(r2);
    vs.y = (pc.z*(pc.x*vc.x + pc.y*vc.y) - x2y2*vc.z) / (r2*sqrtl(x2y2));
    vs.z = (vc.x*pc.y - pc.x*vc.y) / x2y2;

    return vs;
}


/* Get the color of the background image corresponding to the position phi, theta*/
color getColor(color *bimage, ld4 p, int width, int height, ld4 u) {
    color c = green;
    ld theta = p.y; ld phi = p.z;

    phi = -phi; phi += 2.0*M_PI;

    if (phi == 2.0*M_PI)
        phi = 0;
    if (theta == M_PI)
        theta = 0;

    if (fabsl(phi) > 2.0*M_PI)
        cout << "r: " << p.x << ", phi: " << phi << ", theta: " << theta << endl;


    int x = round(  fmodl(phi/(2.0*M_PI)*width, width)  );
    int y = round(  fmodl(theta/M_PI*height, height)  );

    if (x < 0 || x >= width) {
        cout << "ERROR:::getColor::PIXEL_OUT_OF_RANGE: x=" << x << endl;
        cout << "\tr: " << p.x << ", phi: " << phi << ", theta: " << theta << endl;
        cout << "\tdr: " << u.x << ", dphi: " << u.z << ", dtheta: " << u.y << endl;
        return c;
    }
        
    if (y < 0 || y >= height) {
        cout << "ERROR:::getColor::PIXEL_OUT_OF_RANGE: y=" << y << endl;
        cout << "\tr: " << p.x << ", phi: " << phi << ", theta: " << theta << endl;
        return c;
    }

        

    c.r = bimage[y*width + x].r;
    c.g = bimage[y*width + x].g;
    c.b = bimage[y*width + x].b;
    //cout << "color: " << c.str() << endl;



    return c;
}


/* Get the size of a given .ppm image */
void getSize(string dir, int *width, int *height) {
    ifstream image; image.open(dir);
    if (!image.is_open()) {
        cout << "ERROR:::getSize:: Can't open the image at " << dir << endl;
        return;
    }

    string word;
    image >> word;
    image >> word; *width = stoi(word);
    image >> word; *height = stoi(word);
}



/* Loads the background image (projected onto the r_out sphere) from a .ppm file to an array */
void getBackground(string fname, color* bimage) {
    int w = 0, h = 0;
    getSize(fname, &w, &h);
    cout << "Loading the background image of size (" << w << ", " << h << ")\n";

    ifstream image(fname);
    if (!image.is_open()) {
        cout << "readPPM:: Can't open the image at " << fname << endl;
        return;
    }

    string line;
    getline(image, line); getline(image, line); getline(image, line);

    string p;

    for (int i = 0; i < w*h; i++) {
        image >> p; bimage[i].r = stoi(p);
        image >> p; bimage[i].g = stoi(p);
        image >> p; bimage[i].b = stoi(p);
    }
}



/* Save the image stored as an array of RGB values to a .ppm file*/
void saveToPPM(string fname, color *image, int width, int height) {

    cout << "Saving image to " << fname << endl;

    ofstream img(fname, ofstream::trunc);

    img << "P3" << endl;
    img << width << " " << height << endl;
    img << "255" << endl;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            color c = image[j*width+i];
            img << c.r << " " << c.g << " " << c.b << "  ";
        }
        img << endl;
    }


    img.close();
}