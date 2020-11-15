#include <iostream>
#include <omp.h>
#include <vector>

#include "params.h"
#include "utils.h"

#define KERR //SCHWARZSCHILD

#ifdef SCHWARZSCHILD
    #include "schwarzschild.h"
#endif // SCHWARZSCHILD
#ifdef KERR
    #include "kerr.h"
#endif // KERR


using namespace std;

map2d par;
bool render(int f);




static void show_usage(string name) {
    cerr << "Usage: " << name << " <option(s)>\n"
        << "Options:\n"
        << "\t-t, --trace\t\tStart the ray tracing with the given file as parameters\n"
        << "\t-h, --help\t\tShow this help message\n"
        << endl << endl;
}

string cmdinput(int argc, char *argv[]) {
    string fname = ""; //The .params file path

    if (argc == 0) {
        show_usage(argv[0]);
        return "";
    }
    std::vector<std::string> sources; std::string destination;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return "";
        }
        else if ((arg == "-t") || (arg == "--trace")) {
            if (i+1 < argc)
                destination = argv[i++];
            else {
                cerr << "ERROR::: -t and --trace require one argument (the .params file path)." << endl << endl;
                return "";
            }
            fname = argv[i];
        }
        else {
            show_usage(argv[0]);
            return "";
        }
    }
    cout << endl;

    if (fname == "") {
        cerr << "ERROR::: No .params file was given. You need to specify a problem file when running Asteria (e.g. ./Asteria -p params/sod.h)." << endl << endl;
        return "";
    }
    ifstream f(fname);
    if (!f.good()) {
        cerr << "ERROR::: Could not find/open the specified .params file (" << fname << ")." << endl << endl;
        return "";
    }
    f.close();

    return fname;
}




//Check how many frames have to be rendered (0 to i-1)
int checkFrame() {
    for (int i = 0; i < 100000; i++) {
        string header = "pos"+to_string(i);
        if (par.find(header) != par.end()) {
            map<string, string> p = par[header];
            if (!exists(&p, "x") || !exists(&p, "y") || !exists(&p, "z") || !exists(&p, "a") || !exists(&p, "b"))
                return i;

        }
        else
            return i;
    }

    return 0;
}


int main(int argc, char *argv[]){

    string fname = cmdinput(argc, argv);

    readParams(fname, &par); cout << endl;


    int nbrFrames = checkFrame();
    if (nbrFrames == 0) {
        cout << "ERROR::No frame to be rendered !" << endl;
        return 0;
    }
    cout << "Rendering " << nbrFrames << " frame(s)..." << endl << endl << endl;



    for (int f = 0; f < nbrFrames; f++) {
        if (!render(f)) {
            cout << "ERROR::Could not render frame number " << f << endl;
        }
    }


  return 0;
}






ld4 camRay(int i, int j, int height, int width, ld fov, ld3 forward, ld3 right, ld3 down) {
    
    ld a = tanl(fov/2.0); ld b = tanl(fov*height/(2.0*width));
    ld3 v = { 1.0, 0.0, 0.0};
    //v.x = forward.x + a*(-1.0+2.0*i/width)*right.x   +   b*(-1.0+2.0*j/height)*down.x;
    //v.y = forward.y + a*(-1.0+2.0*i/width)*right.y   +   b*(-1.0+2.0*j/height)*down.y;
    //v.z = forward.z + a*(-1.0+2.0*i/width)*right.z   +   b*(-1.0+2.0*j/height)*down.z;

    v = forward + a*(-1.0+2.0*i/width)*right   +   b*(-1.0+2.0*j/height)*down;
    v.normalize();

    return ld4{0.0, v.x, v.y, v.z};
}




bool render(int f) {
    string header = "pos"+to_string(f);
    cout << "Rendering image number " << f << endl;

    /* ------------------------------------------------------------ */
    /* -------------------- 1. Load parameters -------------------- */
    /* ------------------------------------------------------------ */


    /* Camera parameters */
    int width = stoi(par["camera"]["width"]); int height = stoi(par["camera"]["height"]); int tot = width*height;
    ld fov = stod(par["camera"]["fov"])*M_PI/180.0;
    ld4 pc = {0.0, stod(par[header]["x"]), stod(par[header]["y"]), stod(par[header]["z"]) };
    ld alpha = stod(par[header]["a"]); ld beta = stod(par[header]["b"]);
    cout << "\tResolution: (" << width << ", " << height << ")\n";


    /* Tracing parameters */
    ld dl = stod(par["tracer"]["dl"]); ld rout = stod(par["tracer"]["rout"]);
    string bdir = par["tracer"]["bimage"];
    string dir = par[header]["out"];
    cout << "\tTracing parameters: dlMax = " << dl << ", r_out = " << rout;

    /* Black hole parameters */
    ld M = stod(par["bh"]["M"]); ld a = stod(par["bh"]["a"]);
    blackhole bh; bh.M = M; bh.a = a; bh.rs = 2.0*M;
    cout << "\tBlack hole: M = " << M << ", a = " << a << endl;
    


    /* Image array, every pixel has a corresponding set of three ints (RGB) */
    color *image = new color[width*height];

    int bwidth = 0, bheight = 0; getSize(bdir, &bwidth, &bheight);
    color *bimage = new color[bwidth*bheight];
    getBackground(bdir, bimage);
    

    /* Parameters common to all rays */
    ld4 p = pos_cartTOspher(pc); /* Transform position to spherical coordinates*/
    ld3 forward, right, down;

    forward = { sinl(beta)*cosl(alpha), sinl(beta)*sinl(alpha), cosl(beta) };
    down = { cosl(beta)*cosl(beta), cosl(beta)*sinl(alpha), -sinl(beta) };
    right = cross(down, forward);
    cout << "\tForward = " << forward.str() << "\n\tDown = " << down.str() << "\n\tRight = " << right.str() << endl;





    /* ------------------------------------------------------------ */
    /* -------------------- 2. Run with OpenMP -------------------- */
    /* ------------------------------------------------------------ */
    int done = 0;

#pragma omp parallel for schedule(dynamic)
    for (int pix = 0; pix < tot; pix++) {
        if(pix == 0)
            cout << endl << endl << "Starting ray tracing with " << omp_get_num_threads() << " threads..." << endl << endl;

        int i = pix % width; int j = (int)pix/width;
        ld4 x = p;


        

        /* --- 1. Compute ray direction from camera position --- */
        ld4 u = camRay(i, j, height, width, fov, forward, right, down);
        
        /* --- 2. Transform velocity to spherical coordinates --- */
        u = vel_cartTOspher(pc, u);
        

        /* --- 3. Compute initial ut and get four-velocity --- */
        ld u0 = getu0(bh, x, u); u.scale(u0);
        if (u0 <= 0) cout << "ERROR::u0 <= 0" << endl;
        if (u0 >= 5.0) cout << "ERROR::u0 >= 5: " << u0 << endl;

        
        /* --- 4. Perform the ray tracing --- */
        int result = trace(bh, &x, &u, dl, rout);


        /* --- 5. Get color from result ---- */
        if (result == 0) {
            /* The ray fell into the black hole */
            image[pix] = black;
        }
        else if (result == 1){
            /* The ray escaped to r_out */
            image[pix] = getColor(bimage, x, bwidth, bheight, u);
            
        }
        else {
            /* Error -> pixel green so that it stands out */
            cout << "ERROR:::render:: trace() returned -1" << endl;
            image[pix] = green;
        }
        

        done++;

        int d = done; /* Saving 'done' locally, otherwise another ray changes its value between the first and second if*/
        if (d % ((int)tot/100) == 0 && d != 0) 
            cout << round(100.0*d/tot) << "% done" << endl;
        
        if (d % ((int)tot/10) == 0 && d != 0)
            saveToPPM(dir, image, width, height);
        

    }



    /* ---------- 3. Save image to PPM ---------- */
    saveToPPM(dir, image, width, height);



    return true;
}