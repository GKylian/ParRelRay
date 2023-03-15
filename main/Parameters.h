#pragma once
///
///@file Parameters.h
///@author Kylian G.
///@brief Reads and interpret the .params file containing the black hole parameters, camera trajectory, integration parameters and more.
///@version 0.1
///@date 2023-03-15
///
///@copyright Copyright (c) 2023
///

#include "base.h"

class Parameters
{
public:
    static Parameters &get() { static Parameters p; return p; }
    SOLVER getSolver(std::string name) {
        try { return SOLVER_NAMES.at(name); }
        catch (const std::exception &e) {
            printf("Couldn't find solver ! Switching to euler. Error: %s", e.what()); return SOLVER::EULER; }
    }

    SSAA_DISTR getSSAA(std::string name) {
        try { return SSAA_NAMES.at(name); }
        catch (const std::exception &e) {
            printf("Couldn't find SSAA ! Switching to regular. Error: %s", e.what()); return SSAA_DISTR::REGULAR;
        }
    }

    map2d parameters;


private:
    Parameters() { }


    std::map<std::string, SOLVER> SOLVER_NAMES
    {
        {"euler", SOLVER::EULER},
        {"rk", SOLVER::RUNGE_KUTTA},
        {"rkf", SOLVER::RUNGE_KUTTA_FEHLBERG},
        {"adams", SOLVER::ADAMS},
        {"gragg", SOLVER::GRAGG},
        {"dormand", SOLVER::DORMAND},
        {"taylor", SOLVER::TAYLOR},
    };

    std::map<std::string, SSAA_DISTR> SSAA_NAMES
    {
        {"none", SSAA_DISTR::NONE},
        {"regular", SSAA_DISTR::REGULAR},
        {"random", SSAA_DISTR::RANDOM},
        {"poisson", SSAA_DISTR::POISSON},
        {"jittered", SSAA_DISTR::JITTERED},
        {"rotated", SSAA_DISTR::ROTATED},
    };

};


bool headerExists(std::string header);
bool parExists(std::string header, std::string name);

std::string par(std::string header, std::string name);
int pari(std::string header, std::string name);
float parf(std::string header, std::string name);
float parf(std::string header, std::string name, float defaultValue);