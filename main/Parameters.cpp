#include "Parameters.h"







bool headerExists(std::string header) {
    return Parameters::get().parameters.find(header) != Parameters::get().parameters.end();
}


bool parExists(std::string header, std::string name) {
    if (!headerExists(header)) return false;
    return Parameters::get().parameters[header].find(name) != Parameters::get().parameters[header].end();
}


std::string par(std::string header, std::string name) {
    try { if (!parExists(header, name)) return "";   return Parameters::get().parameters[header][name]; }
    catch (const std::exception &e) {
        printf("\tCould not read parameter (string) at [%s][%s]. Error: %s", header.c_str(), name.c_str(), e.what());   return "";
    }
}
int pari(std::string header, std::string name) {
    try { if (!parExists(header, name)) return 0;   return stoi(Parameters::get().parameters[header][name]); }
    catch (const std::exception &e) {
        printf("\tCould not read parameter (int) at [%s][%s]. Error: %s", header.c_str(), name.c_str(), e.what());   return 0;
    }
}
float parf(std::string header, std::string name, float defaultValue) {
    try { if (!parExists(header, name)) return defaultValue;   return stof(Parameters::get().parameters[header][name]); }
    catch (const std::exception &e) {
        return defaultValue;
    }
}
float parf(std::string header, std::string name) {
    try { if (!parExists(header, name)) return 0.0f;   return stof(Parameters::get().parameters[header][name]); }
    catch (const std::exception &e) {
        printf("\tCould not read parameter (float) at [%s][%s]. Error: %s", header.c_str(), name.c_str(), e.what());   return 0.0f;
    }
}

