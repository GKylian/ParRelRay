@ECHO OFF
cls
TITLE Compiling...
ECHO =================================================
ECHO Compiling the CUDA code
ECHO =================================================
nvcc -o prr.exe cudamaths/image.cu main/Spacetime.cpp prr.cpp main/Camera.cpp main/Parameters.cpp -x cu -w -O3
ECHO[
ECHO[
TITLE Running tracing.exe
ECHO =================================================
ECHO Running the CUDA code
ECHO =================================================
nvprof prr.exe -t sim.params
ECHO[
TITLE Done ray tracing