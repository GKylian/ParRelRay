@ECHO OFF
cls
TITLE Running tracing.exe
ECHO =================================================
ECHO Running the CUDA code
ECHO =================================================
nvprof prr.exe -t sim.params
ECHO[
ECHO[
TITLE PPM to PNG
ECHO =================================================
ECHO Python PPM to PNG transformation
ECHO =================================================
python convert.py
TITLE Done ray tracing