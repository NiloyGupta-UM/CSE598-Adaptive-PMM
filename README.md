# CSE598-Adaptive-PMM

This repository contains the code for my adaptive piecewise multilinear map implementation developed for the CSE 598 final project.

To run this code on your own, you will need to be able to interface the files with an existing adjoint-enabled Reynolds-averaged Navier Stokes code that is written in C++. The code requires access to arrays that store field data and the RANS code must also use ADOL-C. The header files can simply be inlcuded in the RANS code file where it makes the most sense, for example where the turbulence model is defined. An example of the model augmentation file is provided in AUG_Wilcox_PrTurb_APMM.hpp. An example of how it can be interfaced within the RANS solver is provided in the RansTurb.h file (forward simulation) and RansTurbAD.h file (adjoint computation).

An example of a valid adaptive PMM mesh file is also provided in Adaptive_PMM_2D_v1.apmm.
