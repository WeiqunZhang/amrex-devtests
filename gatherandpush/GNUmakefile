AMREX_HOME ?= ../../amrex

DEBUG	= TRUE
DEBUG	= FALSE

DIM	= 3

COMP    = gcc

TINY_PROFILE  = TRUE
USE_PARTICLES = TRUE

PRECISION     = DOUBLE

USE_MPI       = TRUE
USE_OMP       = FALSE
USE_CUDA      = TRUE
USE_HIP       = FALSE
USE_DPCPP     = FALSE

#CXXSTD = c++17

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
