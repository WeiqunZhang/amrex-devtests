AMREX_HOME ?= ../../amrex

DEBUG	= TRUE
DEBUG	= FALSE

DIM	= 3

COMP    = gcc

TINY_PROFILE  = FALSE
USE_PARTICLES = TRUE

BL_NO_FORT = TRUE

PRECISION     = DOUBLE

USE_MPI       = FALSE
USE_OMP       = FALSE
USE_CUDA      = FALSE
USE_HIP       = FALSE
USE_SYCL      = TRUE

#CXXSTD = c++17

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(USE_SYCL),TRUE)
# CXXFLAGS += -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend '-device xehp'
endif

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
