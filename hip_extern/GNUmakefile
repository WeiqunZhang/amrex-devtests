AMREX_HOME = ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = TRUE
USE_DPCPP = FALSE

BL_NO_FORT = TRUE
CXXSTD = c++17

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

#ifeq ($(USE_HIP),TRUE)
#  CXXFLAGS += -fgpu-rdc
#  LINKFLAGS += -fgpu-rdc
#endif
