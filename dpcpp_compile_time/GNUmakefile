AMREX_HOME ?= ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = TRUE

BL_NO_FORT = TRUE

LOG_BUILD_TIME = TRUE

#DEFINES += -DSLOW_COMPILE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# disable ccache
CCACHE :=

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

default: $(objEXETempDir)/main.o
	@echo "SUCCESS"

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
