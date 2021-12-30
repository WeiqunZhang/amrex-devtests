DEBUG = FALSE
TEST = TRUE
USE_ASSERTION = TRUE

USE_EB = FALSE

USE_MPI  = TRUE
USE_OMP  = FALSE

USE_HYPRE = FALSE

COMP = gnu

DIM = 2

BL_NO_FORT = TRUE

AMREX_HOME = ../../amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs := Base Boundary LinearSolvers/MLMG

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

