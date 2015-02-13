.DEFAULT_GOAL := all
CUDACC= nvcc
CUDAFLAGS = 

TARGETS = saxpy 

all: $(TARGETS)

SRCS_COMMON =  

DISTFILES += $(SRCS_COMMON) $(DEPS_COMMON)


saxpy$(EXEEXT): saxpy.cu $(SRCS_COMMON) $(DEPS_COMMON)
	$(CUDACC) $(CUDAFLAGS) \
	    -o $@ saxpy.cu $(SRCS_COMMON) $(MPILDFLAGS)
clean:
	rm -f $(TARGETS) 

# eof
