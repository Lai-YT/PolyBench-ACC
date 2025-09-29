DEFINES ?=
NVCC ?= nvcc -arch=native
CUBLAS ?=
FLAGS = 

ifeq ($(CUBLAS), 1)
	FLAGS += -DRUN_ON_CUBLAS -lcublas_static -lcublasLt_static -lculibos
endif

all:
	${NVCC} -O3 ${CUFILES} -I${PATH_TO_UTILS} -o ${EXECUTABLE} ${DEFINES} ${FLAGS}
clean:
	rm -f *~ *.exe
