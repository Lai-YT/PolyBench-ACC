DEFINES ?=
all:
	nvcc -O3 ${CUFILES} -I${PATH_TO_UTILS} -o ${EXECUTABLE} ${DEFINES}
clean:
	rm -f *~ *.exe
