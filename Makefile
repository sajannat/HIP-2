all: vec_add vec_add_managed vec_add_host

vec_add:
	hipcc -o vec_add vecAdd.cpp

vec_add_managed:
	hipcc -o vec_add_managed hipMallocManagedVecAdd.cpp

vec_add_host:
	hipcc -o vec_add_host hipHostMallocVecAdd.cpp

clean:
	rm -f vec_add vec_add_managed vec_add_host

run:
	time ./vec_add
	time ./vec_add_managed
	time ./vec_add_host