#
# external libraries compilation
#
ifeq ($(strip $(GPU)),1)
libs: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make
	cd $(LIBS_DIR)/cuDecomp && mkdir -p build && cd build && cmake .. -DCUDECOMP_CUDA_CC_LIST="121" -DCMAKE_CXX_STANDARD=17 && make -j
libsclean: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make clean
	cd $(LIBS_DIR)/cuDecomp/build && make clean; cd .. && rm -rf build
else
libs: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make
libsclean: $(wildcard $(LIBS_DIR)/2decomp-fft/src/*.f90)
	cd $(LIBS_DIR)/2decomp-fft && make clean
endif
