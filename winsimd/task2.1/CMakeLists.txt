
cmake_minimum_required (VERSION 3.29)
project (task2.1)

enable_testing()

#Executables

set(PROJECT_APP_NAME "task2.1")

#-DUSE_CUDA=OFF -DUSE_EIGEN=ON -DBUILD_SHARED_LIBS=OFF -DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON -DEIGEN_VECTORIZE_AVX=ON -DEIGEN_VECTORIZE_FMA=ON
# /arch:AVX /arch:AVX2 /arch:AVX512
add_definitions(-DEIGEN_ENABLE_AVX512=ON -D__AVX512F__=ON -DEIGEN_VECTORIZE_AVX512=ON -DEIGEN_VECTORIZE_AVX2=ON)

# gcc: mavx -mavx2 -mavx512f
# -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw
#target_compile_options(myproject PRIVATE /arch:AVX2)


set(UTILS_DIR "../common/")
include_directories("${UTILS_DIR}")
file(GLOB_RECURSE UTILS_FILES
    "${UTILS_DIR}/CpuFeatures.h"
    "${UTILS_DIR}/Utils.h"
    "${UTILS_DIR}/SimdRoutines.h"
)

#enable_language(C ASM_MASM)

set(SRC_LIST ./task2.1.cpp)

add_executable(${PROJECT_APP_NAME}
    "${UTILS_FILES}"
    "${SRC_LIST}"
)

target_compile_features(${PROJECT_APP_NAME} PRIVATE cxx_std_17)





