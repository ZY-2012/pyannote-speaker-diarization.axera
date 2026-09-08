cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(community1_diar)

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0 -fopenmp")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2 -fdata-sections -ffunction-sections -fopenmp")
endif()

include(cmake/msp_dependencies.cmake)

include_directories(${MSP_INC_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

set(SRC
    src/EngineWrapper.cpp
    src/diar_engine.cpp
    third_party/kissfft/kiss_fft.c
    third_party/kissfft/kiss_fftr.c
)

add_executable(${PROJECT_NAME} community1_diar.cpp ${SRC})
target_include_directories(${PROJECT_NAME} PRIVATE third_party/kissfft)
target_link_libraries(${PROJECT_NAME} ${MSP_LIBS})



install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ./)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/")

message(STATUS "========================================")
message(STATUS "community1_diar build configuration")
message(STATUS "========================================")
