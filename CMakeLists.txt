cmake_minimum_required(VERSION 2.6)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

Project(opencv-feature-tracker)

add_subdirectory(thirdparty)

# macro to setup the projects
MACRO(QuickSetup)

SET(CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake_modules
    ${CMAKE_MODULE_PATH}
    )
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization graph)
find_package(GFlags)
include_directories(
    ${Boost_INCLUDE_DIR}
    ${GFlags_INCLUDE_DIR}
	${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/daisy-1.8.1/include
)
ENDMACRO(QuickSetup)

# macro to create a project
# you need to call QuickSetup first
MACRO(QuickExecutable proj_name)

add_executable(${proj_name} ${ARGN})
target_link_libraries(${proj_name}
    ${OpenCV_LIBS}
      daisy
    ${GFlags_LIBS}
	${Boost_LIBRARIES}
)
ENDMACRO(QuickExecutable)

# Setup
QuickSetup()

# Get the files
file(GLOB SRCS src/*.cpp src/*.c src/featuretrack/*.cpp src/blobtrack/*.cpp)
file(GLOB HDRS include/*.h include/*.hpp include/opencv2/video/*.hpp)

#### Test executable with GTest
FIND_PACKAGE(GTest)
FILE(GLOB TEST_SRCS tests/test_*.cpp)
include_directories(
	${GTEST_INCLUDE_DIR}
	${PROJECT_SOURCE_DIR}/tests
	${PROJECT_SOURCE_DIR}
	)
link_directories(
	${Boost_LIBRARY_DIRS}
)
MESSAGE(STATUS "Compiling Test executables. Include dir is ${GTEST_INCLUDE_DIR}")
set(TEST_DATA_FOLDER ${PROJECT_SOURCE_DIR}/tests/data)
configure_file(${PROJECT_SOURCE_DIR}/tests/test_configuration.h.in ${PROJECT_SOURCE_DIR}/tests/test_configuration.h) 
add_executable(RunAllTests ${SRCS} ${HDRS} ${TEST_SRCS} tests/test_configuration.h RunAllTests.cxx)
target_link_libraries(RunAllTests
	${OpenCV_LIBS}
	${GTEST_LIBRARIES}
	daisy
	)

# Create other executables
QuickExecutable(FeatureBasedTracking ${SRCS} ${HDRS} FeatureBasedTracking.cxx)
QuickExecutable(BlobTrackPedestrian ${SRCS} ${HDRS} BlobTrackPedestrian.cxx)


