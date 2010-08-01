Project(opencv-feature-tracker)

add_subdirectory(thirdparty)

# macro to setup the projects
MACRO(QuickSetup)
cmake_minimum_required(VERSION 2.6)
SET(CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake_modules
    ${CMAKE_MODULE_PATH}
    )
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
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
)
ENDMACRO(QuickExecutable)

file(GLOB SRCS src/*.cpp src/*.c)
file(GLOB HDRS include/*.h include/*.hpp include/opencv2/video/*.hpp)

QuickSetup()
QuickExecutable(DaisyDescriptor ${SRCS} ${HDRS} DAISY.cxx)
QuickExecutable(InteractiveMatching ${SRCS} ${HDRS} InteractiveMatching.cxx)
QuickExecutable(NCH ${SRCS} ${HDRS} NCH.cxx)
QuickExecutable(test_nch_svm ${SRCS} ${HDRS} test_nch_svm.cpp)
QuickExecutable(PCAConstruct ${SRCS} ${HDRS} PCAConstruct.cxx)
QuickExecutable(PCARecognize ${SRCS} ${HDRS} PCARecognize.cxx)
QuickExecutable(PCARecognizeInterestPoints ${SRCS} ${HDRS} PCARecognizeInterestPoints.cxx)
QuickExecutable(ExtractDaisyToLibSVMPackage ${SRCS} ${HDRS} ExtractDaisyToLibSVMPackage.cxx)
QuickExecutable(ExtractDaisyToTXT ${SRCS} ${HDRS} ExtractDaisyToTXT.cxx)
QuickExecutable(TrackMatchingDistance ${SRCS} ${HDRS} TrackMatchingDistance.cxx)
QuickExecutable(SaunierSayed ${SRCS} ${HDRS} SaunierSayed.cxx)
QuickExecutable(BlobTrackPedestrian ${SRCS} ${HDRS} BlobTrackPedestrian.cxx)

OPTION(COMPILE_TESTS "Compile unit tests using GoogleTest?")
IF(COMPILE_TESTS)
	#### Test executable with GTest
	FIND_PACKAGE(GTest)
	FILE(GLOB TEST_SRCS tests/test_*.cpp)
	include_directories(
		${GTEST_INCLUDE_DIR}
		${PROJECT_SOURCE_DIR}/tests
		${PROJECT_SOURCE_DIR}
		)
	set(TEST_DATA_FOLDER ${PROJECT_SOURCE_DIR}/tests/data)
	configure_file(${PROJECT_SOURCE_DIR}/tests/test_configuration.h.in ${PROJECT_SOURCE_DIR}/tests/test_configuration.h) 
	add_executable(RunAllTests ${SRCS} ${HDRS} ${TEST_SRCS} tests/test_configuration.h RunAllTests.cxx)
	target_link_libraries(RunAllTests
		${OpenCV_LIBS}
		${GTEST_LIBRARIES}
		daisy
		)
ENDIF(COMPILE_TESTS)
	
OPTION(USE_TBB "Use TBB speed up?")
IF(USE_TBB)
    FIND_PACKAGE(TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    link_directories(${TBB_LIBRARY_DIRS})

    add_executable(TBBParallelFor TBBParallelFor.cxx)
    target_link_libraries(TBBParallelFor ${TBB_LIBRARIES})
ENDIF(USE_TBB)

