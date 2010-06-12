cmake_minimum_required(VERSION 2.6)
project(daisy)

SET(SRCS
	src/daisy.cpp 
	src/image_io_bmp.cpp 
	src/image_io_png.cpp 
	src/image_io_pnm.cpp 
	src/image_io_jpeg.cpp 
	src/image_manipulation.cpp 
	src/progress_bar.cpp 
	src/interaction.cpp 
	src/general.cpp 
	src/corecv.cpp
	)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(daisy ${SRCS})
add_executable(daisy_tool src/main.cpp ${SRCS}) 