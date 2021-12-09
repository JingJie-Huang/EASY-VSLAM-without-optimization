cmake_minimum_required( VERSION 2.8 )
project( Easy_SFM )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/g2o/cmake_modules )
#SET( G2O_ROOT /usr/local/include/g2o )

find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories( 
	${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
	${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)


add_executable( pose_estimation_3d2d_nodepth pose_estimation_3d2d_nodepth.cpp )
target_link_libraries( pose_estimation_3d2d_nodepth 
   ${OpenCV_LIBS}
   ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)








