cmake_minimum_required(VERSION 3.0.2)
project(img_gps)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  roscpp
  ros_numpy
  std_msgs
  cv_bridge
  image_geometry
  image_transport
  image_view
  OpenCV
)


catkin_package(CATKIN_DEPENDS
	sensor_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=c++17 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_STANDARD 17) #14
SET(CMAKE_CXX_STANDARD_REQUIRED ON)



#############
## Install ##
#############

catkin_install_python(PROGRAMS src/model_inference.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
