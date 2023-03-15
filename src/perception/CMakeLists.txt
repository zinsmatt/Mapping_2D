cmake_minimum_required(VERSION 3.16.3)
project(mapping_perception)

# find_package(catkin REQUIRED COMPONENTS
#   geometry_msgs
#   roscpp
#   rospy
#   sensor_msgs
#   std_msgs
# )


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)


add_executable(perception_node
main.cpp
perception.cpp)

target_include_directories(perception_node PRIVATE ${catkin_INCLUDE_DIRS})

target_link_libraries(perception_node
${catkin_LIBRARIES}
)


