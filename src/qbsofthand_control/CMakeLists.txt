cmake_minimum_required(VERSION 3.8)
project(qbsofthand_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetClosure.srv"
)

add_executable(qbsofthand_control_node
  src/qbshr_node.cpp
)

target_include_directories(qbsofthand_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/qbrobotics_research_api>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(qbsofthand_control_node
  rclcpp
  std_msgs
  std_srvs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# NOTE: keep target_link_libraries() in "plain" form because ament_target_dependencies()
# internally uses the plain signature.
target_link_libraries(qbsofthand_control_node
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/libqbrobotics_driver.a
  ${CMAKE_CURRENT_SOURCE_DIR}/libs/libSerial.a
  ${cpp_typesupport_target}
)

install(TARGETS qbsofthand_control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()