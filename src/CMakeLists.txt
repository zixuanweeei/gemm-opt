enable_testing()

file(GLOB common_files common/*.cpp)
add_library(common STATIC ${common_files})
target_include_directories(common PRIVATE ${PROJECT_SOURCE_DIR}/inc/
                                          ${PROJECT_SOURCE_DIR}/src/)

find_package(OpenMP)

function(register_example name srcs)
  add_executable(${name} ${srcs})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/inc/
                                             ${PROJECT_SOURCE_DIR}/src/)

  target_link_libraries(${name} PRIVATE common OpenMP::OpenMP_CXX)
  if(GO_PERF_ONLY_DEFAULT)
    target_compile_definitions(${name} PUBLIC -DPERF_ONLY)
  endif()
endfunction()

function(add_opt_test name command)
  add_test(${name} ${command} ${ARGN})
endfunction()

file(GLOB sources opt/*.cpp)

foreach(src ${sources})
  file(RELATIVE_PATH src_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${src})
  string(REGEX REPLACE "[/_\\.]" "-" example_name ${src_rel_path})
  register_example(${example_name} ${src})

  set(cmd "${example_name} --mode=C")
  string(REPLACE " " ";" cmd ${cmd})
  add_opt_test("${example_name}" ${cmd})
endforeach()
