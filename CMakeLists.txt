cmake_minimum_required(VERSION 3.18)
project(slime LANGUAGES CXX CUDA)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/slime/*.h
    ${CMAKE_SOURCE_DIR}/slime/*.cuh
    ${CMAKE_SOURCE_DIR}/slime/*.cpp
    ${CMAKE_SOURCE_DIR}/slime/*.cu
)

file(GLOB_RECURSE SHADER_FILES 
     ${CMAKE_SOURCE_DIR}/slime/*.vert
     ${CMAKE_SOURCE_DIR}/slime/*.frag
)

file(GLOB_RECURSE RESOURCE_FILES
     ${CMAKE_SOURCE_DIR}/slime/*.png
     ${CMAKE_SOURCE_DIR}/slime/*.jpg
)

add_executable(${PROJECT_NAME} ${SRC_FILES} ${SHADER_FILES} ${RESOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/external)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
                            

# OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)

# glfw 
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

# glad
add_subdirectory(${CMAKE_SOURCE_DIR}/external/glad)
target_link_libraries(${PROJECT_NAME} PRIVATE glad)

# glm
add_subdirectory(${CMAKE_SOURCE_DIR}/external/glm)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

# copy shader files
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Release/shaders)
foreach(SHADER ${SHADER_FILES})
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${SHADER}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/)
endforeach()

# copy resource files
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Release/resources)
foreach(RESOURCE ${RESOURCE_FILES})
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${RESOURCE}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>/resources/)
endforeach()