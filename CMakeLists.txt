cmake_minimum_required(VERSION 3.14)

project(slime)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/slime/*.cpp
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/external)

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

# copy auxiliary files
file(COPY "slime/shaders/" DESTINATION "shaders/")
file(COPY "slime/resources/" DESTINATION "resources/")