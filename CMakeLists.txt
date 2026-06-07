cmake_minimum_required(VERSION 3.18)
project(slime LANGUAGES C CXX)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/slime/*.cpp
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
  target_sources(${PROJECT_NAME} PRIVATE
      ${CMAKE_SOURCE_DIR}/slime/simulation/portable_sph_simulator_metal.mm
  )
  target_compile_options(${PROJECT_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE
      "-framework Metal"
      "-framework Foundation"
  )
else()
  enable_language(CUDA)
  target_sources(${PROJECT_NAME} PRIVATE
      ${CMAKE_SOURCE_DIR}/slime/simulation/portable_sph_simulator_cuda.cu
  )
  target_compile_definitions(${PROJECT_NAME} PRIVATE SLIME_USE_CUDA=1)
  set_target_properties(${PROJECT_NAME} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
  )
endif()
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external
)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/external/glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/glad)
target_link_libraries(${PROJECT_NAME} PRIVATE glad)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/glm)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
  )
else()
  target_compile_options(${PROJECT_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
      $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Wpedantic>
  )
endif()

file(GLOB_RECURSE SHADER_FILES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/slime/*.vert
    ${CMAKE_SOURCE_DIR}/slime/*.frag
)
foreach(SHADER ${SHADER_FILES})
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHADER}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders/
  )
endforeach()
