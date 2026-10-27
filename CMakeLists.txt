# Copyright (c) 2025-2026 Ewan Crawford

cmake_minimum_required (VERSION 3.20)
project(VkParticle)

# Find glm and glfw3 dependencies
find_package (glfw3 REQUIRED)
find_package (glm REQUIRED)

# Find the Vulkan SDK, providing the vulkan-hpp C++ headers
find_package (Vulkan REQUIRED)

# Find AdaptiveCpp, the SYCL implementation used for the particle simulation.
# Its Vulkan backend is required, so that the SYCL allocations and events can
# be shared with the Vulkan objects of this application. Configure with
# -DACPP_TARGETS=generic and CMAKE_CXX_COMPILER set to the acpp driver.
find_package (AdaptiveCpp CONFIG REQUIRED)

# Find slangc executable from vulkan SDK
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
# Compile slang shader and add as a dependency to a target
function(add_slang_shader_depedency DEP)
  set(SHADER_SOURCES ${CMAKE_SOURCE_DIR}/shaders/shader.slang)
  # The particle simulation is a SYCL kernel rather than a compute shader, so
  # only the graphics entry points are compiled here.
  set(ENTRY_POINTS -entry vertMain -entry fragMain)

  add_custom_command(
    OUTPUT  ${SHADERS_DIR}/slang.spv
    COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Slang Shaders"
    VERBATIM
   )
   add_custom_target(shader DEPENDS ${SHADERS_DIR}/slang.spv)
   add_dependencies(${DEP} shader)
endfunction()

# Create executable from source files
add_subdirectory(src)
add_executable(vkParticle ${VK_PARTICLE_SOURCES})
set_target_properties(vkParticle PROPERTIES CXX_STANDARD 20)
target_compile_definitions(vkParticle PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)
target_link_libraries(vkParticle Vulkan::Vulkan glfw)

# Compile the SYCL kernels of the target. The Vulkan headers have to be on the
# include path of every source file, as `common.hpp` includes them before
# <sycl/sycl.hpp> to enable the Vulkan backend interop API of AdaptiveCpp.
add_sycl_to_target(TARGET vkParticle SOURCES ${VK_PARTICLE_SOURCES})

# Add shader dependencies
add_slang_shader_depedency(vkParticle)
