// Copyright (c) 2025-2026 Ewan Crawford

#pragma once

#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

// Included after the Vulkan headers so that AdaptiveCpp enables its Vulkan
// backend interop API, which it only does when the Vulkan headers are
// reachable. `ACPP_GLUE_VK_INTEROP_AVAILABLE` is defined by it in that case.
#include <sycl/sycl.hpp>

#if !defined(ACPP_GLUE_VK_INTEROP_AVAILABLE)
#error "AdaptiveCpp was built without the Vulkan backend, or its Vulkan "\
       "interop API is unavailable. The Vulkan backend is required."
#endif

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

/// @brief Class used to interface with shader device code
struct Particle {
  glm::vec2 position;
  glm::vec2 velocity;
  glm::vec4 color;

  // Tells the runtime what stride to use for vertex data
  static vk::VertexInputBindingDescription getBindingDescription() {
    return {0, sizeof(Particle), vk::VertexInputRate::eVertex};
  }

  static std::array<vk::VertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    return {
        // In Vertex shader input, we have a float2 position struct attribute
        // followed by a float4 color attribute.
        vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat,
                                            offsetof(Particle, position)),
        vk::VertexInputAttributeDescription(
            1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color))};
  }
};

/// @brief Class holding RAII state of the application
struct VkSyclParticle {
  /// @brief User code entry-point, called by main.cpp
  void run();

  /// @brief Set by GLFW callback when window is resized.
  bool MFramebufferResized = false;

private:
  /*
   * Member methods
   */

  /// @brief Initializes GLFW and creates a window.
  void initWindow();
  /// @brief Creates required VK objects.
  void initVulkan();
  /// @brief Game loop that draws frames until user exists the program.
  void mainLoop();
  /// @brief Tears down GLFW instance on program exit.
  void cleanup();

  /// @brief Creates a VkInstance with required extensions and layers.
  void createInstance();
  /// @brief Registers validation layers, if enabled.
  void setupDebugMessenger();
  /// @brief Creates a VkSurfaceKHR window surface to interface with GLFW.
  void createSurface();
  /// @brief Selects a VkPhysicalDevice to use from the VK instance based on
  /// the application requirements, and the SYCL device of the AdaptiveCpp
  /// Vulkan backend that refers to the same physical device.
  ///
  /// Both are needed because AdaptiveCpp creates its own VkInstance and
  /// VkDevice, so the handles are not comparable between the two instances.
  /// Sharing memory through opaque file descriptors requires that the two
  /// logical devices originate from the same physical device, which is
  /// established by matching `VkPhysicalDeviceIDProperties::deviceUUID`.
  void pickPhysicalDevice();
  /// @brief Creates a logical device and queue with required capabilities.
  void createLogicalDevice();
  /// @brief Creates a swap chain image buffer for rendering.
  void createSwapChain();
  /// @brief Creates a view into each image in the swap chain.
  void createImageViews();
  /// @brief Creates the SYCL context and queue used for the compute work, on
  /// the SYCL device selected by pickPhysicalDevice().
  void createSyclQueue();
  /// @brief Loads vertex & fragment shaders,
  /// and creates graphics pipeline.
  void createGraphicsPipeline();
  /// @brief Creates a command pool.
  void createCommandPool();
  /// @brief Allocates the `Particle` buffer of every frame as exportable SYCL
  /// device USM, imports the backing device memory into the VkDevice of this
  /// application, and binds a vertex buffer to it.
  void createShaderStorageBuffers();
  /// @brief Creates a command-buffer to use for graphics commands for each of
  /// the possible frames in flight.
  void createGraphicsCommandBuffers();
  /// @brief Creates timeline semaphore and fences for synchronization, and
  /// shares the timeline semaphore with the SYCL context.
  void createSyncObjects();

  /// @brief Add commands to graphics command-buffer
  /// @param[in] imageIndex Index in swap chain of current image for frame.
  void recordGraphicsCommandBuffer(uint32_t imageIndex);
  /// @brief Submits the SYCL kernel updating the particles of the current
  /// frame from those of the previous frame.
  /// @param[in] deltaTime Time elapsed since the last frame.
  /// @returns Event of the submitted kernel.
  [[nodiscard]] sycl::event updateParticles(float deltaTime);
  /// @brief Submits the command-buffers to the queue,
  /// and presents the new frame.
  void drawFrame();
  /// @brief Reconfigures the swap chain image formats if the window is resized.
  void recreateSwapChain();
  /// @brief Resets swap chain state.
  void cleanupSwapChain();
  /// @brief Imports a timeline semaphore of the SYCL context into the
  /// VkDevice of this application, so that the graphics queue can wait on the
  /// completion of SYCL work.
  ///
  /// The import is permanent and the result is cached, since an opaque file
  /// descriptor may only be imported once. AdaptiveCpp may use more than one
  /// timeline, so the semaphore the SYCL event refers to is re-imported if it
  /// differs from the cached one.
  /// @param[in] event Event whose semaphore should be imported.
  /// @returns Semaphore of this application sharing the SYCL timeline.
  vk::Semaphore importSyclSemaphore(const sycl::event &event);

  /*
   * Static member variables
   */
  static const uint32_t SWindowWidth = 800;
  static const uint32_t SWindowHeight = 600;
  static const unsigned SMaxFramesInFlight = 2;
  static const uint64_t SFenceTimeout = 100000000;
  /// Number of work-items in a work-group of the SYCL particle update kernel.
  static constexpr uint32_t SComputeWorkItems = 16;
  /// Number of work-groups of the SYCL particle update kernel.
  static constexpr uint32_t SComputeWorkGroups = 32;
  /// Total number of particles simulated and drawn.
  static constexpr uint32_t SParticleCount =
      SComputeWorkItems * SComputeWorkGroups;
  static constexpr bool SEnableValidationLayers =
#ifdef NDEBUG
      false;
#else
      true;
#endif
  static const std::vector<const char *> SValidationLayers;

  /*
   * Member variables
   */

  GLFWwindow *MWindow = nullptr;
  vk::raii::Context MContext;
  vk::raii::Instance MInstance = nullptr;
  vk::raii::DebugUtilsMessengerEXT MDebugMessenger = nullptr;
  vk::raii::SurfaceKHR MSurface = nullptr;
  vk::raii::PhysicalDevice MPhysicalDevice = nullptr;
  vk::raii::Device MDevice = nullptr;
  uint32_t MQueueIndex = ~0;
  vk::raii::Queue MQueue = nullptr;
  vk::raii::SwapchainKHR MSwapChain = nullptr;
  std::vector<vk::Image> MSwapChainImages;
  vk::SurfaceFormatKHR MSwapChainSurfaceFormat;
  vk::Extent2D MSwapChainExtent;
  std::vector<vk::raii::ImageView> MSwapChainImageViews;

  vk::raii::PipelineLayout MPipelineLayout = nullptr;
  vk::raii::Pipeline MGraphicsPipeline = nullptr;

  // Vertex buffers bound to the device memory imported from the SYCL USM
  // allocations in `MParticles`, one for each frame in flight.
  std::vector<vk::raii::Buffer> MShaderStorageBuffers;
  std::vector<vk::raii::DeviceMemory> MShaderStorageBuffersMemory;

  vk::raii::CommandPool MCommandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> MGraphicsCommandBuffers;

  // Timeline semaphore of this application, signalled once the graphics work
  // of a frame has completed. Created exportable so that it can be imported
  // into the SYCL context as `MSyclWaitSemaphore`, letting SYCL wait until a
  // buffer is no longer read as a vertex buffer before overwriting it.
  vk::raii::Semaphore MSemaphore = nullptr;
  uint64_t MTimelineValue = 0;
  std::vector<vk::raii::Fence> MInFlightFences;
  uint32_t MCurrentFrame = 0;

  /*
   * SYCL members
   */

  // Device of the AdaptiveCpp Vulkan backend on the same physical device as
  // `MPhysicalDevice`, and the context and queue created on it.
  std::optional<sycl::device> MSyclDevice;
  std::optional<sycl::context> MSyclContext;
  std::optional<sycl::queue> MSyclQueue;

  // Particles of each frame in flight, as exportable SYCL device USM.
  std::array<Particle *, SMaxFramesInFlight> MParticles{};

  // `MSemaphore` imported into the SYCL context.
  sycl::vk_interop::imported_semaphore MSyclWaitSemaphore;

  // Timeline semaphore of the SYCL context imported into this application,
  // alongside the VkSemaphore of the SYCL context it was imported from, which
  // is used to detect that a different timeline needs to be imported.
  vk::raii::Semaphore MSyclSemaphore = nullptr;
  VkSemaphore MSyclSemaphoreSource = VK_NULL_HANDLE;

  // Value `MSemaphore` attains once the graphics work that last read the
  // particles of a frame in flight has completed, or 0 if they have not been
  // read yet.
  std::array<uint64_t, SMaxFramesInFlight> MFrameGraphicsValue{};

  double MLastFrameTime = 0.0;
  double MLastTime = 0.0;

  std::vector<const char *> MRequiredDeviceExtension = {
      vk::KHRSwapchainExtensionName,
      vk::KHRSpirv14ExtensionName,
      vk::KHRSynchronization2ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
      vk::KHRShaderDrawParametersExtensionName,
      // Used to import the memory and timeline semaphores that are shared
      // with the VkDevice AdaptiveCpp has created for the SYCL context.
      vk::KHRExternalMemoryFdExtensionName,
      vk::KHRExternalSemaphoreFdExtensionName,
  };
};

/*
 * Free functions from shader_file.cpp
 */

/// @brief Loads a file from disk.
/// @param[in] filename Path on disk for file to read.
/// @returns Vector of char bytes with file contents.
std::vector<char> readFile(const std::string &filename);

/// @brief Creates a VK shader module from shader source
/// @param[in] code Source code of shader.
/// @param[in] device The device to created the model for.
/// @returns The created shader module.
[[nodiscard]] vk::raii::ShaderModule
createShaderModule(const std::vector<char> &code, vk::raii::Device &device);
