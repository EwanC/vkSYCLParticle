// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

const std::vector<const char *> vkParticle::SValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};

namespace {
// Callback invoked when window is resized, which sets a boolean member in an
// instance of the vkParticle class.
void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
  auto app = reinterpret_cast<vkParticle *>(glfwGetWindowUserPointer(window));
  app->MFramebufferResized = true;
}
} // anonymous namespace

void vkParticle::run() {
  initWindow();
  initVulkan();
  mainLoop();
  cleanup();
}

void vkParticle::initWindow() {
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  MWindow = glfwCreateWindow(SWindowWidth, SWindowHeight, "vkParticle", nullptr,
                             nullptr);
  glfwSetWindowUserPointer(MWindow, this);
  glfwSetFramebufferSizeCallback(MWindow, framebufferResizeCallback);
}

void vkParticle::initVulkan() {
  createInstance();
  setupDebugMessenger();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSyclQueue();
  createSwapChain();
  createImageViews();
  createGraphicsPipeline();
  createCommandPool();
  // Creates the timeline semaphore before the buffers, so that the particle
  // allocations are made on a SYCL context that can already be synchronized
  // with this application.
  createSyncObjects();
  createShaderStorageBuffers();
  createGraphicsCommandBuffers();
}

void vkParticle::cleanup() {
  // Objects shared between this application and the SYCL context have to be
  // destroyed in order, as the RAII members below are otherwise torn down
  // after the SYCL context, which still refers to them.
  if (*MDevice) {
    MDevice.waitIdle();
  }
  // Drops the import of `MSemaphore` into the SYCL context, which has to
  // happen before `MSemaphore` itself is destroyed.
  MSyclWaitSemaphore.reset();
  MSyclSemaphore = nullptr;
  MSyclSemaphoreSource = VK_NULL_HANDLE;

  // Release the buffers bound to the imported memory, and the imports, before
  // the SYCL allocations that own the memory.
  MShaderStorageBuffers.clear();
  MShaderStorageBuffersMemory.clear();
  if (MSyclContext.has_value()) {
    for (Particle *particles : MParticles) {
      if (particles) {
        sycl::free(particles, *MSyclContext);
      }
    }
  }
  MParticles.fill(nullptr);
  MSyclQueue.reset();
  MSyclContext.reset();
  MSyclDevice.reset();

  glfwDestroyWindow(MWindow);
  glfwTerminate();
}

namespace {
std::vector<const char *>
getGLFWRequiredExtensions(bool enableValidationLayers) {
  uint32_t glfwExtCount = 0;
  auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtCount);

  std::vector extensions(glfwExtensions, glfwExtensions + glfwExtCount);
  if (enableValidationLayers) {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }
  return extensions;
}
} // namespace

void vkParticle::createInstance() {
  constexpr vk::ApplicationInfo appInfo{
      .pApplicationName = "vkParticle",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14};

  // Enable validation layer if requested
  std::vector<char const *> requiredLayers;
  if (SEnableValidationLayers) {
    requiredLayers.assign(SValidationLayers.begin(), SValidationLayers.end());
  }

  auto layerProperties = MContext.enumerateInstanceLayerProperties();
  for (auto const &requiredLayer : requiredLayers) {
    bool layerUnsupported = std::ranges::none_of(
        layerProperties, [requiredLayer](auto const &layerProperty) {
          return strcmp(layerProperty.layerName, requiredLayer) == 0;
        });
    if (layerUnsupported) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(requiredLayer));
    }
  }

  // Get required GLFW extensions and check all are supported by the Vulkan
  // implementation.
  auto requiredExtensions = getGLFWRequiredExtensions(SEnableValidationLayers);
  auto extensionProperties = MContext.enumerateInstanceExtensionProperties();
  for (auto const &requiredExtension : requiredExtensions) {
    bool extUnsupported = std::ranges::none_of(
        extensionProperties,
        [requiredExtension](auto const &extensionProperty) {
          return strcmp(extensionProperty.extensionName, requiredExtension) ==
                 0;
        });
    if (extUnsupported) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(requiredExtension));
    }
  }

  vk::InstanceCreateInfo createInfo{
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
      .ppEnabledLayerNames = requiredLayers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
      .ppEnabledExtensionNames = requiredExtensions.data()};
  MInstance = vk::raii::Instance(MContext, createInfo);
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
  if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
      severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::cerr << "validation layer: type " << to_string(type)
              << " msg: " << pCallbackData->pMessage << std::endl;
  }

  return vk::False;
}

void vkParticle::setupDebugMessenger() {
  if (!SEnableValidationLayers)
    return;

  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
      .messageSeverity = severityFlags,
      .messageType = messageTypeFlags,
      .pfnUserCallback = &debugCallback};

  MDebugMessenger =
      MInstance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void vkParticle::mainLoop() {
  // Exit on escape key press or GUI window close
  while (glfwGetKey(MWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
         !glfwWindowShouldClose(MWindow)) {
    glfwPollEvents();
    drawFrame();
    // We want to animate the particle system using the last frames time to get
    // smooth, frame-rate independent animation
    double currentTime = glfwGetTime();
    MLastFrameTime = (currentTime - MLastTime) * 1000.0;
    MLastTime = currentTime;
  }
  MDevice.waitIdle();
}

void vkParticle::createSyncObjects() {
  MInFlightFences.clear();

  // Create a timeline semaphore with counter initialized to zero, exportable
  // so that the SYCL context can wait on the graphics work of this
  // application completing before overwriting a particle buffer.
  vk::ExportSemaphoreCreateInfo exportInfo{
      .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd};
  vk::SemaphoreTypeCreateInfo semaphoreType{.pNext = &exportInfo,
                                            .semaphoreType =
                                                vk::SemaphoreType::eTimeline,
                                            .initialValue = 0};
  MSemaphore = vk::raii::Semaphore(MDevice, {.pNext = &semaphoreType});
  MTimelineValue = 0;
  MFrameGraphicsValue.fill(0);

  // Import the semaphore into the SYCL context. The file descriptor is
  // consumed by a successful import, and closed by `imported_semaphore` if
  // the import fails.
  const int semaphoreFd = MDevice.getSemaphoreFdKHR(
      {.semaphore = MSemaphore,
       .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd});
  MSyclWaitSemaphore =
      sycl::vk_interop::imported_semaphore(*MSyclContext, semaphoreFd);
  if (!MSyclWaitSemaphore) {
    throw std::runtime_error(
        "failed to import the graphics timeline semaphore into the SYCL "
        "context!");
  }

  // Fence for host synchronization for each possible frame.
  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    vk::FenceCreateInfo fenceInfo{};
    MInFlightFences.emplace_back(MDevice, fenceInfo);
  }
}
