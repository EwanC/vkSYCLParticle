// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
#include <algorithm>
#include <stdexcept>

namespace {
/// @brief Universally unique identifier of a physical device, which unlike a
/// VkPhysicalDevice handle can be compared between Vulkan instances.
using DeviceUUID = std::array<uint8_t, VK_UUID_SIZE>;

/// @brief Queries the UUID of a physical device of the instance of this
/// application.
DeviceUUID getDeviceUUID(const vk::raii::PhysicalDevice &device) {
  auto properties = device.getProperties2<vk::PhysicalDeviceProperties2,
                                          vk::PhysicalDeviceIDProperties>();
  const auto &id = properties.get<vk::PhysicalDeviceIDProperties>();
  DeviceUUID uuid;
  std::copy(std::begin(id.deviceUUID), std::end(id.deviceUUID), uuid.begin());
  return uuid;
}

/// @brief Queries the UUID of the VkPhysicalDevice backing a SYCL device.
///
/// The physical device belongs to the VkInstance that AdaptiveCpp has created,
/// so the function pointer has to be resolved from that instance rather than
/// using the dispatcher of this application.
/// @param[in] device SYCL device of the Vulkan backend.
/// @param[in] getInstanceProcAddr Loader entry point.
/// @returns UUID of the physical device, or an empty optional on failure.
std::optional<DeviceUUID>
getSyclDeviceUUID(const sycl::device &device,
                  PFN_vkGetInstanceProcAddr getInstanceProcAddr) {
  VkInstance instance =
      sycl::get_native<sycl::backend::vk>(device.get_platform());
  VkPhysicalDevice physicalDevice = sycl::get_native<sycl::backend::vk>(device);
  if (instance == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE) {
    return std::nullopt;
  }

  auto getProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      getInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
  if (!getProperties2) {
    return std::nullopt;
  }

  VkPhysicalDeviceIDProperties idProperties{};
  idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDeviceProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties.pNext = &idProperties;
  getProperties2(physicalDevice, &properties);

  DeviceUUID uuid;
  std::copy(std::begin(idProperties.deviceUUID),
            std::end(idProperties.deviceUUID), uuid.begin());
  return uuid;
}

/// @brief Finds the SYCL devices of the AdaptiveCpp Vulkan backend that can
/// share memory and semaphores, paired with the UUID of the physical device
/// they run on.
std::vector<std::pair<sycl::device, DeviceUUID>>
getShareableSyclDevices(PFN_vkGetInstanceProcAddr getInstanceProcAddr) {
  std::vector<std::pair<sycl::device, DeviceUUID>> result;
  for (const sycl::device &device : sycl::device::get_devices()) {
    if (device.get_backend() != sycl::backend::vk) {
      continue;
    }
    // Both extensions are optional in AdaptiveCpp, and without them the
    // allocations and timeline semaphores of the device cannot be shared.
    if (!sycl::vk_interop::has_external_memory_support(device) ||
        !sycl::vk_interop::has_external_semaphore_support(device)) {
      continue;
    }
    if (auto uuid = getSyclDeviceUUID(device, getInstanceProcAddr)) {
      result.emplace_back(device, *uuid);
    }
  }
  return result;
}
} // anonymous namespace

void VkSyclParticle::pickPhysicalDevice() {
  // AdaptiveCpp creates its own VkInstance and VkDevice, which this
  // application cannot influence, so rather than sharing them the two drive
  // separate logical devices and share memory and synchronization through
  // opaque file descriptors. That requires both to be created from the same
  // physical device, so only consider devices that one of the SYCL devices
  // also runs on.
  const auto syclDevices =
      getShareableSyclDevices(MContext.getDispatcher()->vkGetInstanceProcAddr);
  if (syclDevices.empty()) {
    throw std::runtime_error(
        "No SYCL device of the Vulkan backend supporting external memory and "
        "semaphores found. Set ACPP_VISIBILITY_MASK=vk to restrict "
        "AdaptiveCpp to its Vulkan backend.");
  }

  std::vector<vk::raii::PhysicalDevice> devices =
      MInstance.enumeratePhysicalDevices();
  const auto devIter = std::ranges::find_if(devices, [&](auto const &device) {
    // Check if the device supports the Vulkan 1.3 API version
    bool supportsVulkan1_3 =
        device.getProperties().apiVersion >= VK_API_VERSION_1_3;

    // Check if any of the queue families support graphics operations
    auto queueFamilies = device.getQueueFamilyProperties();
    bool supportsGraphics =
        std::ranges::any_of(queueFamilies, [](auto const &qfp) {
          return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

    // Check if all required device extensions are available
    auto availableDeviceExtensions =
        device.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions = std::ranges::all_of(
        MRequiredDeviceExtension,
        [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
          return std::ranges::any_of(
              availableDeviceExtensions,
              [requiredDeviceExtension](auto const &availableDeviceExtension) {
                return strcmp(availableDeviceExtension.extensionName,
                              requiredDeviceExtension) == 0;
              });
        });

    // Check if all required features are available
    auto features = device.template getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>();
    bool supportsRequiredFeatures =
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .synchronization2 &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .maintenance4 &&
        features
            .template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState &&
        features.template get<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>()
            .timelineSemaphore;

    if (!supportsVulkan1_3 || !supportsGraphics ||
        !supportsAllRequiredExtensions || !supportsRequiredFeatures) {
      return false;
    }

    // Only usable if one of the SYCL devices runs on the same physical device
    const DeviceUUID uuid = getDeviceUUID(device);
    const auto syclIter =
        std::ranges::find_if(syclDevices, [&uuid](const auto &entry) {
          return entry.second == uuid;
        });
    if (syclIter == syclDevices.end()) {
      return false;
    }
    MSyclDevice = syclIter->first;
    return true;
  });
  if (devIter == devices.end()) {
    throw std::runtime_error(
        "failed to find a GPU meeting the requirements of both this "
        "application and the SYCL Vulkan backend!");
  }
  MPhysicalDevice = *devIter;
}

void VkSyclParticle::createSyclQueue() {
  assert(MSyclDevice.has_value());
  // A context of a single device, as required to unambiguously map it onto the
  // VkDevice AdaptiveCpp has created. An in-order queue is used so that the
  // kernel of each frame is submitted to the same backend queue, and therefore
  // signals the timeline semaphore that has already been imported.
  MSyclContext.emplace(*MSyclDevice);
  MSyclQueue.emplace(*MSyclContext, *MSyclDevice,
                     sycl::property_list{sycl::property::queue::in_order{}});
}

void VkSyclParticle::createLogicalDevice() {
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
      MPhysicalDevice.getQueueFamilyProperties();

  // Find the index of the first queue family that supports graphics and compute
  for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size();
       qfpIndex++) {
    if ((queueFamilyProperties[qfpIndex].queueFlags &
         vk::QueueFlagBits::eGraphics) &&
        (queueFamilyProperties[qfpIndex].queueFlags &
         vk::QueueFlagBits::eCompute) &&
        MPhysicalDevice.getSurfaceSupportKHR(qfpIndex, *MSurface)) {
      // found a queue family that supports both graphics and present
      MQueueIndex = qfpIndex;
      break;
    }
  }
  if (MQueueIndex == ~0) {
    throw std::runtime_error(
        "Could not find a queue for graphics and present -> terminating");
  }

  // Setup pointer chain of structs with required features to create logical
  // device with.
  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                     vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>
      featureChain = {
          {}, // vk::PhysicalDeviceFeatures2
          {.synchronization2 = true,
           .dynamicRendering = true,
           .maintenance4 = true}, // vk::PhysicalDeviceVulkan13Features
          {.extendedDynamicState =
               true}, // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
          {.timelineSemaphore = true} // vk::PhysicalDeviceTimelineSemaphoreKHR
      };

  // create a logical device and queue
  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
      .queueFamilyIndex = MQueueIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority};
  vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
      .enabledExtensionCount =
          static_cast<uint32_t>(MRequiredDeviceExtension.size()),
      .ppEnabledExtensionNames = MRequiredDeviceExtension.data()};

  MDevice = vk::raii::Device(MPhysicalDevice, deviceCreateInfo);
  MQueue = vk::raii::Queue(MDevice, MQueueIndex, 0);
}
