// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
#include <stdexcept>
#include <unistd.h>

vk::Semaphore VkSyclParticle::importSyclSemaphore(const sycl::event &event) {
  assert(MSyclContext.has_value());

  // The timeline semaphore AdaptiveCpp signals when this event completes.
  // An in-order queue is used, so in practice this is the same semaphore for
  // every frame, but AdaptiveCpp makes no guarantee of that and may use
  // several timelines.
  const auto nativeEvent = sycl::get_native<sycl::backend::vk>(event);
  if (nativeEvent.semaphore == VK_NULL_HANDLE) {
    throw std::runtime_error("the SYCL event has no timeline semaphore!");
  }

  // Importing an opaque file descriptor consumes it, and the import is
  // permanent, so the result is cached rather than repeated every frame.
  if (*MSyclSemaphore && nativeEvent.semaphore == MSyclSemaphoreSource) {
    return *MSyclSemaphore;
  }

  const auto exported = sycl::AdaptiveCpp_vk_interop::export_semaphore(event);
  if (!exported.handle.is_valid()) {
    throw std::runtime_error("failed to export the SYCL timeline semaphore!");
  }

  // Create the semaphore to import into, which has to be a timeline to match
  // the semaphore it is imported from.
  vk::SemaphoreTypeCreateInfo semaphoreType{
      .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
  vk::raii::Semaphore semaphore(MDevice, {.pNext = &semaphoreType});

  // A permanent import, as a temporary one is reset by the first wait on it,
  // and is not permitted for timeline semaphores in any case.
  try {
    MDevice.importSemaphoreFdKHR(
        {.semaphore = semaphore,
         .handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd,
         .fd = exported.handle.fd});
  } catch (...) {
    close(exported.handle.fd);
    throw;
  }

  MSyclSemaphore = std::move(semaphore);
  MSyclSemaphoreSource = nativeEvent.semaphore;
  return *MSyclSemaphore;
}
