// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
#include <stdexcept>

void vkParticle::drawFrame() {
  // The image at `imageIndex` will be available when the fence for the
  // current frame is signalled
  auto [result, imageIndex] = MSwapChain.acquireNextImage(
      UINT64_MAX, nullptr, *MInFlightFences[MCurrentFrame]);
  while (vk::Result::eTimeout ==
         MDevice.waitForFences(*MInFlightFences[MCurrentFrame], vk::True,
                               UINT64_MAX)) {
    ;
  }
  // Reset fence back to unsignalled state after it has been signalled.
  MDevice.resetFences(*MInFlightFences[MCurrentFrame]);

  // Submit the particle simulation to the SYCL queue, which runs on its own
  // VkDevice, writing the particles of this frame into memory shared with the
  // vertex buffer of this frame.
  const sycl::event computeEvent =
      updateParticles(static_cast<float>(MLastFrameTime));

  // Import the timeline semaphore the SYCL queue signals on completion, so
  // that the graphics work below can wait on it on the device.
  //
  // NOTE: querying the native event forces AdaptiveCpp to submit the work, so
  // this must happen before the graphics submission waits on it.
  const vk::Semaphore computeSemaphore = importSyclSemaphore(computeEvent);
  const uint64_t computeSignalValue =
      sycl::get_native<sycl::backend::vk>(computeEvent).value;

  // Value this application's timeline reaches once the graphics work below
  // has completed, which is what the SYCL kernel of a later frame waits on
  // before overwriting the particles read here.
  const uint64_t graphicsSignalValue = ++MTimelineValue;

  // Submit graphics work to device
  {
    // Setup graphics command-buffer with commands.
    recordGraphicsCommandBuffer(imageIndex);

    // Waits on the timeline of the SYCL context, signals the timeline of this
    // application. The two semaphores are distinct, so unlike a single
    // timeline the wait and signal values are unrelated.
    const std::array waitSemaphores{computeSemaphore};
    const std::array waitValues{computeSignalValue};
    const std::array signalSemaphores{*MSemaphore};
    const std::array signalValues{graphicsSignalValue};

    vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
        .waitSemaphoreValueCount = waitValues.size(),
        .pWaitSemaphoreValues = waitValues.data(),
        .signalSemaphoreValueCount = signalValues.size(),
        .pSignalSemaphoreValues = signalValues.data()};

    // Wait at vertex input stage for the SYCL kernel to finish
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eVertexInput;

    // Submit command-buffer to queue
    vk::SubmitInfo graphicsSubmitInfo{
        .pNext = &graphicsTimelineInfo,
        .waitSemaphoreCount = waitSemaphores.size(),
        .pWaitSemaphores = waitSemaphores.data(),
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &*MGraphicsCommandBuffers[MCurrentFrame],
        .signalSemaphoreCount = signalSemaphores.size(),
        .pSignalSemaphores = signalSemaphores.data()};
    MQueue.submit(graphicsSubmitInfo, nullptr);

    // Record that the particles of this frame may not be overwritten by SYCL
    // until the graphics work just submitted has completed.
    MFrameGraphicsValue[MCurrentFrame] = graphicsSignalValue;

    // Present the image (wait for graphics to finish)
    vk::SemaphoreWaitInfo waitInfo{.semaphoreCount = 1,
                                   .pSemaphores = &*MSemaphore,
                                   .pValues = &graphicsSignalValue};

    // Wait for graphics to complete before presenting rendered frame
    while (vk::Result::eTimeout == MDevice.waitSemaphores(waitInfo, UINT64_MAX))
      ;

    // Before an application can display an image it's format must
    // be transitioned to an appropriate layout.
    vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 0,
                                   .pWaitSemaphores = nullptr,
                                   .swapchainCount = 1,
                                   .pSwapchains = &*MSwapChain,
                                   .pImageIndices = &imageIndex};
    result = MQueue.presentKHR(presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR || MFramebufferResized) {
      MFramebufferResized = false;
      recreateSwapChain();
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("failed to present swap chain image!");
    }
  }

  // Update current frame counter
  MCurrentFrame = (MCurrentFrame + 1) % SMaxFramesInFlight;
}
