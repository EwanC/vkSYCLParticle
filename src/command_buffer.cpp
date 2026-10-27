// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"

void vkParticle::createCommandPool() {
  // Reset command-buffer bit means that command-buffers can be reset
  // individually, rather than as a group, which is what we want to
  // reset a command-buffer each frame.
  vk::CommandPoolCreateInfo poolInfo{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = MQueueIndex};
  MCommandPool = vk::raii::CommandPool(MDevice, poolInfo);
}

void vkParticle::createGraphicsCommandBuffers() {
  MGraphicsCommandBuffers.clear();
  // Use primary command-buffers, as they are submitted directly to a queue,
  // rather than indirectly from other command-buffers.
  vk::CommandBufferAllocateInfo allocInfo{
      .commandPool = MCommandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = SMaxFramesInFlight};
  MGraphicsCommandBuffers = vk::raii::CommandBuffers(MDevice, allocInfo);
}

namespace {
// Transitions an image to an appropriate layout for future command-buffer
// commands.
void transitionImageLayout(vk::raii::CommandBuffer &commandBuffer,
                           vk::Image image, vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout,
                           vk::AccessFlags2 srcAccessMask,
                           vk::AccessFlags2 dstAccessMask,
                           vk::PipelineStageFlags2 srcStageMask,
                           vk::PipelineStageFlags2 dstStageMask) {
  // Define image memory barrier
  vk::ImageMemoryBarrier2 barrier = {
      .srcStageMask = srcStageMask,
      .srcAccessMask = srcAccessMask,
      .dstStageMask = dstStageMask,
      .dstAccessMask = dstAccessMask,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};

  vk::DependencyInfo dependency_info = {.dependencyFlags = {},
                                        .imageMemoryBarrierCount = 1,
                                        .pImageMemoryBarriers = &barrier};
  // vkCmdPipelineBarrier2 Insert a memory dependency between commands defined
  // before and after
  commandBuffer.pipelineBarrier2(dependency_info);
}
// Transfers ownership of a buffer backed by memory shared with another
// Vulkan device between that device and the queue family of this one.
//
// Buffers are created with `VK_SHARING_MODE_EXCLUSIVE`, so ownership has to be
// transferred explicitly to make writes of the releasing device visible. Both
// devices use `VK_QUEUE_FAMILY_EXTERNAL` for the other side, as the queue
// family indices of a foreign device are not meaningful here.
void transferBufferOwnership(vk::raii::CommandBuffer &commandBuffer,
                             vk::Buffer buffer, uint32_t srcQueueFamilyIndex,
                             uint32_t dstQueueFamilyIndex,
                             vk::AccessFlags2 srcAccessMask,
                             vk::AccessFlags2 dstAccessMask,
                             vk::PipelineStageFlags2 srcStageMask,
                             vk::PipelineStageFlags2 dstStageMask) {
  vk::BufferMemoryBarrier2 barrier = {
      .srcStageMask = srcStageMask,
      .srcAccessMask = srcAccessMask,
      .dstStageMask = dstStageMask,
      .dstAccessMask = dstAccessMask,
      .srcQueueFamilyIndex = srcQueueFamilyIndex,
      .dstQueueFamilyIndex = dstQueueFamilyIndex,
      .buffer = buffer,
      .offset = 0,
      .size = vk::WholeSize};
  vk::DependencyInfo dependencyInfo = {.bufferMemoryBarrierCount = 1,
                                       .pBufferMemoryBarriers = &barrier};
  commandBuffer.pipelineBarrier2(dependencyInfo);
}
} // anonymous namespace

void vkParticle::recordGraphicsCommandBuffer(uint32_t imageIndex) {
  MGraphicsCommandBuffers[MCurrentFrame].reset();

  // Don't need to set one-time-submit, simultanteous-ues, or render-pass flags
  MGraphicsCommandBuffers[MCurrentFrame].begin({});

  // Before starting rendering, transition the swapchain image to
  // optimal color attachment
  transitionImageLayout(
      MGraphicsCommandBuffers[MCurrentFrame], MSwapChainImages[imageIndex],
      vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
      {}, // srcAccessMask (no need to wait for previous operations)
      vk::AccessFlagBits2::eColorAttachmentWrite, // dstAccessMask
      // top of the pipe is always the first stage
      vk::PipelineStageFlagBits2::eTopOfPipe, // srcStage
      // Color attachment output specifies the stage of the pipeline where final
      // color values are output from the pipeline.
      vk::PipelineStageFlagBits2::eColorAttachmentOutput // dstStage
  );

  // Dynamic rendering setup
  vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  vk::RenderingAttachmentInfo attachmentInfo = {
      .imageView = MSwapChainImageViews[imageIndex],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      // loadOp - what to do with image before rendering -> clear to black
      .loadOp = vk::AttachmentLoadOp::eClear,
      // storeOp - what to do with image after rendering -> store frame
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clearColor};
  vk::RenderingInfo renderingInfo = {
      // renderArea - size of the render area
      .renderArea = {.offset = {0, 0}, .extent = MSwapChainExtent},
      .layerCount = 1, // number of layers to render to
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachmentInfo};
  // Acquire the particle buffer from the device of the SYCL context, which
  // has just written it. Semaphores alone are not enough to make those writes
  // visible here, as the buffer is owned exclusively.
  transferBufferOwnership(
      MGraphicsCommandBuffers[MCurrentFrame],
      MShaderStorageBuffers[MCurrentFrame], vk::QueueFamilyExternal,
      MQueueIndex, {},                           // srcAccessMask (acquire)
      vk::AccessFlagBits2::eVertexAttributeRead, // dstAccessMask
      vk::PipelineStageFlagBits2::eTopOfPipe,    // srcStage
      vk::PipelineStageFlagBits2::eVertexAttributeInput // dstStage
  );

  MGraphicsCommandBuffers[MCurrentFrame].beginRendering(renderingInfo);

  // Bind command-buffer to graphics pipeline
  MGraphicsCommandBuffers[MCurrentFrame].bindPipeline(
      vk::PipelineBindPoint::eGraphics, *MGraphicsPipeline);
  // Set dynamic viewport and scissor state to full swapchain dimensions
  MGraphicsCommandBuffers[MCurrentFrame].setViewport(
      0, vk::Viewport(0.0f, 0.0f, static_cast<float>(MSwapChainExtent.width),
                      static_cast<float>(MSwapChainExtent.height), 0.0f, 1.0f));
  MGraphicsCommandBuffers[MCurrentFrame].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), MSwapChainExtent));

  // Bind command-buffer to buffer with GPU visible data used for vertex buffer
  // input.
  MGraphicsCommandBuffers[MCurrentFrame].bindVertexBuffers(
      0, {MShaderStorageBuffers[MCurrentFrame]}, {0});

  // Draw each of our particles, without using an index buffer as we're using
  // dots for vertices rather than triangles
  MGraphicsCommandBuffers[MCurrentFrame].draw(SParticleCount, 1,
                                              0 /* offset into SV_VertexId*/,
                                              0 /* offset into SV_InstanceID*/);
  MGraphicsCommandBuffers[MCurrentFrame].endRendering();

  // Release the particle buffer back to the device of the SYCL context, which
  // will overwrite it once the semaphore signalled by this submission
  // reaches the value recorded for this frame.
  transferBufferOwnership(
      MGraphicsCommandBuffers[MCurrentFrame],
      MShaderStorageBuffers[MCurrentFrame], MQueueIndex,
      vk::QueueFamilyExternal,
      vk::AccessFlagBits2::eVertexAttributeRead, // srcAccessMask
      {},                                        // dstAccessMask (release)
      vk::PipelineStageFlagBits2::eVertexAttributeInput, // srcStage
      vk::PipelineStageFlagBits2::eBottomOfPipe          // dstStage
  );

  // After rendering, transition the swapchain image to present src layout
  transitionImageLayout(
      MGraphicsCommandBuffers[MCurrentFrame], MSwapChainImages[imageIndex],
      vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
      vk::AccessFlagBits2::eColorAttachmentWrite, // srcAccessMask
      {},                                         // dstAccessMask
      // Color attachment output specifies the stage of the pipeline where final
      // color values are output from the pipeline.
      vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
      // Bottom of the pipe is always the last stage
      vk::PipelineStageFlagBits2::eBottomOfPipe // dstStage
  );
  MGraphicsCommandBuffers[MCurrentFrame].end();
}
