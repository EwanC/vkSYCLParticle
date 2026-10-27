// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"

sycl::event vkParticle::updateParticles(float deltaTime) {
  assert(MSyclQueue.has_value());

  // Particles of the previous frame are read, and the particles of the
  // current frame written, so that the graphics work of the previous frame
  // may still be reading from its own buffer.
  const uint32_t previousFrame = (MCurrentFrame + 1) % SMaxFramesInFlight;
  const Particle *particlesIn = MParticles[previousFrame];
  Particle *particlesOut = MParticles[MCurrentFrame];

  return MSyclQueue->submit([&](sycl::handler &cgh) {
    // The buffer written to was last read as a vertex buffer by the graphics
    // work of a previous frame, so the kernel is made to wait until that has
    // completed. `MSemaphore` is signalled by the graphics queue of this
    // application, and imported into the SYCL context as `MSyclWaitSemaphore`
    // so that the wait happens on the device rather than on the host.
    const uint64_t waitValue = MFrameGraphicsValue[MCurrentFrame];
    if (waitValue != 0) {
      cgh.depends_on(sycl::vk_interop::make_event(MSyclWaitSemaphore, waitValue,
                                                  *MSyclContext));
    }

    const sycl::nd_range<1> range{sycl::range<1>{SParticleCount},
                                  sycl::range<1>{SComputeWorkItems}};
    cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
      const size_t index = item.get_global_id(0);
      const Particle &in = particlesIn[index];

      float positionX = in.position.x + in.velocity.x * deltaTime;
      float positionY = in.position.y + in.velocity.y * deltaTime;
      float velocityX = in.velocity.x;
      float velocityY = in.velocity.y;

      // Flip movement at window border
      if (positionX <= -1.0f || positionX >= 1.0f) {
        velocityX = -velocityX;
      }
      if (positionY <= -1.0f || positionY >= 1.0f) {
        velocityY = -velocityY;
      }

      Particle &out = particlesOut[index];
      out.position.x = positionX;
      out.position.y = positionY;
      out.velocity.x = velocityX;
      out.velocity.y = velocityY;
      out.color = in.color;
    });
  });
}
