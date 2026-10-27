// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <unistd.h>

namespace {
/// @brief Creates the initial state of the particle system.
/// @param[in] count Number of particles to create.
/// @param[in] aspectRatio Height over width of the window.
std::vector<Particle> createParticles(uint32_t count, float aspectRatio) {
  // Setup random distribution to use for particle initial locations
  std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
  std::uniform_real_distribution rndDist(0.0f, 1.0f);

  std::vector<Particle> particles(count);
  for (auto &particle : particles) {
    // Initial particle positions on a circle
    float r = 0.25f * sqrtf(rndDist(rndEngine));
    float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
    float x = r * cosf(theta) * aspectRatio;
    float y = r * sinf(theta);
    particle.position = glm::vec2(x, y);
    particle.velocity = normalize(glm::vec2(x, y)) * 0.00025f;
    particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
                               rndDist(rndEngine), 1.0f);
  }
  return particles;
}

/// @brief Owner of a file descriptor, so that it is closed if importing it
/// fails. Vulkan takes ownership of the descriptor once imported successfully,
/// at which point ownership here is dropped with `release()`.
class FileDescriptor {
public:
  explicit FileDescriptor(int fd) : MFd(fd) {}
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  ~FileDescriptor() {
    if (MFd >= 0) {
      close(MFd);
    }
  }

  int get() const { return MFd; }
  void release() { MFd = -1; }

private:
  int MFd;
};
} // anonymous namespace

void VkSyclParticle::createShaderStorageBuffers() {
  assert(MSyclQueue.has_value());

  const std::vector<Particle> particles = createParticles(
      SParticleCount, static_cast<float>(SWindowHeight) / SWindowWidth);
  const vk::DeviceSize bufferSize = sizeof(Particle) * SParticleCount;

  MShaderStorageBuffers.clear();
  MShaderStorageBuffersMemory.clear();

  for (size_t i = 0; i < SMaxFramesInFlight; i++) {
    // Allocate the particles of this frame as device USM AdaptiveCpp is able
    // to export, and upload the initial state to it. Every frame reads the
    // particles of the previous frame, so all of them are initialized.
    MParticles[i] = sycl::malloc_device<Particle>(
        SParticleCount, *MSyclDevice, *MSyclContext,
        sycl::property_list{sycl::property::usm::AdaptiveCpp_exportable{}});
    if (!MParticles[i]) {
      throw std::runtime_error("failed to allocate particles as device USM!");
    }
    MSyclQueue->memcpy(MParticles[i], particles.data(), bufferSize).wait();

    // Export the device memory backing the allocation, so that a buffer of
    // the VkDevice of this application can be bound to it. The file
    // descriptor returned is consumed by a successful import below.
    auto exported =
        sycl::vk_interop::export_memory(MParticles[i], *MSyclContext);
    if (exported.fd < 0) {
      throw std::runtime_error("failed to export the SYCL particle memory!");
    }
    FileDescriptor fd{exported.fd};

    // The buffer is created with the usage this application needs, which is
    // what makes sharing through external memory preferable to sharing the
    // VkBuffer itself: AdaptiveCpp creates its own buffer with only transfer
    // and device address usage, and that does not constrain the buffer here.
    vk::ExternalMemoryBufferCreateInfo externalBufferInfo{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
    vk::BufferCreateInfo bufferInfo{.pNext = &externalBufferInfo,
                                    .size = bufferSize,
                                    .usage =
                                        vk::BufferUsageFlagBits::eVertexBuffer,
                                    .sharingMode = vk::SharingMode::eExclusive};
    vk::raii::Buffer buffer(MDevice, bufferInfo);

    // The memory type of the allocation is only meaningful on the physical
    // device it was made on, which is also the one this application uses, but
    // the buffer created here may have stricter requirements than the buffer
    // AdaptiveCpp created for the same memory.
    const vk::MemoryRequirements memRequirements =
        buffer.getMemoryRequirements();
    if (!(memRequirements.memoryTypeBits &
          (1u << exported.memory_type_index))) {
      throw std::runtime_error(
          "the exported SYCL memory is of a type that cannot back a vertex "
          "buffer!");
    }
    if (memRequirements.size > exported.allocation_size) {
      throw std::runtime_error("the exported SYCL memory is too small!");
    }

    vk::ImportMemoryFdInfoKHR importInfo{
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd,
        .fd = fd.get()};
    vk::MemoryAllocateInfo allocInfo{.pNext = &importInfo,
                                     .allocationSize = exported.allocation_size,
                                     .memoryTypeIndex =
                                         exported.memory_type_index};
    vk::raii::DeviceMemory memory(MDevice, allocInfo);
    fd.release();

    // An exportable allocation owns the whole of its VkDeviceMemory, which
    // AdaptiveCpp binds its own buffer to at offset zero.
    assert(exported.offset == 0);
    buffer.bindMemory(memory, exported.offset);

    MShaderStorageBuffers.emplace_back(std::move(buffer));
    MShaderStorageBuffersMemory.emplace_back(std::move(memory));
  }
}
