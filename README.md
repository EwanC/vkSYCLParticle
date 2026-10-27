# VK-SYCL Particle

Vulkan GPU particle simulation where the rendering is done with Vulkan
graphics, and the simulation with a SYCL kernel sharing data and
synchronization with it. Based on the
[Khronos Vulkan Compute Shader Tutorial](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html).

Modifications from tutorial include:
* The compute shader that updates the particles is replaced with a SYCL kernel
  run through the Vulkan backend of
  [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), see
  [SYCL interop](#sycl-interop).

## SYCL interop

AdaptiveCpp creates and owns its own `VkInstance` and `VkDevice`, which this
application cannot influence. Rather than sharing them, which would restrict
the layers, extensions, features and queues the renderer is able to ask for,
the two sides keep separate logical devices and share the Vulkan objects they
need to have in common through
[external memory](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_external_memory_fd.html)
and
[external semaphore](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_external_semaphore_fd.html)
opaque file descriptors.

* **Device selection** (`src/device.cpp`) — a `VkPhysicalDevice` handle is only
  meaningful within the instance it came from, so the physical device of this
  application and that of the SYCL device are matched on their
  `VkPhysicalDeviceIDProperties::deviceUUID`. Sharing memory by file descriptor
  requires both logical devices to originate from the same physical device.
* **Particle buffers** (`src/buffer.cpp`) — the particles of each frame in
  flight are allocated as SYCL device USM with the
  `sycl::property::usm::AdaptiveCpp_exportable` property, exported with
  `sycl::AdaptiveCpp_vk_interop::export_memory`, and the resulting file
  descriptor imported into the `VkDevice` of this application and bound to a
  vertex buffer. The buffer created here chooses its own usage flags, so the
  buffer AdaptiveCpp creates for the same memory does not constrain the
  renderer.
* **Synchronization** (`src/draw.cpp`, `src/interop.cpp`) — two timeline
  semaphores are shared in opposite directions. The semaphore AdaptiveCpp
  signals on kernel completion is imported here, and waited on by the graphics
  submission at the vertex input stage. The semaphore this application signals
  on completion of its graphics work is imported into the SYCL context, and
  waited on by the kernel of a later frame before it overwrites a buffer that
  is still being read.
* **Ownership transfers** (`src/command_buffer.cpp`) — the shared buffers are
  created with `VK_SHARING_MODE_EXCLUSIVE`, so the graphics command-buffer
  acquires each buffer from `VK_QUEUE_FAMILY_EXTERNAL` before reading it and
  releases it afterwards. Semaphores alone do not make the writes of the other
  device visible.

This relies on the Vulkan backend interop of AdaptiveCpp, which is only
available on Linux, as the opaque file descriptor handle type is POSIX only.

## Building

Tested on Ubuntu 24.04 and 26.04. Requires an AdaptiveCpp SYCL compiler, the Vulkan-SDK, and
AdaptiveCpp built with its Vulkan backend and the Vulkan backend interop API.
See
[Development Environment](https://docs.vulkan.org/tutorial/latest/02_Development_environment.html)
for more details on the Vulkan build setup.

Sample command line for building and running the application, the validation
layer is enabled in Debug builds:
```sh
$ mkdir build && cd build
$ cmake ../ -GNinja -DCMAKE_CXX_COMPILER=acpp -DACPP_TARGETS=generic
$ ninja
$ ACPP_VISIBILITY_MASK=vk ./vkSyclParticle
```

`ACPP_VISIBILITY_MASK=vk` restricts AdaptiveCpp to its Vulkan backend. Without
it a device of another backend may be chosen, which cannot share memory with
the Vulkan objects of this application.

Press Escape key to exit, or close window in GUI.

![capture](img/capture.gif)
