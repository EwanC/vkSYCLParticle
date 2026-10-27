// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"
#include <iostream>

// Storage for the vulkan-hpp dynamic dispatcher, which must be defined in
// exactly one translation unit when VULKAN_HPP_DISPATCH_LOADER_DYNAMIC is set.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

int main() {
  vkParticle app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
