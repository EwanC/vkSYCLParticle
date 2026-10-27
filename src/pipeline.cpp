// Copyright (c) 2025-2026 Ewan Crawford

#include "common.hpp"

void vkParticle::createGraphicsPipeline() {
  // Setup vertex & fragment shaders
  vk::raii::ShaderModule shaderModule =
      createShaderModule(readFile("slang.spv"), MDevice);

  vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = shaderModule,
      .pName = "vertMain"};
  vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = shaderModule,
      .pName = "fragMain"};
  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

  // Defines the stride between vertex shader input elements
  auto bindingDescription = Particle::getBindingDescription();
  // Defines how the individual elements in the vertex shader input struct
  // are laid out.
  auto attributeDescriptions = Particle::getAttributeDescriptions();
  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescription,
      .vertexAttributeDescriptionCount =
          static_cast<uint32_t>(attributeDescriptions.size()),
      .pVertexAttributeDescriptions = attributeDescriptions.data()};

  // Each vertex is a point rather than a triangle, to represent a particle
  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
      .topology = vk::PrimitiveTopology::ePointList,
      .primitiveRestartEnable = vk::False};

  // Viewport describes the rectangular region of the framebuffer that the
  // output image will be rendered to,
  // with any pixels outside the scissor rectangle being discarded.
  vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1,
                                                    .scissorCount = 1};
  // viewport and scissor can be changed without recreating the pipeline
  std::vector dynamicStates = {vk::DynamicState::eViewport,
                               vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()};

  vk::PipelineRasterizationStateCreateInfo rasterizer{
      // discard fragments beyond near/far planes
      .depthClampEnable = vk::False,
      // If set to true when geometry never passes through rasterizer,
      // not what we want
      .rasterizerDiscardEnable = vk::False,
      // Fill the polygon area with fragments, rather than point or line
      .polygonMode = vk::PolygonMode::eFill,
      // backface culling
      .cullMode = vk::CullModeFlagBits::eBack,
      // interprets how a polygon is decided as being front facing
      // Shouldn't matter here as we're drawing points, not triangles
      .frontFace = vk::FrontFace::eClockwise,
      // Whether to add an offset to depth value
      .depthBiasEnable = vk::False,
      // scalar factor applied to a fragment’s slope in depth bias calculations.
      .depthBiasSlopeFactor = 1.0f,
      // Thickness of line in fragments, larger than 1.0 requires 'wideline'
      // GPU feature.
      .lineWidth = 1.0f};

  // Multi-sampling combines the fragment shader results of multiple polygons
  // that rasterize to the same pixel, to help avoid aliasing. Disabled here.
  vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = vk::False};

  // Color blending step of the pipeline runs after fragment shading to combine
  // with color already in framebuffer.
  vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable = vk::True, // enable blending
      // Alpha blending - colors are blended based on opacity
      .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
      .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .dstAlphaBlendFactor = vk::BlendFactor::eZero,
      .alphaBlendOp = vk::BlendOp::eAdd,
      // Blend RGBA channels
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  // Global color blend settings
  vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment};

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
  MPipelineLayout = vk::raii::PipelineLayout(MDevice, pipelineLayoutInfo);

  vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &MSwapChainSurfaceFormat.format};
  vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext = &pipelineRenderingCreateInfo,
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = *MPipelineLayout,
      .renderPass = nullptr};

  MGraphicsPipeline = vk::raii::Pipeline(MDevice, nullptr, pipelineInfo);
}
