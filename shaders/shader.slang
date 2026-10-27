// Copyright (c) 2025-2026 Ewan Crawford

struct VertexShaderInput {
  float2 inPosition; // (x,y) position attribute
  float4 inColor;    // (r,g,b,a) color attribute
};

struct VertexShaderOutput {
  float4 pos : SV_Position; // SV prefix means System Value in HLSL
  float pointSize : SV_PointSize; // Like 'gl_PointSize'
  float3 fragColor : COLOR0; // COLOR0 is diffuse lighting
};

struct FragShaderInput {
  float4 pos : SV_POSITION;
  float3 fragColor : COLOR0;
  float2 pointCoord : SV_PointCoord; // Like 'gl_PointCoord'
};

[shader("vertex")] VertexShaderOutput vertMain(VertexShaderInput input) {
  VertexShaderOutput output;
  output.pointSize = 14.0;
  output.pos = float4(input.inPosition, 1.0, 1.0);
  output.fragColor = input.inColor.rgb;
  return output;
}

[shader("fragment")]
float4 fragMain(FragShaderInput input) : SV_Target {
  float2 coord = input.pointCoord - float2(.5);
  return float4(input.fragColor, .5 - length(coord));
}

// NOTE: The particle simulation that used to be the `compMain` compute shader
// here is now a SYCL kernel, see `src/compute.cpp`. The particle buffer it
// writes is shared with this shader as a vertex buffer through Vulkan
// external memory.
