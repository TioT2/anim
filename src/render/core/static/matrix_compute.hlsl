/// Matrix compute shader

#include <anim/common.hlsl>

StructuredBuffer<float4x4> input_matrices : register(u0);
RWStructuredBuffer<MatrixBufferItem> output_matrices : register(u1);

[numthreads(1, 1, 1)]
void cs_main() {

}
