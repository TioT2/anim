/// Matrix compute shader

#include <anim/common.hlsl>

// Input world matirces
StructuredBuffer<float4x4> input_matrices : register(t0);

// Output matrices
RWStructuredBuffer<anim::MatrixBufferItem> output_matrices : register(u1);

// Camera buffer. Just camera buffer.
cbuffer CameraBuffer : register(b2) {
    anim::CameraBuffer camera_buffer;
}

/// Calculate inverse of the 3x3 matrix
float3x3 inverse_float_3x3(float3x3 m) {
    // Adjoint matrix (copied from WGSL shader, so...)
    float3x3 adj = {
        m[1].y * m[2].z - m[2].y * m[1].z,
        m[2].y * m[0].z - m[0].y * m[2].z,
        m[0].y * m[1].z - m[1].y * m[0].z,

        m[2].x * m[1].z - m[1].x * m[2].z,
        m[0].x * m[2].z - m[2].x * m[0].z,
        m[1].x * m[0].z - m[0].x * m[1].z,

        m[1].x * m[2].y - m[2].x * m[1].y,
        m[2].x * m[0].y - m[0].x * m[2].y,
        m[0].x * m[1].y - m[1].x * m[0].y,
    };

    return adj * (1.0 / dot(adj[0], float3(m[0].x, m[1].x, m[2].x)));
}

/// Compute shader main
[numthreads(1, 1, 1)]
void cs_main(uint3 thread_id: SV_DispatchThreadID) {
    float4x4 world = input_matrices[thread_id.x];

    // Combine calculation with 'copying' to avoid binding of
    // slow host-visible matrix buffer to the render shaders
    output_matrices[thread_id.x].world = world;
    output_matrices[thread_id.x].world_view_projection = mul(camera_buffer.view_projection, world);
    output_matrices[thread_id.x].world_inverse = inverse_float_3x3(world);
}
