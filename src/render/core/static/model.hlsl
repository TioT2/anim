/// Ubershader (I hope it will become...)

#include <anim/common.hlsl>

// Binding to buffer that holds instanced object matrices
StructuredBuffer<anim::MatrixBufferItem> matrix_buffer : register(t0);

// Binding to buffer that holds camera data
cbuffer CameraBuffer : register(b1) {
    /// Camera buffer structure
    anim::CameraBuffer camera_buffer;
}

/// Push constant with world_view_projection matrix (temp sh*t)
[[vk::push_constant]]
struct {
    /// Offset of the matrix in the matrix buffer
    uint matrix_offset;
} push_constant;

/// Vertex shader output
struct VSOut {
    /// Device-space position
    float4 position : SV_POSITION;

    /// World-space vertex normal
    [[vk::location(0)]] float3 normal : NORMAL0;
};

/// Vertex shader main
VSOut vs_main(anim::CommonVertex vertex) {
    const uint matrix_offset = push_constant.matrix_offset;
    VSOut result;

    // Calculate position
    result.position = mul(
        matrix_buffer[matrix_offset].world_view_projection,
        float4(vertex.position, 1.0)
    );

    // Calculate vertex normal
    result.normal = normalize(mul(
        normalize(anim::octmap::unpack(vertex.normal)),
        matrix_buffer[matrix_offset].world_inverse
    ));

    return result;
}

/// Fragment shader main
float4 fs_main(VSOut input): SV_TARGET0 {
    return float4((normalize(input.normal) + 1.0) / 2.0, 0.0);
}

/// Depth-only vertex shader main
float4 vs_depth_main(anim::CommonVertex vertex): SV_POSITION {
    return mul(
        matrix_buffer[push_constant.matrix_offset].world_view_projection,
        float4(vertex.position, 1.0)
    );
}
