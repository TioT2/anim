/// The most basic model shader possible

#include <anim/common.hlsl>

StructuredBuffer<anim::MatrixBufferItem> matrix_buffer : register(t0);
cbuffer CameraBuffer : register(b1) {
    anim::CameraBuffer camera_buffer;
}

/// Push constant with world_view_projection matrix (temp sh*t)
[[vk::push_constant]]
struct PushConstant {
    /// Offset of the matrix in the matrix buffer
    uint matrix_offset;
} push_constant;

struct VSOut {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float3 normal : NORMAL0;
};

/// Vertex shader main
VSOut vs_main(anim::CommonVertex vertex) {
    VSOut result;

    result.position = mul(
        matrix_buffer[push_constant.matrix_offset].world_view_projection,
        float4(vertex.position, 1.0)
    );
    result.normal = normalize(anim::octmap::unpack(vertex.normal));

    return result;
}

/// Fragment shader main
float4 fs_main(VSOut input): SV_TARGET0 {
    return float4((normalize(input.normal) + 1.0) / 2.0, 0.0);
}

/// Depth-only vertex shader main
float4 vs_depth_main(anim::CommonVertex vertex): SV_POSITION {
    return mul(matrix_buffer[push_constant.matrix_offset].world_view_projection, float4(vertex.position, 1.0));
}
