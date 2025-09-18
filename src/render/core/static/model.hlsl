/// The most basic model shader possible

#include <anim/common.hlsl>

struct VSOut {
    float4 position : SV_POSITION;
    [[vk::location(0)]] float3 color : COLOR0;
};

/// Push constant with world_view_projection matrix (temp sh*t)
[[vk::push_constant]]
struct PushConstant {
    /// WVP matrix
    float4x4 world_view_projection;
} push_constant;

/// Vertex shader main
VSOut vs_main(anim::CommonVertex vertex) {
    VSOut result;

    result.position = mul(push_constant.world_view_projection, float4(vertex.position, 1.0));
    result.color = normalize(anim::octmap::unpack(vertex.normal));

    return result;
}

/// Fragment shader main
float4 fs_main(VSOut input): SV_TARGET0 {
    return float4((normalize(input.color) + 1.0) / 2.0, 0.0);
}

/// Depth-only vertex shader main
float4 vs_depth_main(anim::CommonVertex vertex): SV_POSITION {
    return mul(push_constant.world_view_projection, float4(vertex.position, 1.0));
}
