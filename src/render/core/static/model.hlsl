/// The most basic model shader possible

/// ANIM vertex structure
struct Vertex {
    [[vk::location(0)]] float3 position : POSITION0;
    [[vk::location(1)]] float2 tex_coord : TEXCOORD0;
    [[vk::location(2)]] float2 normal : NORMAL0;
    [[vk::location(3)]] float2 tangent : TANGENT0;
    [[vk::location(4)]] float4 misc : COLOR0;
};

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
VSOut vs_main(Vertex vertex) {
    VSOut result;

    result.position = mul(push_constant.world_view_projection, vertex.position);
    result.color = float3(vertex.normal, 0.0);

    return result;
}

/// Fragment shader main
float4 fs_main(VSOut input): SV_TARGET0 {
    return float4(input.color, 0.0);
}
