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

/// Unpack octmapped unit vector (without renormalization, thus)
float3 octmap_unpack(float2 packed) {
    float3 n = float3(packed.x, packed.y, 1.0 - abs(packed.x) - abs(packed.y));
	float t = saturate(-n.z);
	n.xy += select(n.xy >= 0.0, t, -t);
	return n;
}

/// Pack octmapped vector
float2 octmap_pack(float3 u) {
    // Normalize by Manhattan norm
    u /= (abs(u.x) + abs(u.y) + abs(u.z));

    // Perform octahedral wrap for lower octahedron part
    return u.z >= 0.0
        ? u.xy
        : ((1.0 - abs(u.yx)) * select(u.xy >= 0.0, -1.0, 1.0));
}

/// Vertex shader main
VSOut vs_main(Vertex vertex) {
    VSOut result;

    result.position = mul(push_constant.world_view_projection, float4(vertex.position, 1.0));
    result.color = normalize(octmap_unpack(vertex.normal));

    return result;
}

/// Fragment shader main
float4 fs_main(VSOut input): SV_TARGET0 {
    return float4((normalize(input.color) + 1.0) / 2.0, 0.0);
}
