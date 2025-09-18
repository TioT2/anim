/// Common definition file

#ifndef ANIM_COMMON_HLSL_
#define ANIM_COMMON_HLSL_

namespace anim {
    /// Buffer that resembles renderer camera state
    struct CameraBuffer {
        /// View-projection matrix
        float4x4 view_projection;

        /// View matrix
        float4x4 view;

        /// Projection matrix
        float4x4 projection;

        /// Forward direction
        float3 dir_forward;

        /// Right direction
        float3 dir_right;

        /// Up direction
        float3 dir_up;

        /// Current location
        float3 location;
    };

    /// Instance buffer
    struct InstanceBuffer {
        /// Color texture
        uint32_t texture_color;

        /// Metallic-roughness texture
        uint32_t texture_metallic_roughness;

        /// Offset of the instance in matrix buffer
        uint32_t matrix_buffer_offset;
    };

    /// Buffer that contains all matrices required
    struct MatrixBufferItem {
        /// World matrix
        float4x4 world;

        /// World-view-projection matrix
        float4x4 world_view_projection;

        /// Inversed and stripped world matrix
        float3x3 world_inverse;
    };

    /// Common vertex structure
    struct CommonVertex {
        [[vk::location(0)]] float3 position : POSITION0;
        [[vk::location(1)]] float2 tex_coord : TEXCOORD0;
        [[vk::location(2)]] float2 normal : NORMAL0;
        [[vk::location(3)]] float2 tangent : TANGENT0;
        [[vk::location(4)]] float4 misc : COLOR0;
    };

    // Octmap implementation
    namespace octmap {
        /// Unpack octmapped unit vector (without renormalization, thus)
        float3 unpack(float2 packed) {
            float3 n = float3(packed.x, packed.y, 1.0 - abs(packed.x) - abs(packed.y));
       	float t = saturate(-n.z);
       	n.xy += select(n.xy >= 0.0, t, -t);
       	return n;
        }

        /// Pack octmapped vector
        float2 pack(float3 u) {
            // Normalize by Manhattan norm
            u /= (abs(u.x) + abs(u.y) + abs(u.z));

            // Perform octahedral wrap for lower octahedron part
            return u.z >= 0.0
                ? u.xy
                : ((1.0 - abs(u.yx)) * select(u.xy >= 0.0, -1.0, 1.0));
        }
    }
}

#endif // !defined(ANIM_COMMON_HLSL_)
