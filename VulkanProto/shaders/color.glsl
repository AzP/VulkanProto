@common
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

@vert
layout (std140, binding = 0) uniform bufferVals {
    mat4 mvp;
} myBufferVals;
layout (location = 0) in vec4 pos;
layout (location = 1) in vec4 inColor;
layout (location = 0) out vec4 outColor;

void main() {
    outColor = inColor;
    gl_Position = myBufferVals.mvp * pos;
}

@frag
layout (location = 0) in vec4 color;
layout (location = 0) out vec4 outColor;

void main() {
    outColor = color;
}
