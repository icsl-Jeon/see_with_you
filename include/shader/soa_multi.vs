#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aSoa;

uniform mat4 world;
uniform mat4 model;
uniform mat4 views[100];
uniform mat4 projection;

out vec3 renderColor;

void main(){
    gl_Position = projection * views[gl_InstanceID] * model * world *vec4(aPos, 1.0);
    if (aSoa == 1.0) { renderColor = vec3(1.0,0.0,0.0); }
    else { renderColor = vec3(0.0,1.0,0.0); }
}