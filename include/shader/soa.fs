#version 330 core
out vec4 FragColor;
in vec4 renderColor;
void main()
{
    FragColor = renderColor;
}
