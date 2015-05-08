#version 100

precision mediump float;

varying vec2 UV;

uniform sampler2D inputTexture;
uniform vec2 size;
uniform float threshold;

void main() {
	vec3 val = texture2D(inputTexture, UV).rgb;
	float vmax = max(val.r, max(val.g, val.b));
	float vmin = min(val.r, min(val.g, val.b));
	float sat = vmax == 0.0 ? 0.0 : ((vmax - vmin) / vmax);
	sat = sat > threshold ? sat : 0.0;
	
    gl_FragColor = vec4(sat, sat, sat, 1.0);
}
