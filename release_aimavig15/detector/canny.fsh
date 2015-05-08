#version 100

precision mediump float;

varying vec2 UV;

uniform sampler2D inputTexture;
uniform vec2 size;

vec2 dir(float a) {
    if(a > -0.414213 && a < 0.414213) return vec2(1.0, 0.0);
    if(a < -2.414213 || a > 2.414213) return vec2(0.0, 1.0);
	if(a > 0.414213 && a < 2.414213) return vec2(1.0, 1.0);
    return vec2(1.0, -1.0);
}

void main() {
	vec3 val = texture2D(inputTexture, UV).xyz;
	
	vec2 d = dir((val.y - 0.5) / (val.x - 0.5)) * vec2(1.0 / size.x, 1.0 / size.y);
	float fwd = texture2D(inputTexture, UV + d).z;
	float bwd = texture2D(inputTexture, UV - d).z;
	if(fwd > val.z || bwd > val.z) {
		gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	} else {
		//if(val.z * 5.656854 > 3.45) gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        if(val.z > 0.75) gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
		//else if(val.z > 0.4) gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
		else gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	}
}
