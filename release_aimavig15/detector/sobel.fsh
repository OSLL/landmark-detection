#version 100

precision mediump float;

varying vec2 UV;

uniform sampler2D inputTexture;
uniform vec2 size;
uniform bool horizontal;

vec2 normH(float x, float y) {
	return vec2((x + 1.0) / 2.0, y / 4.0);
}

vec2 denormH(vec2 v) {
	return vec2(v.x * 2.0 - 1.0, v.y * 4.0);
}

float norm(float x) {
	return (x + 4.0) / 8.0;
}

void main() {
/*
	float sum = 0.0;
	sum += -1.0 * texture2D(inputTexture, UV + vec2(-1.0 / size.x, -1.0 / size.y)).x;
	sum +=  1.0 * texture2D(inputTexture, UV + vec2(1.0 / size.x, -1.0 / size.y)).x;
	sum += -2.0 * texture2D(inputTexture, UV + vec2(-1.0 / size.x, 0.0)).x;
	sum +=  2.0 * texture2D(inputTexture, UV + vec2(1.0 / size.x, 0.0)).x;
	sum += -1.0 * texture2D(inputTexture, UV + vec2(-1.0 / size.x, 1.0 / size.y)).x;
	sum +=  1.0 * texture2D(inputTexture, UV + vec2(1.0 / size.x, 1.0 / size.y)).x;

	sum += -3.0 * texture2D(inputTexture, UV + vec2(-1.0 / size.x, -1.0 / size.y)).x;
	sum += -10.0 * texture2D(inputTexture, UV + vec2(0.0, -1.0 / size.y)).x;
	sum += -3.0 * texture2D(inputTexture, UV + vec2(1.0 / size.x, -1.0 / size.y)).x;
	sum += 3.0 * texture2D(inputTexture, UV + vec2(-1.0 / size.x, 1.0 / size.y)).x;
	sum += 10.0 * texture2D(inputTexture, UV + vec2(0.0, 1.0 / size.y)).x;
	sum += 3.0 * texture2D(inputTexture, UV + vec2(1.0 / size.x, 1.0 / size.y)).x;
	gl_FragColor = vec4(vec3(sum), 1.0);
*/
	
	if(horizontal) {
		float offset = 1.0 / size.x;
		vec2 tl = texture2D(inputTexture, vec2(UV.x - offset, UV.y)).xy;
		vec2 tc = texture2D(inputTexture, UV).xy;
		vec2 tr = texture2D(inputTexture, vec2(UV.x + offset, UV.y)).xy;
		float gx = -tl.x + tr.x;
		float gy = tl.y + 2.0 * tc.y + tr.y;
		gl_FragColor = vec4(normH(gx, gy), 0.0, 1.0);
	} else {
		float offset = 1.0 / size.y;
		vec2 tt = denormH(texture2D(inputTexture, vec2(UV.x, UV.y - offset)).xy);
		vec2 tc = denormH(texture2D(inputTexture, UV).xy);
		vec2 tb = denormH(texture2D(inputTexture, vec2(UV.x, UV.y + offset)).xy);
		float gx = tt.x + 2.0 * tc.x + tb.x;
		float gy = -tt.y + tb.y;
		float g = sqrt(gx * gx + gy * gy)/* / 5.656854*/;
		gl_FragColor = vec4(norm(gx), norm(gy), g, 1.0);
	}
}
