#version 100

precision mediump float;

varying vec2 UV;

uniform sampler2D inputTexture;
uniform vec2 size;
uniform float radius;
uniform bool horizontal;

void main() {
	float blur = 0.0, hstep = 0.0, vstep = 0.0;
	
	if(horizontal) {
		hstep = 1.0;
		blur = radius / size.x;
	} else {
		vstep = 1.0;
		blur = radius / size.y;
	}

	vec4 sum = texture2D(inputTexture, UV) * 0.132981;
	sum += texture2D(inputTexture, vec2(UV.x - 1.0 * blur * hstep, UV.y - 1.0 * blur * vstep)) * 0.125794;
	sum += texture2D(inputTexture, vec2(UV.x - 2.0 * blur * hstep, UV.y - 2.0 * blur * vstep)) * 0.106483;
	sum += texture2D(inputTexture, vec2(UV.x - 3.0 * blur * hstep, UV.y - 3.0 * blur * vstep)) * 0.080657;
	sum += texture2D(inputTexture, vec2(UV.x - 4.0 * blur * hstep, UV.y - 4.0 * blur * vstep)) * 0.054670;
	sum += texture2D(inputTexture, vec2(UV.x + 1.0 * blur * hstep, UV.y + 1.0 * blur * vstep)) * 0.125794;
	sum += texture2D(inputTexture, vec2(UV.x + 2.0 * blur * hstep, UV.y + 2.0 * blur * vstep)) * 0.106483;
	sum += texture2D(inputTexture, vec2(UV.x + 3.0 * blur * hstep, UV.y + 3.0 * blur * vstep)) * 0.080657;
	sum += texture2D(inputTexture, vec2(UV.x + 4.0 * blur * hstep, UV.y + 4.0 * blur * vstep)) * 0.054670;
	
    gl_FragColor = vec4(sum.rgb, 1.0);
}
