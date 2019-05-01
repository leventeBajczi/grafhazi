//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, torus, mobius
// Camera: perspective
// Light: point
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 50;
class Ladybug;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
	Ladybug *follow;
public:
	Camera() {
		asp = (float)windowWidth/windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 10;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
											 u.y, v.y, w.y, 0,
											 u.z, v.z, w.z, 0,
											 0,   0,   0,   1);
	}
	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(float t);
};

//---------------------------
struct Material {
//---------------------------
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

float lengthsq(const vec4& obj)
{
	return obj.x*obj.x+obj.y*obj.y+obj.z*obj.z+obj.w*obj.w;
}

vec4 invert(const vec4& obj)
{
	return vec4{obj.x, -obj.y, -obj.z, -obj.w}/lengthsq(obj);
}

//---------------------------
struct Light {
//---------------------------
	vec3 La, Le;
	vec4 wLightPos;
	vec4 wLightPos_original;
	vec4 center;

	void Animate(float t) {
		vec4 quaternion{
			cosf(t/4),
			sinf(t/4)*cosf(t)/2,
			sinf(t/4)*sinf(t)/2,
			sinf(t/4)*sqrtf(3.0f/4)};
		vec4 light{0, wLightPos_original.x, wLightPos_original.y, wLightPos_original.z};
		light = quaternion * light * invert(quaternion);
		wLightPos = vec4{light.y, light.z, light.w, 0} + center;
	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		wLightPos.SetUniform(shaderProg, buffer);
	}
};

//---------------------------
struct CheckerBoardTexture : public Texture {
//---------------------------
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);
		const vec3 yellow(1, 1, 0), blue(0, 0, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

//---------------------------
struct DiniTexture : public Texture {
//---------------------------
	DiniTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);
		const vec3 red(1, 0, 0), black(0, 0, 0);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? red : black;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};
//---------------------------
struct LadyBugTexture : public Texture {
//---------------------------
	LadyBugTexture(const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);
		const vec3 red(1, 0, 0), black(0, 0, 0);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = 
				x < width/4 || x > width*3/4 ||
				x == width*4/10 && y % (height/5) == 0 && y > height / 5 && y <= height * 4 / 5 ||
				x == width*6/10 && y % (height/5) == 0 && y > height / 5 && y <= height * 4 / 5 ||
				x == width/2 && y == height/5
				? black : red;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//--------------------------
public:
	virtual void Bind(RenderState state) = 0;
};


//---------------------------
class PhongShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
class NPRShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.lights[0].wLightPos.SetUniform(getId(), "wLightPos");

		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
struct VertexData {
//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
//---------------------------
protected:
	unsigned int vao;        // vertex array object
public:
	Geometry( ) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Draw() = 0;
};

//---------------------------
class ParamSurface : public Geometry {
//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = tessellationLevel, int M = tessellationLevel) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position)); 
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw()	{
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
struct Clifford {
//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }


//---------------------------
class Ellipsoid : public ParamSurface {
//---------------------------
private:
	vec3 params{0.5,0.5,0.75};
public:
	Ellipsoid() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 2.0f * M_PI - M_PI;
		float V = v * M_PI - M_PI/2.0f;
		Clifford x = Cos(T(U)) * params.x*cosf(V);
		Clifford y = Sin(T(U)) * params.y*cosf(V);
		Clifford z = params.z * sinf(V);		
		Clifford Vx = Cos(T(V)) * cosf(U) * params.x;
		Clifford Vy = Cos(T(V)) * params.y*sinf(U);
		Clifford Vz = Sin(T(V)) * params.z;		
		vd.position = vec3(x.f, y.f, z.f);
		vec3 drdU(x.d, y.d, z.d);
		vec3 drdV(Vx.d, Vy.d, Vz.d);
		vd.normal = normalize(cross(drdU, drdV));
		if(vd.position.x < 0)
		{
			vd.position.x = 0;
			vd.normal = vec3(-1, 0, 0);
		}
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class KleinBottle : public ParamSurface {
//---------------------------
	float minX = 0, minY = 0, minZ = 0,
		  maxX = 0, maxY = 0, maxZ = 0;
public:
	vec4 getAABBCenter()
	{
		return vec4(
			(maxX + minX)/2,
			(maxY + minY)/2,
			(maxZ + minZ)/2,
			0);
	}
	KleinBottle() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = 2 * M_PI * u;
		float V = 2 * M_PI * v;
		float a = 6 * cosf(U)*(1+sinf(U));
		float b = 16* sinf(U);
		float c = 4 * (1-cosf(U)/2);

		Clifford x = M_PI < U <= 2*M_PI ?
			Cos(T(U))*(Sin(T(U))+1)*6 + (Cos(T(U))*-0.5f+1)*4*cosf(V+M_PI) :
			Cos(T(U))*(Cos(T(U))*-0.5f+1)*4*cosf(V) + Cos(T(U))*(Sin(T(U))+1)*6;
		Clifford y = M_PI < U <= 2*M_PI ?
			Sin(T(U)) * 16 :
			Sin(T(U))*(Cos(T(U))*-0.5f+1)*4*cosf(V) + Sin(T(U)) * 16;
		Clifford z = (Cos(T(U))*-0.5f+1)*4*sinf(V);
		Clifford Vx = M_PI < U <= 2*M_PI ? Cos(T(V)+M_PI)*c + a : Cos(T(V))*c*cosf(U) + a;
		Clifford Vy = M_PI < U <= 2*M_PI ? b : Cos(T(V))*c*sinf(U) + b;
		Clifford Vz = Sin(T(V))*c;	
		vd.position = vec3(x.f, y.f, z.f);
		if(x.f < minX) minX = x.f;
		if(y.f < minY) minY = y.f;
		if(z.f < minZ) minZ = z.f;
		if(x.f > maxX) maxX = x.f;
		if(y.f > maxY) maxY = y.f;
		if(z.f > maxZ) maxZ = z.f;
		vec3 drdU(x.d, y.d, z.d);
		vec3 drdV(Vx.d, Vy.d, Vz.d);
		vd.normal = normalize(cross(drdU, drdV));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class DiniSurface : public ParamSurface {
//---------------------------
private:
	float a = 1;
	float b = 0.15;
public:
	DiniSurface() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = 4 * M_PI * u;
		float V = v*0.99+0.01;
		Clifford x = Cos(T(U))*a*sinf(V);
		Clifford y = Sin(T(U))*a*sinf(V);
		Clifford z = T(U) * b + a*(cosf(V) + log(tanf(V/2))/log(M_E));
		Clifford Vx = Sin(T(V))*a*cosf(U);
		Clifford Vy = Sin(T(V))*a*sinf(U);
		Clifford Vz = (Cos(T(V)) + Log(Tan(T(V)/2))/log(M_E))*a + b*U;	
		vd.position = vec3(x.f, y.f, z.f);
		vec3 drdU(x.d, y.d, z.d);
		vec3 drdV(Vx.d, Vy.d, Vz.d);
		vd.normal = normalize(cross(drdU, drdV));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
struct Object {
//---------------------------
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis, normal;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tend) {  }
};

struct Ladybug : public Object
{
	using Object::Object;
	float distance = 4;
	KleinBottle* kleinBottle;
	Object* kleinBottle1;
	float angle = 0;
	virtual void Animate(float tend) override final{
		VertexData vd = kleinBottle->GenVertexData(tend*0.1*cosf(angle), tend*0.1*sinf(angle));
		translation = vd.position * kleinBottle1->scale;
		rotationAxis = cross(vd.normal, vec3(-1,0,0));
		rotationAngle = acosf(dot(vd.normal, vec3(1, 0, 0)));
		scale = vec3(0.25, 0.25, 0.25);
		normal = vd.normal;
	}
	virtual void Draw(RenderState state) override final{
		state.M = ScaleMatrix(scale) * RotationMatrix(-angle-M_PI/2, vec3(-1,0,0)) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * RotationMatrix(angle+M_PI/2, vec3(-1,0,0)) *ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

};

//---------------------------
class Scene {
//---------------------------
	std::vector<Object *> objects;
	Ladybug* ladybug;
public:
	Camera camera; // 3D camera
	std::vector<Light> lights;

	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();
		Shader * nprShader = new NPRShader();

		// Materials
		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		// Textures
		Texture * texture15x20 = new CheckerBoardTexture(15, 20);
		Texture * diniTexture = new DiniTexture(25, 25);
		Texture * ladyBugTexture = new LadyBugTexture(25, 25);

		Geometry * ellipsoid = new Ellipsoid();
		KleinBottle * kleinBottle = new KleinBottle();
		Geometry * diniSurface = new DiniSurface();


		Object * kleinBottle1 = new Object(phongShader, material0, texture15x20, kleinBottle);
		kleinBottle1->translation = vec3(0, 0, 0);
		kleinBottle1->rotationAxis = vec3(0, 3, 0);
		kleinBottle1->scale = vec3(0.1, 0.1, 0.1);
		objects.push_back(kleinBottle1);

		Object * diniSurface1 = new Object(phongShader, material0, diniTexture, diniSurface);
		diniSurface1->scale = vec3(0.15, 0.15, 0.15);
		VertexData vd = kleinBottle->GenVertexData(0.25, 0.75);
		diniSurface1->translation = vd.position * kleinBottle1->scale + vd.normal*0.5;
		diniSurface1->rotationAxis = cross(vec3(0,0,1),vd.normal);
		diniSurface1->rotationAngle = acosf(dot(vd.normal, vec3(0, 0, 1)));
		objects.push_back(diniSurface1);
		
		Object * diniSurface2 = new Object(phongShader, material0, diniTexture, diniSurface);
		diniSurface2->scale = vec3(0.15, 0.15, 0.15);
		vd = kleinBottle->GenVertexData(0.75, 0.25);
		diniSurface2->translation = vd.position * kleinBottle1->scale + vd.normal*0.5;
		diniSurface2->rotationAxis = cross(vec3(0,0,1),vd.normal);
		diniSurface2->rotationAngle = acosf(dot(vd.normal, vec3(0, 0, 1)));
		objects.push_back(diniSurface2);

		ladybug = new Ladybug(nprShader, material0, ladyBugTexture, ellipsoid);
		ladybug->distance = 4;
		ladybug->kleinBottle1 = kleinBottle1;
		ladybug->kleinBottle = kleinBottle;
		objects.push_back(ladybug);

		// Camera
		camera.follow = ladybug;
		camera.wEye = vec3(0, 0, 6);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(1);
		lights[0].wLightPos_original = vec4(10, 10, 10, 0);	// ideal point -> directional light source
		lights[0].La = vec3(1, 1, 1);
		lights[0].Le = vec3(3, 3, 3);
		lights[0].center = kleinBottle->getAABBCenter();

	}
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float tend) {
		camera.Animate(tend);
		for (int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		for (int i = 0; i < objects.size(); i++) { objects[i]->Animate(tend); }
	}
	void HandleKey(char key)
	{
		switch(key)
		{
			case 'a': ladybug->angle+=M_PI/8; break;
			case 's': ladybug->angle-=M_PI/8; break;
			case ' ': ladybug->distance=1.5;
			default:
			break;
		}
	}
};

Scene scene;

void Camera::Animate(float t) {
	vec4 front = vec4(0, 1, 0, 1) * RotationMatrix(follow->rotationAngle, follow->rotationAxis);
	wEye = follow->translation + follow->normal*follow->distance;
	wLookat = follow->translation;
	wVup = follow->translation + vec3(front.x, front.y, front.z);
}
 
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
	scene.HandleKey(key);
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t + Dt); 
	}
	glutPostRedisplay();
}