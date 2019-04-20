//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bajczi Levente
// Neptun : XAO5ER
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char * const vertexSource = R"(
    #version 330
    precision highp float;

    uniform vec3 wLookAt, wRight, wUp;

    layout(location = 0) in vec2 cCamWindowVertex;
    out vec3 p;

    void main() {
        gl_Position = vec4(cCamWindowVertex, 0, 1);
        p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
    }
)";

const char * const fragmentSource = R"(
    #version 330
    precision highp float;
    

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Ellipsoid {
		vec3 center;
		vec3 params;
	};

	struct Rectangle {
		vec4 params;
		vec2 z_bounds;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;
	};

	struct Ray {
		vec3 start, dir;
	};

	const int maxEllipsoids = 3;
	const int maxMirrors = 10;
	const float epsilon = 0.0001f;
	const float PI = 3.1415926536;

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[5];
	uniform int nEllipsoids;
	uniform int nMirrors;
	uniform int matType;
	uniform Ellipsoid ellipsoids[maxEllipsoids];
	uniform Rectangle rectangles[maxMirrors];

	in  vec3 p;
	out vec4 fragmentColor;

	Hit intersect(const Rectangle object, const Ray ray) {
		Hit hit;
		hit.t = -1;

		vec3 params3 = vec3(object.params.x, object.params.y, object.params.z);
		if(abs(dot(params3, ray.dir)) < epsilon) return hit;
		float t = -1 * dot(object.params, vec4(ray.start, 1))/dot(params3, ray.dir);
		float x = ray.start.x + ray.dir.x*t;
		float y = ray.start.y + ray.dir.y*t;
		float z = ray.start.z + ray.dir.z*t;
		float xy_bound = 2 / cos(PI/nMirrors);
		if(object.z_bounds.x > z || object.z_bounds.y < z || abs(x) > xy_bound || abs(y) > xy_bound) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		
		hit.normal = normalize(params3);

		return hit;
	}

	Hit intersect(const Ellipsoid object, const Ray ray) {
		Hit hit;
		hit.t = -1;

		vec3 dirsquare = vec3(ray.dir.x*ray.dir.x, ray.dir.y*ray.dir.y, ray.dir.z*ray.dir.z);
		vec3 dirstart = vec3((ray.start.x-object.center.x)*ray.dir.x, (ray.start.y-object.center.y)*ray.dir.y, (ray.start.z-object.center.z)*ray.dir.z);
		vec3 startsquare = vec3((ray.start.x-object.center.x)*(ray.start.x-object.center.x), (ray.start.y-object.center.y)*(ray.start.y-object.center.y), (ray.start.z-object.center.z)*(ray.start.z-object.center.z));

		float a = dot(object.params, dirsquare);
		float b = dot(dirstart, object.params) * 2.0;
		float c = dot(startsquare, object.params) - 1;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		
		float n0 = -(hit.position.x-object.center.x)*object.params.x/((hit.position.z-object.center.z)*object.params.z);
		float n1 = -(hit.position.y-object.center.y)*object.params.y/((hit.position.z-object.center.z)*object.params.z);
		
		hit.normal = normalize(vec3(n0, n1, -1));

		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		for (int o = 0; o < nEllipsoids; o++) {
			Hit hit = intersect(ellipsoids[o], ray);
			hit.mat = o;
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		for (int r = 0; r < nMirrors; r++) {
			Hit hit = intersect(rectangles[r], ray);
			hit.mat = matType;
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (int o = 0; o < nEllipsoids; o++) if (intersect(ellipsoids[o], ray).t > 0) return true;
		for (int r = 0; r < nMirrors; r++) if (intersect(rectangles[r], ray).t > 0) return true;
		return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const int maxdepth = 16;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * light.La;
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			} else return outRadiance;
		}
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

float rnd() { return (float)rand() / RAND_MAX; }

class Material {
protected:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	bool rough, reflective;
public:
	Material RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
	Material SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
	void SetUniform(unsigned int shaderProg, int mat) {
		char buffer[256];
		sprintf(buffer, "materials[%d].ka", mat);
		ka.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].kd", mat);
		kd.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].ks", mat);
		ks.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].shininess", mat);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform material.shininess cannot be set\n");
		sprintf(buffer, "materials[%d].F0", mat);
		F0.SetUniform(shaderProg, buffer);

		sprintf(buffer, "materials[%d].rough", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform material.rough cannot be set\n");
		sprintf(buffer, "materials[%d].reflective", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform material.reflective cannot be set\n");
	}
};

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

class SmoothMaterial : public Material {
	float calcF0(float n, float k)
	{
		return ((n-1)*(n-1) + k*k)/((n+1)*(n+1) + k*k);
	}
public:
	SmoothMaterial(vec3 n, vec3 k) {
		F0.x = calcF0(n.x, k.x);
		F0.y = calcF0(n.y, k.y);
		F0.z = calcF0(n.z, k.z);
		rough = false;
		reflective = true;
	}
};

struct Mirror {
	vec4 params;
	vec2 z_bounds;

	Mirror(const vec4& _params, const vec2& _z_bounds) : params{_params}, z_bounds{_z_bounds} {}

	void SetUniform(unsigned int shaderProg, int r) {
		char buffer[256];
		sprintf(buffer, "rectangles[%d].params", r);
		params.SetUniform(shaderProg, buffer);
		sprintf(buffer, "rectangles[%d].z_bounds", r);
		z_bounds.SetUniform(shaderProg, buffer);
	}
	
};

struct Ellipsoid {
	vec3 center, params;

	Ellipsoid(const vec3& _center, const vec3& _params) : center{_center}, params{_params}{}
	
	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "ellipsoids[%d].center", o);
		center.SetUniform(shaderProg, buffer);
		sprintf(buffer, "ellipsoids[%d].params", o);
		params.SetUniform(shaderProg, buffer);
	}

	void Animate()
	{
		if(center.x > 2)
			center.x -= rnd()*0.1;
		else if(center.x < -2)
			center.x += rnd()*0.1;
		else if(center.y > 2)
			center.y -= rnd()*0.1;
		else if(center.y < -2)
			center.y += rnd()*0.1;
		else
			center = center + vec3(rnd()*0.1-0.05, rnd()*0.1-0.05, rnd()*0.1-0.05);
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}

	void SetUniform(unsigned int shaderProg) {
		eye.SetUniform(shaderProg, "wEye");
		lookat.SetUniform(shaderProg, "wLookAt");
		right.SetUniform(shaderProg, "wRight");
		up.SetUniform(shaderProg, "wUp");
	}
};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
	void SetUniform(unsigned int shaderProg) {
		La.SetUniform(shaderProg, "light.La");
		Le.SetUniform(shaderProg, "light.Le");
		direction.SetUniform(shaderProg, "light.direction");
	}
};

class Scene {
	std::vector<Ellipsoid *> ellipsoids;
	std::vector<Mirror *> mirrors;
	std::vector<Light *> lights;
	Camera camera;
	std::vector<Material *> materials;
	bool gold = true;
public:
	void setGold(bool _gold){gold = _gold;}
	void build() {
		vec3 eye = vec3(0, 0, 2);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(1, 1, 1), vec3(1, 1, 1), vec3(1, 1, 1)));

		ellipsoids.push_back(new Ellipsoid(vec3(-1+2*rnd(), -1+2*rnd(), -30), vec3((rnd() + 0.5) * 2, (rnd() + 0.5) * 2, (rnd() + 0.5) * 2)));
		ellipsoids.push_back(new Ellipsoid(vec3(-1+2*rnd(), -1+2*rnd(), -30), vec3((rnd() + 0.5) * 2, (rnd() + 0.5) * 2, (rnd() + 0.5) * 2)));
		ellipsoids.push_back(new Ellipsoid(vec3(-1+2*rnd(), -1+2*rnd(), -30), vec3((rnd() + 0.5) * 2, (rnd() + 0.5) * 2, (rnd() + 0.5) * 2)));

		AddMirror();
		AddMirror();
		AddMirror();

		materials.push_back(new RoughMaterial(vec3(1, 0, 0), vec3(50, 50, 50), 50));
		materials.push_back(new RoughMaterial(vec3(0, 1, 0), vec3(50, 50, 50), 50));
		materials.push_back(new RoughMaterial(vec3(0, 0, 1), vec3(50, 50, 50), 50));
		materials.push_back(new SmoothMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9)));
		materials.push_back(new SmoothMaterial(vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1)));
	}
	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nEllipsoids");
		if (location >= 0) glUniform1i(location, ellipsoids.size()); else printf("uniform nEllipsoids cannot be set\n");
		for (int o = 0; o < ellipsoids.size(); o++) ellipsoids[o]->SetUniform(shaderProg, o);
		location = glGetUniformLocation(shaderProg, "matType");
		if (location >= 0) glUniform1i(location, gold ? 3 : 4); else printf("uniform matType cannot be set\n");
		location = glGetUniformLocation(shaderProg, "nMirrors");
		if (location >= 0) glUniform1i(location, mirrors.size()); else printf("uniform nMirrors cannot be set\n");
		for (int r = 0; r < mirrors.size(); r++) mirrors[r]->SetUniform(shaderProg, r);
		lights[0]->SetUniform(shaderProg);
		camera.SetUniform(shaderProg);
		for (int mat = 0; mat < materials.size(); mat++) materials[mat]->SetUniform(shaderProg, mat);
	}
	void AddMirror()
	{
		unsigned int size = mirrors.size();
		if(size < 10)
		{
			mirrors.clear();
			for(unsigned i = 0; i<=size; i++)
			{
				float alfa = (i * 1.0f/(size+1)) * 2 * M_PI;
				mirrors.push_back(new Mirror(vec4(2*sinf(alfa), 2*cosf(alfa), 0, 4), vec2(-25, 0)));
			}
		}
	}
	void Animate()
	{
		for(auto i : ellipsoids)
		{
			i->Animate();
		}
	}
};

GPUProgram gpuProgram;
Scene scene;

class FullScreenTexturedQuad {
	unsigned int vao;
public:
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.Create();

	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
	gpuProgram.Use();
}

void onDisplay() {
	glClearColor(0, 0, 0, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.SetUniform(gpuProgram.getId());
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if(key == 'a')
	{
		scene.AddMirror();
		glutPostRedisplay();
	}
	else if (key == 'g')
	{
		scene.setGold(true);
        glutPostRedisplay();
	}
	else if (key == 's')
	{
		scene.setGold(false);
        glutPostRedisplay();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

long lastTime = 0;

void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    long elapsedTime = time - lastTime;
    for(long l = 0; l<elapsedTime; l+=10) scene.Animate();
	glutPostRedisplay();
    lastTime = time;
}
