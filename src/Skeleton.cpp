//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2018. osztol.
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
#define EPS 1E-7
const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 MVP;
	layout(location = 0) in vec2 vp;
	
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform vec3 color;
	out vec4 outColor;
	
	void main() {
		
		outColor = vec4(color, 1);
	}
)";

const char * const texturingVertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 MVP;
	layout(location = 0) in vec2 vp;
	layout(location = 1) in vec2 vuv;
	out vec2 tex;
	void main() {
		tex = vuv;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;
	}
)";


const char * const texturingFragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform sampler2D textureUnit;
	in vec2 tex;
	out vec4 outColor;
	
	void main() {
		outColor = texture(textureUnit, tex);
	}
)";

class GpuProgramSwitcher
{
  private:
	GPUProgram gpuProgram;
	GPUProgram gpuProgramWithTexturing;
	bool texturing = false;
  public:
	GpuProgramSwitcher()
	{
		gpuProgram.Create(vertexSource, fragmentSource, "outColor");
		gpuProgramWithTexturing.Create(texturingVertexSource, texturingFragmentSource, "outColor");
	}
	unsigned int getId()
	{
		return texturing ?
			gpuProgramWithTexturing.getId():
			gpuProgram.getId();
	}
	void setTexturing(bool _texturing)
	{
		texturing = _texturing;
		if(texturing) gpuProgramWithTexturing.Use();
		else gpuProgram.Use();
	}
};

GpuProgramSwitcher* gprog;

class Camera
{
  private:
	vec2 center = vec2(0.0f, 0.0f);
  public:
	void setCenter(vec2 _center)
	{
		center = _center;
	}
	mat4 getTranslationMatrix() const
	{
		return TranslateMatrix(-center);
	}
	mat4 getInvTranslationMatrix() const
	{
		return TranslateMatrix(center);
	}
};

Camera* camera;

class Drawable
{
  private:
	unsigned int vao;
  public:
	virtual void draw() = 0;
  protected:  
	Drawable()
	{
		glGenVertexArrays(1, &vao);
		bindVao();
	}
  	void bindVao()
	{
		glBindVertexArray(vao);
	}
};

class ColouredDrawable : public Drawable
{
  private:
	vec3 color;
	unsigned int vbo;
  public:
	virtual void draw() = 0;
  protected:  
	const int resolution = 100;
	ColouredDrawable(vec3 _color) : color{_color}
	{
		glGenBuffers(1, &vbo);
		bindVbo();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0)); 

	}
	void bindVbo()
	{
		bindVao();
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	void setMatrices(mat4 matrix = {1,0,0,0,
									0,1,0,0,
									0,0,1,0,
									0,0,0,1})
	{
		gprog->setTexturing(false);
		bindVbo();
		mat4 MVP = camera->getTranslationMatrix()*matrix;
		MVP.SetUniform(gprog->getId(), (char*)"MVP");
		color.SetUniform(gprog->getId(), (char*)"color");			
	}
};

class TexturedDrawable : public Drawable
{
  private:
	const float* coords;
	const float* uvs;
	size_t size;
	unsigned int vbo[2];
  public:
	virtual void draw() = 0;
  protected:  
	Texture* texture;
	TexturedDrawable(size_t _size, const float* _coords, const float* _uvs) :
		size{_size},
		coords{_coords},
		uvs{_uvs}
	{
		glGenBuffers(2, vbo);
		bufferVbo(0);
		bufferVbo(1);
	}
	void bindVbo(unsigned int id)
	{
		bindVao();
		glBindBuffer(GL_ARRAY_BUFFER, vbo[id]);
	}
	void bufferVbo(unsigned int id)
	{
		bindVbo(id);
		glBufferData(GL_ARRAY_BUFFER, size*sizeof(float), (id ? uvs : coords), GL_STATIC_DRAW);
		for(int i = 0; i<size; i++)
		glEnableVertexAttribArray(id);
		glVertexAttribPointer(id, 2, GL_FLOAT, GL_FALSE, 0,  reinterpret_cast<void*>(0)); 		
	}
	void setMatrices()
	{
		gprog->setTexturing(true);
		bindVbo(0);
		mat4 MVP{1, 0, 0, 0,
				 0, 1, 0, 0,
				 0, 0, 1, 0,
				 0, 0, 0, 1};
		MVP.SetUniform(gprog->getId(), (char*)"MVP");
		if(texture) texture->SetUniform(gprog->getId(), (char*)"textureUnit");
	}
};

class KochanekBartels
{
  private:
	const float t;

	std::vector<vec2> data;
	std::vector<vec2> c_points;
	float v(std::vector<vec2>::iterator it)
	{
		if(it == c_points.begin() || it+1 == c_points.end())
			return 0.0f;
		float dx1 = it->x - (it-1)->x;	
		float dy1 = it->y - (it-1)->y;	
		float dx2 = (it+1)->x - it->x;	
		float dy2 = (it+1)->y - it->y;	
		return (1.0f - t) / 2.0f * (dy1/dx1 + dy2/dx2);
	}
  protected:
	const float bottom;
	bool dirty = true;
	static const int resolution = 600;
	KochanekBartels(float _t, float _height = 0.0f, float _b = -2.0f) :
		t{_t},
		bottom{_b}
	{
		c_points.push_back(vec2(-1.0f, _height));
		c_points.push_back(vec2(1.0f, _height));
	}
	vec2 getData(unsigned int i)
	{
		return data[i];
	}
	size_t getDataSize()
	{
		return data.size();
	}
	void putData(vec2 v)
	{
		data.push_back(v);
	}
	void clearData()
	{
		data.clear();
	}
	vec2 getFrontCP()
	{
		return c_points.front();
	}
	vec2 getBackCP()
	{
		return c_points.back();
	}
	vec2* getData()
	{
		return data.data();
	}
	void calc()
	{
		float currX = c_points.front().x;
		float currY = c_points.front().y;
		for(auto it = c_points.begin(); it+1 != c_points.end(); it++)
		{
			if(currX >= (it+1)->x)
			{
				data.push_back(vec2(it->x, bottom));
				data.push_back(vec2(it->x, currY = it->y));
			}
			while(currX < (it + 1)->x)
			{
				
				data.push_back(vec2(currX, bottom));
				if(fabs(currX - it->x)<EPS)
				{
					data.push_back(vec2(it->x, currY = it->y));					
				}
				else
				{
					float a3 = (v(it+1)+v(it))/((it+1)->x - it->x)/((it+1)->x - it->x) + (it->y - (it+1)->y)*2.0f/((it+1)->x - it->x)/((it+1)->x - it->x)/((it+1)->x - it->x); 
					float a2 = ((it+1)->y - it->y)*3.0f/((it+1)->x - it->x)/((it+1)->x - it->x) - (v(it+1)+2*v(it))/((it+1)->x - it->x);
					float a1 = v(it);
					float a0 = it->y;
					data.push_back(vec2(currX, currY = a3*(currX - it->x)*(currX - it->x)*(currX - it->x) + a2*(currX - it->x)*(currX - it->x) + a1*(currX - it->x) + a0));
				}
				currX+=2.0f/resolution;
			}
		}
		vec2 last = c_points.back();
		data.push_back(vec2(last.x, bottom));
		data.push_back(last);
	}
  public:
	void add(vec2 _v)
	{
		dirty = true;
		vec4 vec(_v.x, _v.y, 0, 1);
		vec = vec*camera->getInvTranslationMatrix();
		vec2 v(vec.x, vec.y);
		if(fabs(vec.x) > 1.0f || fabs(vec.y) > 1.0f)
			return;
		auto it = c_points.begin();
		for(; it != c_points.end(); ++it)
		{
			if(fabs((*it).x - v.x) < EPS)
			{
				(*it).y = v.y;
				break;
			}
			else if((*it).x > v.x)
			{
				c_points.insert(it, v);
				break;
			}
		}
		if(it == c_points.end())
		{
			c_points.push_back(v);
		}
	}
};

class Hill : public KochanekBartels, public TexturedDrawable
{
	static const float c[8];
	static const float u[8];
  public:
	Hill() :
		KochanekBartels(0.5f, 0.5f, -1.0f),
		TexturedDrawable(sizeof(c)/sizeof(float), c, u)
	{
		add(vec2(-0.7f, 0.85f));
		add(vec2(0.18f, 0.11f));
		add(vec2(0.52f, 0.61f));
		calc();
		std::vector<vec4> image(resolution * resolution);
		for(int i = 0; i<resolution; i++)
		{
			for(int j = 0; j<resolution; j++)
			{
				if(2.0f * i / resolution - 1.0f > getData(2*j+1).y)
				{
					image[i*resolution + j] = vec4(0.2f, 0.2f, 0.6f, 1.0f);
				}
				else
				{
					if((i > 4*resolution / 5) || ((i > 2 * resolution / 3 ) && (i * 7.5f / resolution - 5 )*RAND_MAX >= rand() ))	
						image[i*resolution + j] = vec4(1, 1, 1, 1);
					else
						image[i*resolution + j] = vec4(0.3f, 0.3f, 0.3f, 1.0f);
				}
			}
		}
		texture = new Texture(resolution, resolution, image);
	}
	void draw()
	{
		setMatrices();
		glDrawArrays(GL_TRIANGLE_FAN, 0, sizeof(c)/sizeof(float)/2);
	}
};
const float Hill::c[8] = {-1.0f ,-1.0f,
						   1.0f ,-1.0f,
						   1.0f , 1.0f ,
						 -1.0f , 1.0f };
const float Hill::u[8] = {0, 0,
						 1, 0,
						 1, 1,
						 0, 1};

class Course : public ColouredDrawable, public KochanekBartels
{
  private:
	unsigned int start_index = 0;
	unsigned int stop_index  = 0;
	unsigned int current = 3;
  public:
	Course() :
		ColouredDrawable(vec3(0, 0, 0)),
		KochanekBartels(-0.5f, 0.0f)
	{}
	float getY(float x, float* tangent)
	{
		if(getData(start_index).x > x)
		{
			*tangent = 0.0f;
			return getData(start_index).y;
		}
		else if(getData(stop_index).x < x)
		{
			*tangent = 0.0f;
			return getData(stop_index).y;
		}
		if(getData(current).x < x)
		{
			for(int i = current; i < getDataSize()-2; i+=2)
			{
				if(getData(i).x <= x && getData(i+2).x >= x)
				{
					float offset = x - getData(i).x;
					*tangent = (getData(i+2).y - getData(i).y)/(getData(i+2).x - getData(i).x);
					current = i;
					return *tangent*offset + getData(i).y;
				}
			}
		}
		else
		{
			for(int i = current; i >= 2; i-=2)
			{
				if(getData(i-2).x <= x && getData(i).x >= x)
				{
					float offset = x - getData(i-2).x;
					*tangent = (getData(i).y - getData(i-2).y)/(getData(i).x - getData(i-2).x);
					current = i;
					return *tangent*offset + getData(i-2).y;
				}
			}
		}
		return 1/0;
	}

	void draw()
	{
		setMatrices();
		if(dirty)
		{
			clearData();

			putData(vec2(-1.0f, bottom));
			putData(vec2(-2.0f, bottom));
			putData(vec2(-2.0f, getFrontCP().y));
			putData(vec2(-1.0f, getFrontCP().y));


			start_index = getDataSize();
			calc();
			stop_index = getDataSize();

			putData(vec2(2.0f, bottom));

			putData(vec2(2.0f, getBackCP().y));
			glBufferData(GL_ARRAY_BUFFER, getDataSize()*sizeof(vec2), getData(), GL_DYNAMIC_DRAW);
			dirty = false;
		}
		glDrawArrays(GL_TRIANGLE_STRIP, 0, getDataSize());
	}

};

class UnicycleWheel : public ColouredDrawable
{
  private:
	const int spokes;
	const float wheelSize;
	vec2 center;
	float pedal = 0;
  public:
	UnicycleWheel(const int _spokes, const float _wheelSize):
		ColouredDrawable(vec3(1.0f, 0.506f, 0.125f)),
		spokes{_spokes},
		wheelSize{_wheelSize}
	{}
	void draw()
	{
		setMatrices(TranslateMatrix(center));
		std::vector<float> data;
		int offset = resolution*pedal/M_PI/2.0f;
		for(int i = 0; i<=resolution; i++)
		{
			data.push_back(cosf(2.0f*M_PI/resolution*i)*wheelSize);
			data.push_back(sinf(2.0f*M_PI/resolution*i)*wheelSize);
			if(abs((i-offset)*spokes % resolution) < spokes)
			{
				data.push_back(0.0f);
				data.push_back(0.0f);
				data.push_back(cosf(2.0f*M_PI/resolution*i)*wheelSize);
				data.push_back(sinf(2.0f*M_PI/resolution*i)*wheelSize);
			}
		}
		glLineWidth(1.0f);
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, data.size()/2);
	}
	
	void setPedal(float _pedal)
	{
		pedal = _pedal;
	}

	void setCenter(vec2 _center)
	{
		center = _center;
	}
};

class UnicycleBody : public ColouredDrawable
{
  private:
	const float bodyLength;
	vec2 bodyBottom;
	std::vector<float> data;
  public:
	UnicycleBody(const float _bodyLength) :
		ColouredDrawable(vec3(0.011f, 1.0f, 0.475f)),
		bodyLength{_bodyLength}
	{
		data.push_back(0.0f);
		data.push_back(0.0f);
		data.push_back(0.0f);
		data.push_back(bodyLength);
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_STATIC_DRAW);
	}
	void draw()
	{
		setMatrices(TranslateMatrix(bodyBottom));
		glLineWidth(4.0f);		
		glDrawArrays(GL_LINE_STRIP, 0, data.size()/2);
	}
	void setBodyBottom(vec2 _bodyBottom)
	{
		bodyBottom = _bodyBottom;
	}

};

class UnicycleHead : public ColouredDrawable
{
  private:
	const float headSize;
	vec2 headCenter;
	std::vector<float> data;
  public:
	UnicycleHead(const float _headSize) :
		ColouredDrawable(vec3(0.011f, 1.0f, 0.475f)),
		headSize{_headSize}
	{
		for(int i = 0; i<=resolution; i++)
		{
			data.push_back(cosf(2.0f*M_PI/resolution*i - M_PI/2)*headSize);
			data.push_back(sinf(2.0f*M_PI/resolution*i - M_PI/2)*headSize);
		}
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_STATIC_DRAW);
	}
	void draw()
	{
		setMatrices(TranslateMatrix(headCenter));
		glLineWidth(4.0f);		
		glDrawArrays(GL_TRIANGLE_FAN, 0, data.size()/2);	
	}
	void setHeadCenter(vec2 _headCenter)
	{
		headCenter = _headCenter;
	}

};
class UnicycleLeg : public ColouredDrawable
{
  private:
	const float wheelSize;
	float phase;
	float pedal = 0;
	vec2 bodyBottom;
	int orientation = 1;
  public:
	UnicycleLeg(float _phase, const float _wheelSize) :
		ColouredDrawable(vec3(0.011f, 1.0f, 0.475f)),
		wheelSize{_wheelSize},
		phase{_phase}
	{

	}
	void draw()
	{
		setMatrices(TranslateMatrix(bodyBottom));		
		std::vector<float> data;
		data.push_back(0.0f);
		data.push_back(0.0f);
		vec2 foot(wheelSize * 0.5f * cosf(pedal + phase), -wheelSize + wheelSize * 0.5f * sinf(pedal + phase));
		vec2 d = foot*0.5f;
		float angle = fabs((-d).x) < EPS ? -M_PI/2 : (atanf((-d).y/(-d).x));
		float length = sqrtf((-d).x*(-d).x + (-d).y*(-d).y);
		float shift = sqrtf(wheelSize*0.75f*wheelSize*0.75f - length*length);
		if(foot.x > 0.0f)
		{
			data.push_back(d.x + orientation*(shift*cosf(angle + M_PI/2)));		//itt valami nemjó..
			data.push_back(d.y + orientation*(shift*sinf(angle + M_PI/2)));
		}
		else
		{
			data.push_back(d.x - orientation*(shift*cosf(angle + M_PI/2)));		//itt valami nemjó..
			data.push_back(d.y - orientation*(shift*sinf(angle + M_PI/2)));
		}
		data.push_back(foot.x);
		data.push_back(foot.y);
		glLineWidth(4.0f);
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, data.size()/2);
	}
	void setBodyBottom(vec2 _bodyBottom)
	{
		bodyBottom = _bodyBottom;
	}
	
	void setPedal(float _pedal)
	{
		pedal = _pedal;
	}
	void setOrientation(int _orientation)
	{
		orientation = _orientation;
	}
};

class Unicycle
{
  private:
	const int spokes = 7;
	const float wheelSize = 0.05f;
	const float bodyLength = 0.1f;
	const float headSize = 0.025f;
  private:
	UnicycleWheel wheel;
	UnicycleBody body;
	UnicycleHead head;
	UnicycleLeg legs[2] = {UnicycleLeg{0.0f, wheelSize},
						   UnicycleLeg{M_PI, wheelSize}};
	float pedal = 0;
  public:
	Unicycle() :
		wheel{spokes, wheelSize},
		body{bodyLength},
		head{headSize}
	{
	}
	void setDPedal(float _pedal)
	{
		pedal += _pedal/wheelSize;
		wheel.setPedal(pedal);
		legs[0].setPedal(pedal);
		legs[1].setPedal(pedal);
	}

	void draw()
	{
		wheel.draw();
		body.draw();
		head.draw();
		legs[0].draw();
		legs[1].draw();
	}
	void setHoldingPoint(int orientation, vec2 holdingPoint, float holdingTangent)
	{
		legs[0].setOrientation(orientation);
		legs[1].setOrientation(orientation);
		float angle = fabs(holdingTangent) < EPS ? M_PI/2 : atanf(-1.0f / holdingTangent);
		vec2 center = holdingPoint + vec2(holdingTangent <= 0 ? (cosf(angle)) : -(cosf(angle)), holdingTangent <= 0 ? (sinf(angle)) : -(sinf(angle)))*wheelSize;
		wheel.setCenter(center);
		vec2 bodyBottom = center + vec2{0, 1}*wheelSize;
		body.setBodyBottom(bodyBottom);
		head.setHeadCenter(bodyBottom + vec2(0, bodyLength));
		legs[0].setBodyBottom(bodyBottom);
		legs[1].setBodyBottom(bodyBottom);
	}
};


class World
{
  private:
	Hill* h = nullptr;
	Course* c = nullptr;
	Unicycle* u = nullptr;
  public:
	void init()
	{
		camera = new Camera();
		gprog = new GpuProgramSwitcher();
		h = new Hill();
		c = new Course();
		u = new Unicycle();
		u->setHoldingPoint(1, vec2(0.0f, 0.0f), 0);
	}
	void addCP(vec2 v)
	{
		c->add(v);
	}
	void draw()
	{
		if(h) h->draw();
		if(c) c->draw();
		if(u) u->draw();
	}
	~World()
	{
		delete h;
		delete c;
		delete camera;
		delete gprog;
	}
	void setDPedal(float pedal)
	{
		u->setDPedal(pedal);
	}
	void setHoldingPoint(int _orientation, vec2 _holdingPoint, float _holdingTangent)
	{
		u->setHoldingPoint(_orientation, _holdingPoint, _holdingTangent);
	}
	float getY(float _x, float* _tangent)
	{
		return c->getY(_x, _tangent);
	}
} world;

void onInitialization() {
	srand(123);
	glViewport(0, 0, windowWidth, windowHeight);
	world.init();
}

void onDisplay() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	world.draw();

	glutSwapBuffers();
}
bool doing = false;

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ')
	{
		doing = true;
	} 

}

void onMouse(int button, int state, int pX, int pY) {
	if(state == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
	{
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;	
		world.addCP(vec2(cX, cY));
		glutPostRedisplay();
	}
}
long lastTime = 0;
float currX = -1.0f;
int orientation = 1;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	float elapsedTime = (time - lastTime)/1000.0f;
	const float dt = 0.01f;
	const float F = 0.4f;
	const float m = 0.02f;
	const float g = 9.81f;
	const float ro = 1.0f;
	float tangent;
	float y = world.getY(currX, &tangent);
	for(float t = 0; t < elapsedTime; t+=dt)
	{
		if(currX>=1.0f)
		{
			currX = 1.0f;
			y = world.getY(currX, &tangent);
			orientation=-1;
		}
		else if(currX<=-1.0f)
		{
			currX = -1.0f;
			y = world.getY(currX, &tangent);
			orientation=1;
		}
		float Dt = fmin(dt, elapsedTime-t);
		float v = (F-orientation*m*g*sinf(atanf(tangent)))/ro;
		float dx = v*Dt / sqrt(1+tangent*tangent);
		world.setDPedal(-orientation*v*Dt);
		currX+=orientation*dx;
		y = world.getY(currX, &tangent);
	}
	world.setHoldingPoint(orientation, vec2(currX, y), tangent);
	if(doing)camera->setCenter(vec2(currX, y));
	glutPostRedisplay();
	lastTime = time;
}



void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}