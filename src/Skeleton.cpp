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
	#version 330				// Shader 3.3

	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
	
	void main() {
		
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

const char * const texturingVertexSource = R"(
	#version 330				// Shader 3.3

	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vuv;
	out vec2 tex;
	void main() {
		tex = vuv;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";


const char * const texturingFragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	in vec2 tex;
	out vec4 outColor;		// computed color of the current pixel
	
	void main() {
		
		outColor = texture(textureUnit, tex);
	}
)";

GPUProgram gpuProgram;
GPUProgram gpuProgramWithTexturing;

class Camera
{
  private:
	vec2 center = vec2(0.0f, 0.0f);
  public:
	void setCenter(vec2 _center)
	{
		center = _center;
	}
	mat4 getTranslationMatrix()
	{
		return TranslateMatrix(-center);
	}
	mat4 getInvTranslationMatrix()
	{
		return TranslateMatrix(center);
	}
} camera;

class KochanekBartels
{
  private:
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
	const float bottom;
  protected:
	const float t;
	unsigned int vao;
	std::vector<vec2> data;
	std::vector<vec2> c_points;
	const int resolution = 600;
  public:
	KochanekBartels(float tension = 0.0, float start = 0.0f, float _bottom = -2.0f) : t(tension), bottom(_bottom)
	{
		add(vec2(-1.0f, start));
		add(vec2(1.0f, start));
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
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

	void add(vec2 _v)
	{
		vec4 vec(_v.x, _v.y, 0, 1);
		vec = vec*camera.getInvTranslationMatrix();
		vec2 v(vec.x, vec.y);
		if(fabs(vec.x) > 1.0f || fabs(vec.y) > 1.0f)
			return;
		auto it = c_points.begin();
		for(; it != c_points.end(); ++it)
		{
			if(fabs((*it).x - v.x) < EPS)	// nincs uj elem beszurva, egy X-hez 1 Y tartozhat
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
			c_points.push_back(v);
		
	}
};

class Hill : public KochanekBartels
{
  private:
	unsigned int vbo[2];
	Texture* text;
  public:
	Hill() : KochanekBartels(0.5f, 0.5f, -1.0f)
	{

		add(vec2(-0.7f, 0.85f));
		add(vec2(0.18f, 0.11f));
		add(vec2(0.52f, 0.61f));

		float coords[8] = {-1.0f, -1.0f,
						   1.0f, -1.0f,
						   1.0f, 1.0f,
						   -1.0f, 1.0f};

		float uvs[8] = {0, 0,
						1, 0,
						1, 1,
						0, 1};

		glGenBuffers(2, vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(coords), coords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0)); 		
	
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0)); 		
		calc();
		std::vector<vec4> image(resolution * resolution);
		for(int i = 0; i<resolution; i++)
		{
			for(int j = 0; j<resolution; j++)
			{
				if(2.0f * i / resolution - 1.0f > data[2*j+1].y)
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
		text = new Texture(resolution, resolution, image);

	}
	void draw()
	{
		glBindVertexArray(vao);
		gpuProgramWithTexturing.Use();
		mat4 MVPTransform{1, 0, 0, 0,
						  0, 1, 0, 0,
						  0, 0, 1, 0,
						  0, 0, 0, 1};

		MVPTransform.SetUniform(gpuProgramWithTexturing.getId(), "MVP");

		text->SetUniform(gpuProgramWithTexturing.getId(), "textureUnit");

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
		gpuProgram.Use();
	}
};

class Course : public KochanekBartels
{
  private:
	unsigned int vbo;
	unsigned int line_vbo;
	unsigned int line_vao;
	float color[3] = {0, 0, 0};
	unsigned int start_index = 0;
	unsigned int stop_index = 0;
	float v(std::vector<vec2>::iterator it)
	{
		if(it == c_points.begin() || it+2 == c_points.end())
			return 0.0f;
		float dx1 = it->x - (it-2)->x;	
		float dy1 = it->y - (it-2)->y;	
		float dx2 = (it+2)->x - it->x;	
		float dy2 = (it+2)->y - it->y;	
		return (1.0f - t) / 2.0f * (dy1/dx1 + dy2/dx2);
	}
	unsigned int current = 3;
  public:
	Course() : KochanekBartels(-0.5f, 0.0f)
	{
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0)); 

		glGenVertexArrays(1, &line_vao);
		glBindVertexArray(line_vao);
		glGenBuffers(1, &line_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0)); 
		setColor(vec3(0,0,0));
	}	

	void moveUnicycle(float);

	float getY(float x, float* tangent)
	{
		if(data[start_index].x > x)
		{
			*tangent = 0.0f;
			return data[start_index].y;
		}
		else if(data[stop_index].x < x)
		{
			*tangent = 0.0f;
			return data[stop_index].y;
		}
		if(data[current].x < x)
		{
			for(int i = current; i < data.size()-2; i+=2)
			{
				if(data[i].x <= x && data[i+2].x >= x)
				{
					float offset = x - data[i].x;
					*tangent = (data[i+2].y - data[i].y)/(data[i+2].x - data[i].x);
					current = i;
					return *tangent*offset + data[i].y;
				}
			}
		}
		else
		{
			for(int i = current; i >= 2; i-=2)
			{
				if(data[i-2].x <= x && data[i].x >= x)
				{
					float offset = x - data[i-2].x;
					*tangent = (data[i].y - data[i-2].y)/(data[i].x - data[i-2].x);
					current = i;
					return *tangent*offset + data[i-2].y;
				}
			}
		}
		return 1/0;
	}


	void setColor(vec3 _color)
	{
		color[0] = _color.x;
		color[1] = _color.y;
		color[2] = _color.z;
	}
	mat4 getMatrix()
	{
		return camera.getTranslationMatrix();
	}

	void startStopDraw()
	{
		glBindVertexArray(line_vao);
		glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
		float f[] = {c_points.front().x, c_points.front().y, 
					 c_points.front().x, 2.0f,
					 c_points.back().x, c_points.back().y,
					 c_points.back().x, 2.0f};
		mat4 MVPTransform = getMatrix();
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.65f, 0.16f, 0.16f);
		glLineWidth(5.0f);
		glBufferData(GL_ARRAY_BUFFER, 8*sizeof(float), f, GL_STATIC_DRAW);
		glDrawArrays(GL_LINES, 0, 4);
	}
	
	void baseDraw()
	{
		if(c_points.size() > 1)
		{
			glBindVertexArray(vao);		
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			mat4 MVPTransform = getMatrix();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, color[0], color[1], color[2]);
			glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(vec2), data.data(), GL_STATIC_DRAW);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, data.size());
		}
	}


	void draw()
	{
		data.clear();

		data.push_back(vec2(-1.0f, -2.0f));
		data.push_back(vec2(-2.0f, -2.0f));
		data.push_back(vec2(-2.0f, c_points.front().y));
		data.push_back(vec2(-1.0f, c_points.front().y));


		start_index = data.size();
		calc();
		stop_index = data.size();

		data.push_back(vec2(2.0f, -2.0f));

		data.push_back(vec2(2.0f, c_points.front().y));

		baseDraw();
		startStopDraw();
	}
};

class Unicycle
{
	const int resolution = 100;
	const int spokes = 7;
	const float bodyLength = 0.15f;
	const float headSize = 0.025f;
  private:
	unsigned int vbo;
	unsigned int vao;
	std::vector<float> data;
	vec2 holdingPoint;
	float holdingTangent;
	vec2 center;
	vec2 b;
	float angle;
  public:
	const float wheelSize = 0.05f;
	const float v_pedal = -M_PI/2;
	float pedal = M_PI;
	int orientation = 1;
	Unicycle()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0)); 
		holdingPoint = vec2(-1.0f, 0.0f);
		holdingTangent = 0.0f;
	}
	void setHoldingPoint(int _orientation, vec2 _holdingPoint, float _holdingTangent)
	{
		orientation = _orientation;
		holdingPoint = _holdingPoint;
		holdingTangent = _holdingTangent;
	}
	
	void drawWheel()
	{
		data.clear();
		angle = fabs(holdingTangent) < EPS ? M_PI/2 : atanf(-1.0f / holdingTangent);
		center = holdingPoint + vec2(holdingTangent <= 0 ? (cosf(angle)) : -(cosf(angle)), holdingTangent <= 0 ? (sinf(angle)) : -(sinf(angle)))*wheelSize;
		int offset = resolution*pedal/M_PI/2.0f;
		for(int i = 0; i<=resolution; i++)
		{
			data.push_back(cosf(2.0f*M_PI/resolution*i)*wheelSize + center.x);
			data.push_back(sinf(2.0f*M_PI/resolution*i)*wheelSize + center.y);
			if(abs((i-offset)*spokes % resolution) < spokes)
			{
				data.push_back(center.x);
				data.push_back(center.y);
				data.push_back(cosf(2.0f*M_PI/resolution*i)*wheelSize + center.x);
				data.push_back(sinf(2.0f*M_PI/resolution*i)*wheelSize + center.y);
			}
		}
		glLineWidth(2.0f);
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, data.size()/2);

	}
	void drawBody()
	{
		data.clear();
		b = center + vec2(0, 1)*wheelSize;
		data.push_back(b.x);
		data.push_back(b.y);
		data.push_back(b.x);
		data.push_back(b.y + bodyLength);
		glLineWidth(3.0f);		
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_STATIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, data.size()/2);
		data.clear();
		for(int i = 0; i<=resolution; i++)
		{
			data.push_back(cosf(2.0f*M_PI/resolution*i - M_PI/2)*headSize + b.x);
			data.push_back(sinf(2.0f*M_PI/resolution*i - M_PI/2)*headSize + b.y + bodyLength);
		}
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_STATIC_DRAW);
		glDrawArrays(GL_TRIANGLE_FAN, 0, data.size()/2);

	}
	void drawLegs(float phase)
	{
		data.clear();
		data.push_back(b.x);
		data.push_back(b.y);
		vec2 foot(center.x + wheelSize * 0.5f * cosf(pedal + phase), center.y + wheelSize * 0.5f * sinf(pedal + phase));
		vec2 d = foot + (b - foot)*0.5f;
		float angle = fabs((b-d).x) < EPS ? -M_PI/2 : (atanf((b-d).y/(b-d).x));
		float length = sqrtf((b-d).x*(b-d).x + (b-d).y*(b-d).y);
		float shift = sqrtf(wheelSize*0.75f*wheelSize*0.75f - length*length);
		if(foot.x > b.x)
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
		glLineWidth(2.0f);
		glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_STATIC_DRAW);
		glDrawArrays(GL_LINE_STRIP, 0, data.size()/2);

	}
	void draw()
	{
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		mat4 MVPTransform = camera.getTranslationMatrix();
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.5f, 0.5f, 0.0f);
		glLineWidth(2.0f);		
		drawWheel();
		drawBody();
		drawLegs(0.0f);
		drawLegs(M_PI);

	}
};

Hill* hills;
Course* course;
Unicycle* uni;

void onInitialization() {
	srand(123);
	glViewport(0, 0, windowWidth, windowHeight);

	hills = new Hill();
	course = new Course();
	uni = new Unicycle();

	gpuProgramWithTexturing.Create(texturingVertexSource, texturingFragmentSource, "outColor");
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");

}

void onDisplay() {
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	hills->draw();
	course->draw();
	uni->draw();

	glutSwapBuffers();
}
bool doing = true;

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ')
	{
		doing = !doing;
	} 

}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
	if(state == GLUT_DOWN && button == GLUT_LEFT_BUTTON)
	{
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;	
		course->add(vec2(cX, cY));
		glutPostRedisplay();
	}
}

long lastTime = 0;
int i = 0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	course->moveUnicycle((time - lastTime)/1000.0f);		
	glutPostRedisplay();
	lastTime = time;
}

float currX = -1.0f;
int orientation = 1;
void Course::moveUnicycle(float elapsedTime)
{
	const float dt = 0.01f;
	const float F = 0.5;
	const float m = 0.05;
	const float g = 9.81;
	const float ro = 13;
	float tangent;
	float y = getY(currX, &tangent);
	for(float t = 0; t < elapsedTime; t+=dt)
	{
		if(currX>=1.0f)
		{
			currX = 1.0f;
			y = getY(currX, &tangent);
			orientation=-1;
		}
		else if(currX<=-1.0f)
		{
			currX = -1.0f;
			y = getY(currX, &tangent);
			orientation=1;
		}
		float Dt = fmin(dt, elapsedTime-t);
		float v = (F-orientation*m*g*sinf(atanf(tangent))/ro);
		float dx = v*Dt / sqrt(1+tangent*tangent);
		uni->pedal -= orientation*v*Dt/uni->wheelSize*2;
		currX+=orientation*dx;
		y = getY(currX, &tangent);
	}
	uni->setHoldingPoint(orientation, vec2(currX, y), tangent);
	camera.setCenter(vec2(currX, y));

}
