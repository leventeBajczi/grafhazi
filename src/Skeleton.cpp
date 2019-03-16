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

GPUProgram gpuProgram;

class KochanekBartels
{
  private:
	std::vector<float> data;
	std::vector<vec2> c_points;
	unsigned int vbo;
	unsigned int vao;

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
	float color[3] = {0, 0, 0};

  protected:
	const float t;
	const int resolution = 100;

  public:
	KochanekBartels(float tension = 0.0, float start = 0.0f) : t(tension)
	{
		add(vec2(-1.0f, start));
		add(vec2(1.0f, start));
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(data.data()), data.data(), GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0)); 
	}
	void setColor(vec3 _color)
	{
		color[0] = _color.x;
		color[1] = _color.y;
		color[2] = _color.z;
	}
	void calc()
	{
		data.clear();
		float currX = c_points.front().x;
		float currY = c_points.front().y;
		for(auto it = c_points.begin(); it+1 != c_points.end(); it++)
		{
			while(currX < (it + 1)->x)
			{
				if(fabs(currX - it->x)<EPS)
				{
					data.push_back(it->x);	
					data.push_back(currY = it->y);	
				}
				else
				{
					float a3 = (v(it+1)+v(it))/((it+1)->x - it->x)/((it+1)->x - it->x) - ((it+1)->y - it->y)*2.0f/((it+1)->x - it->x)/((it+1)->x - it->x)/((it+1)->x - it->x); 
					float a2 = ((it+1)->y - it->y)*3.0f/((it+1)->x - it->x)/((it+1)->x - it->x) - (v(it+1)+2*v(it))/((it+1)->x - it->x);
					float a1 = v(it);
					float a0 = it->y;
					data.push_back(currX);
					data.push_back(a3*(currX - it->x)*(currX - it->x)*(currX - it->x) + a2*(currX - it->x)*(currX - it->x) + a1*(currX - it->x) + a0);
				}
				currX+=2.0f/resolution;
			}
		}
		data.push_back(c_points.back().x);
		data.push_back(c_points.back().y);
		glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
	}

	void draw()
	{
		if(c_points.size() > 1)
		{
			glBindVertexArray(vao);		
			glBindBuffer(GL_ARRAY_BUFFER, vbo);

			glBindVertexArray(vao);
			mat4 MVPTransform{ 1, 0, 0, 0,    // MVP matrix, 
		                      0, 1, 0, 0,    // row-major!
		                      0, 0, 1, 0,
		                      0, 0, 0, 1 };
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, color[0], color[1], color[2]);
			calc();
			printf("vbo: %d\npoints: %d\n", vbo, c_points.size());
			glDrawArrays(GL_LINE_STRIP, 0, data.size() / 2);
		}
	}
	void add(vec2 v)
	{
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

KochanekBartels* hills;
KochanekBartels* course;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	hills = new KochanekBartels(0.5f, 0.5f);
	hills->setColor(vec3(1,1,0));
	course = new KochanekBartels(-0.5f);
	course->setColor(vec3(0,1,0));

	//TODO: Create unicycle

	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);

	hills->draw();
	course->draw();

	//TODO: Unicycle draw

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ')
	{
		//TODO: move camera
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

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	// TODO: animation
}
