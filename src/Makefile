run: grafhazi
	./grafhazi
grafhazi: Skeleton.cpp framework.cpp framework.h
	g++ -g *.cpp -lGL -lglut -lGLEW -o grafhazi
clean:
	rm grafhazi
debug: grafhazi
	gdb grafhazi 