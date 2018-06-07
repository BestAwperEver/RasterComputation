#pragma once
enum OpType {
	ADD,
	MULTIPLY,
	SUBSTRACT,
	DIVIDE,
	SQRT,
	//POW,
	SIN,
	COS,
	TAN,
	CTG,
	COMPLEX,
	MAX
};

void doWithCuda(int n, float *x, float *y, OpType OT);
bool runTest(int argc, char **argv, const char *imageFilename, const char *refFilename);