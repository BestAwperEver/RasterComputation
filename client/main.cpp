#include "stdafx.h"
#include <iostream>
#include <chrono>
#include <CL\cl.hpp>

#define SIMDPP_ARCH_X86_SSE3
#include <simdpp\simd.h>

#include "fasttrigo.h"
#include "cuda_common.h"
#include "tcp_connection.h"

struct times {
	float cuda_time;
	float cuda_full_time;
	float scalar_time;
	float avx_time;
	times operator += (const times& t) {
		cuda_time += t.cuda_time;
		cuda_full_time += t.cuda_full_time;
		scalar_time += t.scalar_time;
		avx_time += t.avx_time;
		return *this;
	}
	times operator /= (const float f) {
		cuda_time /= f;
		cuda_full_time /= f;
		scalar_time /= f;
		avx_time /= f;
		return *this;
	}
};

times array_test(OpType OT)
{
	times t;
	const size_t N = 1 << 26;
	float *x_device, *y_device;
	float *x, *y;

	int v;
	cudaDeviceGetAttribute(&v, cudaDevAttrManagedMemory, 0);
	v = 0;

	if (v == 1) {
		cudaMallocManaged(&x_device, N * sizeof(float));
		cudaMallocManaged(&y_device, N * sizeof(float));
		x = x_device;
		y = y_device;
	}
	else {
		x = new float[N];
		y = new float[N];
	}

	for (int i = 0; i < N; ++i) {
		x[i] = 9.0f;
		y[i] = 2.0f;
	}

	if (v != 1) {
		cudaMalloc((void **)&x_device, N * sizeof(float));
		cudaMalloc((void **)&y_device, N * sizeof(float));
	}

	auto start = std::chrono::high_resolution_clock::now();

	if (v != 1) {
		cudaMemcpy(x_device, x, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(y_device, y, N * sizeof(float), cudaMemcpyHostToDevice);
	}

	auto start2 = std::chrono::high_resolution_clock::now();

	doWithCuda(N, x_device, y_device, OT);

	cudaDeviceSynchronize();

	auto elapsed2 = std::chrono::high_resolution_clock::now() - start2;

	if (v != 1) {
		cudaMemcpy(x, x_device, N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(y, y_device, N * sizeof(float), cudaMemcpyDeviceToHost);
	}

	auto elapsed = std::chrono::high_resolution_clock::now() - start;
	long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	t.cuda_full_time = microseconds;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed2).count();
	t.cuda_time = microseconds;

	float* result2 = new float[N];

	start = std::chrono::high_resolution_clock::now();

	switch (OT) {
	case ADD:
		for (int i = 0; i < N; ++i) {
			result2[i] = x[i] + y[i];
		} break;
	case SUBSTRACT:
		for (int i = 0; i < N; ++i) {
			result2[i] = x[i] - y[i];
		} break;
	case MULTIPLY:
		for (int i = 0; i < N; ++i) {
			result2[i] = x[i] * y[i];
		} break;
	case DIVIDE:
		for (int i = 0; i < N; ++i) {
			result2[i] = x[i] / y[i];
		} break;
	case SQRT:
		for (int i = 0; i < N; ++i) {
			result2[i] = std::sqrtf(x[i]);
		} break;
	//case POW:
	//	for (int i = 0; i < N; ++i) {
	//		result2[i] = std::pow(x[i], y[i]);
	//	} break;
	case SIN:
		for (int i = 0; i < N; ++i) {
			result2[i] = std::sinf(x[i]);
		} break;
	case COS:
		for (int i = 0; i < N; ++i) {
			result2[i] = std::cosf(x[i]);
		} break;
	case TAN:
		for (int i = 0; i < N; ++i) {
			result2[i] = std::sinf(x[i])/std::cosf(x[i]);
		} break;
	case CTG:
		for (int i = 0; i < N; ++i) {
			result2[i] = std::cosf(x[i])/std::sinf(x[i]);
		} break;
	case COMPLEX:
		for (int i = 0; i < N; ++i) {
			result2[i] = std::sqrtf(std::cosf(x[i]) / std::sinf(x[i])) / std::sqrt(std::sinf(y[i]) / std::cosf(y[i]));
		} break;
	}

	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

	t.scalar_time = microseconds;

	float* result = new float[N];

	start = std::chrono::high_resolution_clock::now();

	switch (OT) {
	case ADD:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = simdpp::add(xmmA, xmmB);
			simdpp::store(result + i, xmmC);
		} break;
	case SUBSTRACT:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = simdpp::sub(xmmA, xmmB);
			simdpp::store(result + i, xmmC);
		} break;
	case MULTIPLY:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = simdpp::mul(xmmA, xmmB);
			simdpp::store(result + i, xmmC);
		} break;
	case DIVIDE:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = simdpp::div(xmmA, xmmB);
			simdpp::store(result + i, xmmC);
		} break;
	case SQRT:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = simdpp::sqrt(xmmA);
			simdpp::store(result + i, xmmC);
		} break;
	//no such thing as _mm_pow_ps, _mm_exp_ps or _mm_log_ps
	//case POW:
	//	for (int i = 0; i < N; i += 4) {
	//		simdpp::float32<4> xmmA = simdpp::load(x + i);
	//		simdpp::float32<4> xmmB = simdpp::load(y + i);
	//		simdpp::float32<4> xmmC = simdpp::pow(xmmA, xmmB);
	//		simdpp::store(result + i, xmmC);
	//	} break;
	case SIN:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = FT::sin_ps(xmmA.native());
			simdpp::store(result + i, xmmC);
		} break;
	case COS:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = FT::cos_ps(xmmA.native());
			simdpp::store(result + i, xmmC);
		} break;
	case TAN:
		for (int i = 0; i < N; i += 4) {
			//__m128 xmmA = _mm_load_ps(x + i);
			//__m128 xmmB = _mm_load_ps(y + i);
			//__m128 xmmC = _mm_div_ps(FT::sin_ps(xmmA), FT::cos_ps(xmmA));
			//memcpy_s(result + i, sizeof(float), &xmmC, sizeof(float));
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = simdpp::div(simdpp::float32<4>(FT::sin_ps(xmmA.native()))
				, simdpp::float32<4>(FT::cos_ps(xmmA.native())));
			simdpp::store(result + i, xmmC);
		} break;
	case CTG:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC = simdpp::div(simdpp::float32<4>(FT::cos_ps(xmmA.native()))
				, simdpp::float32<4>(FT::sin_ps(xmmA.native())));
			simdpp::store(result + i, xmmC);
		} break;
	case COMPLEX:
		for (int i = 0; i < N; i += 4) {
			simdpp::float32<4> xmmA = simdpp::load(x + i);
			simdpp::float32<4> xmmB = simdpp::load(y + i);
			simdpp::float32<4> xmmC =
				simdpp::div(
					simdpp::sqrt(simdpp::div(simdpp::float32<4>(FT::cos_ps(xmmA.native()))
						, simdpp::float32<4>(FT::sin_ps(xmmA.native())))),
					simdpp::div(simdpp::float32<4>(FT::sin_ps(xmmB.native()))
						, simdpp::float32<4>(FT::cos_ps(xmmB.native()))));
			simdpp::store(result + i, xmmC);
	//		result2[i] = std::sqrtf(std::cosf(x[i]) / std::sinf(x[i])) / std::sqrt(std::sinf(y[i]) / std::cosf(y[i]));
		} break;
	}

	elapsed = std::chrono::high_resolution_clock::now() - start;
	microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

	t.avx_time = microseconds;

	delete[] result;
	delete[] result2;

	if (v == 1) {
		cudaFree(x);
		cudaFree(y);
	}
	else {
		cudaFree(x_device);
		cudaFree(y_device);
		delete[] x;
		delete[] y;
	}

	return t;
}

int texture_rotation(int argc, char **argv)
{
	const char *imageFilename = "lena_bw.pgm";
	const char *refFilename = "ref_rotated.pgm";

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "input"))
		{
			getCmdLineArgumentString(argc,
				(const char **)argv,
				"input",
				(char **)&imageFilename);

			if (checkCmdLineFlag(argc, (const char **)argv, "reference"))
			{
				getCmdLineArgumentString(argc,
					(const char **)argv,
					"reference",
					(char **)&refFilename);
			}
			else
			{
				printf("-input flag should be used with -reference flag");
				exit(EXIT_FAILURE);
			}
		}
		else if (checkCmdLineFlag(argc, (const char **)argv, "reference"))
		{
			printf("-reference flag should be used with -input flag");
			exit(EXIT_FAILURE);
		}
	}

	bool testResult = runTest(argc, argv, imageFilename, refFilename);

	printf("Process completed, returned %s\n",
		testResult ? "OK" : "ERROR!");

	system("pause");

	exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

int opencl_test() {
	//get all platforms (drivers)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	//get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";


	cl::Context context({ default_device });

	cl::Program::Sources sources;

	// kernel calculates for each element C=A+B
	std::string kernel_code =
		"   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
		"       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
		"   }                                                                               ";
	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}


	// create buffers on the device
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

	int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

	//create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	//write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);


	//run the kernel
	//cl 1.1
	//cl::KernelFunctor simple_add(cl::Kernel(program,"simple_add"),queue,cl::NullRange,cl::NDRange(10),cl::NullRange);
	//simple_add(buffer_A,buffer_B,buffer_C);

	//cl 1.2
	//cl::make_kernel simple_add(cl::Kernel(program, "simple_add"));
	//cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
	//simple_add(eargs, buffer_A, buffer_B, buffer_C).wait();

	//alternative way to run the kernel
	cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
	kernel_add.setArg(0,buffer_A);
	kernel_add.setArg(1,buffer_B);
	kernel_add.setArg(2,buffer_C);
	queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
	queue.finish();

	int C[10];
	//read result C from the device to array C
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

	std::cout << " result: \n";
	for (int i = 0; i<10; i++) {
		std::cout << C[i] << " ";
	}

	system("pause");

	return 0;
}

const char* OpTypeString[] = {
	"ADD",
	"MULTIPLY",
	"SUBSTRACT",
	"DIVIDE",
	"SQRT",
	//"POW",
	"SIN",
	"COS",
	"TAN",
	"CTG",
	"COMPLEX",
	"MAX"
};

void test() {
	int N = 10;
	for (int i = ADD; i < MAX; ++i) {
		std::cout << OpTypeString[i] << std::endl;
		times t = { .0f, .0f, .0f };
		for (int j = 0; j < N; ++j) {
			t += array_test(OpType(i));
		}
		t /= N;
		std::cout << "Scalar: " << t.scalar_time << std::endl;
		std::cout << "AVX: " << t.avx_time << std::endl;
		std::cout << "CUDA full time: " << t.cuda_full_time << std::endl;
		std::cout << "CUDA computation only time: " << t.cuda_time << std::endl;
		std::cout << std::endl;
	}
}

boost::asio::io_service service;
talk_to_svr::ptr ttr;

void handle_thread() {
	srand(time(0));
	service.run();
}

void start() {
	boost::system::error_code err;
	boost::asio::ip::address adr = boost::asio::ip::address::from_string("radagast.asuscomm.com", err);
	boost::asio::ip::tcp::endpoint ep(boost::asio::ip::address::from_string("radagast.asuscomm.com"), 27015);
	std::thread(std::bind(handle_thread)).detach();
	ttr = talk_to_svr::start(service, ep, "admin");
}

int main(int argc, char **argv) {
	//texture_rotation(argc, argv);
	//test();
	//opencl_test();
	start();

	system("pause");

	return 0;
}

