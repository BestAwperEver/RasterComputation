/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
*/

#include "stdafx.h"

#define MAX_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// Constants
const float angle = 0.5f;        // angle to rotate image by (in radians)

								 // Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

////////////////////////////////////////////////////////////////////////////////
//! Transform an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void transformKernel(float *outputData,
	int width,
	int height,
	float theta)
{
	int stridex = blockDim.x * gridDim.x;
	int stridey = blockDim.y * gridDim.y;

	// calculate normalized texture coordinates
	
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	while (y < height) {

		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

		while (x < width) {

			float u = (float)x - (float)width / 2;
			float v = (float)y - (float)height / 2;
			float tu = u*cosf(theta) - v*sinf(theta);
			float tv = v*cosf(theta) + u*sinf(theta);

			tu /= (float)width;
			tv /= (float)height;

			// read from texture and write to global memory
			outputData[y*width + x] = tex2D(tex, tu + 0.5f, tv + 0.5f);

			x += stridex;
		}
		y += stridey;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, const char *imageFilename, const char *refFilename)
{
	int devID = findCudaDevice(argc, (const char **)argv);

	// load image from disk
	float *hData = NULL;
	unsigned int width, height;
	char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

	if (imagePath == NULL)
	{
		printf("Unable to source image file: %s\n", imageFilename);
		exit(EXIT_FAILURE);
	}

	sdkLoadPGM(imagePath, &hData, &width, &height);

	unsigned int size = width * height * sizeof(float);
	printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

	//Load reference image from image (output)
	float *hDataRef = (float *)malloc(size);
	char *refPath = sdkFindFilePath(refFilename, argv[0]);

	if (refPath == NULL)
	{
		printf("Unable to find reference image file: %s\n", refFilename);
		exit(EXIT_FAILURE);
	}

	sdkLoadPGM(refPath, &hDataRef, &width, &height);

	// Allocate device memory for result
	float *dData = NULL;
	checkCudaErrors(cudaMalloc((void **)&dData, size));

	// Allocate array and copy image data
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArray;
	checkCudaErrors(cudaMallocArray(&cuArray,
		&channelDesc,
		width,
		height));
	checkCudaErrors(cudaMemcpyToArray(cuArray,
		0,
		0,
		hData,
		size,
		cudaMemcpyHostToDevice));

	// Set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = true;    // access with normalized texture coordinates

							  // Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

	dim3 dimBlock(8, 8, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

	//dim3 dimBlock(2, 5, 1);
	//dim3 dimGrid(14, height / 13, 1);

	// Warmup
	transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

	checkCudaErrors(cudaDeviceSynchronize());
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	// Execute the kernel
	transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height, angle);

	// Check if kernel execution generated an error
	getLastCudaError("Kernel execution failed");

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
	printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
	printf("%.2f Mpixels/sec\n",
		(width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
	sdkDeleteTimer(&timer);

	// Allocate mem for the result on host side
	float *hOutputData = (float *)malloc(size);
	// copy result from device to host
	checkCudaErrors(cudaMemcpy(hOutputData,
		dData,
		size,
		cudaMemcpyDeviceToHost));

	// Write result to file
	char outputFilename[1024];
	strcpy(outputFilename, imagePath);
	strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
	sdkSavePGM(outputFilename, hOutputData, width, height);
	printf("Wrote '%s'\n", outputFilename);

	// Auto-Verification Code
	bool testResult = true;

	// Write regression file if necessary
	if (checkCmdLineFlag(argc, (const char **)argv, "regression"))
	{
		// Write file for regression test
		sdkWriteFile<float>("./data/regression.dat",
			hOutputData,
			width*height,
			0.0f,
			false);
	}
	else
	{
		// We need to reload the data from disk,
		// because it is inverted upon output
		sdkLoadPGM(outputFilename, &hOutputData, &width, &height);

		printf("Comparing files\n");
		printf("\toutput:    <%s>\n", outputFilename);
		printf("\treference: <%s>\n", refPath);

		testResult = compareData(hOutputData,
			hDataRef,
			width*height,
			MAX_EPSILON_ERROR,
			0.15f);
	}

	checkCudaErrors(cudaFree(dData));
	checkCudaErrors(cudaFreeArray(cuArray));
	free(imagePath);
	free(refPath);

	return testResult;
}