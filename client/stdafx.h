#pragma once
#ifndef _STDAFX_H_
#define _STDAFX_H_

#include <boost/asio.hpp>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <deque>
#include <fstream>
#include <iostream>
#include <cassert>
#include <vector>
#include <mutex>
#include <thread>
#include <memory>
#include <iomanip>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
//#  define NOMINMAX
#  include <windows.h>
#endif

#endif