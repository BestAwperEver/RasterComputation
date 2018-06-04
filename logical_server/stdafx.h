#define _USE_ASSERTS_

#define _WIN32_WINNT 0x0A00

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <list>
#include <vector>
#include <mutex>

#ifdef WIN32
#include <stdio.h>
#endif