#pragma once
#ifndef _LOGGER_H_
#define _LOGGER_H_
#include "stdafx.h"

#define _LOG(x) Logger::getInstance().Log(x)
#define _CONSOLELOG(x) Logger::getInstance().ConsoleLog(x)
#define _FILELOG(x) Logger::getInstance().FileLog(x)
#define _LOG_COMMAND(x) Logger::getInstance().FileLog(x, false)

#define ENABLE_FILE_LOG

#define LogFilePath "Log.log"

class Logger
{
	//boost::mutex m_Mutex;
	std::mutex m_Mutex;
	typedef std::string String;

	std::ofstream file_log_file;
	bool m_bFileLogCreated;
	Logger(void);

	bool createFileLogFile();
public:

	~Logger(void);

	static Logger& getInstance();

	void FileLog(const Logger::String& msg, bool only = true);
	void ConsoleLog(const Logger::String& msg);
	void Log(const Logger::String& msg);

	void FileLog(Logger::String&& msg, bool only = true);
	void ConsoleLog(Logger::String&& msg);
	void Log(Logger::String&& msg);

};

#endif
