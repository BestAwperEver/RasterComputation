#include "StdAfx.h"
#include "Logger.h"
#include <iomanip>

Logger::Logger(void)
{
	m_bFileLogCreated = createFileLogFile();
}

Logger::~Logger(void)
{
	if (file_log_file.is_open()) {
		file_log_file.close();
	}
}

Logger& Logger::getInstance() {
	static Logger instance;
	return instance;
}

bool Logger::createFileLogFile() {
	//boost::mutex::scoped_lock lock(m_Mutex);
	std::unique_lock<std::mutex> lock(m_Mutex);

	file_log_file.open(LogFilePath);
	if (!file_log_file.is_open()) {
		Log("Can't create " LogFilePath " file");
		return false;
	}
	std::time_t t = time(nullptr);
	std::tm npaTm;// = localtime(&t);
	std::tm* aTm = &npaTm;
	auto e = localtime_s(aTm, &t);
	file_log_file << aTm->tm_mday << "." << std::setw(2) << std::setfill('0') << aTm->tm_mon+1
		<< "." << aTm->tm_year+1900 << std::endl;
	//	delete aTm;
	file_log_file.close();
	return true;
}

void Logger::FileLog(const Logger::String& msg, bool palki) {
#ifdef ENABLE_FILE_LOG
	if (!m_bFileLogCreated) {
		return;
	}


	//std::unique_lock<std::mutex> lock(m_Mutex);
	//std::unique_lock(m_Mutex);
	std::unique_lock<std::mutex> lock(m_Mutex);

	file_log_file.open(LogFilePath, std::ios_base::app);

	std::time_t t = time(nullptr);
	std::tm npaTm;// = localtime(&t);
	std::tm* aTm = &npaTm;
	auto e = localtime_s(aTm, &t);
	file_log_file << std::setw(2) << std::setfill('0') << aTm->tm_hour << ":"
		<< std::setw(2) << std::setfill('0') << aTm->tm_min <<":"
		<< std::setw(2) << std::setfill('0') << aTm->tm_sec << ":";
	if (palki) {
		file_log_file << " -- " << msg << " --";
	}
	else {
		file_log_file << " " << msg;
	}
	file_log_file << std::endl;
	//	delete aTm;

	file_log_file.close();

#endif
}
void Logger::ConsoleLog(const Logger::String& msg) {
	std::cout << msg << std::endl;
}
void Logger::Log(const Logger::String& msg) {
	ConsoleLog(msg);
	FileLog(msg, false);
}

void Logger::FileLog(Logger::String&& msg, bool palki) {
#ifdef ENABLE_FILE_LOG
	if (!m_bFileLogCreated) {
		return;
	}

	//boost::mutex::scoped_lock lock(m_Mutex);
	std::unique_lock<std::mutex> lock(m_Mutex);

	file_log_file.open(LogFilePath, std::ios_base::app);

	std::time_t t = time(nullptr);
	std::tm npaTm;// = localtime(&t);
	std::tm* aTm = &npaTm;
	auto e = localtime_s(aTm, &t);
	file_log_file << std::setw(2) << std::setfill('0') << aTm->tm_hour << ":"
		<< std::setw(2) << std::setfill('0') << aTm->tm_min <<":"
		<< std::setw(2) << std::setfill('0') << aTm->tm_sec << ":";
	if (palki) {
		file_log_file << " -- " << msg << " --";
	}
	else {
		file_log_file << " " << msg;
	}
	file_log_file << std::endl;
	//	delete aTm;

	file_log_file.close();

#endif
}
void Logger::ConsoleLog(Logger::String&& msg) {
	std::cout << msg << std::endl;
}
void Logger::Log(Logger::String&& msg) {
	FileLog(msg, false);
}