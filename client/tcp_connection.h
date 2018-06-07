#pragma once
#ifndef _TCP_CONNECTION_H_
#define _TCP_CONNECTION_H_

//#define CHECK_PING
#define EMULATE_PING

#define MEM_FN(x)       std::bind(&self_type::x, shared_from_this())
#define MEM_FN1(x,y)    std::bind(&self_type::x, shared_from_this(),y)
#define MEM_FN2(x,y,z)  std::bind(&self_type::x, shared_from_this(),y,z)

#define ONLY_IMPORTANT_MESSAGES_TO_LOG true
#define ENABLE_NET_LOG
//#define ENABLE_CONSOLE_NET_LOG
#define MAX_SERVER_RESPONDONG_TIME 500

//#include "shared_enums.h"

//class myapp;

enum class REQUEST_TYPE : char {
	LOGIN,
	PING,
	WANT_TO_DISCONNECT,
	END_MESSAGE
};

class talk_to_svr : public std::enable_shared_from_this<talk_to_svr> {
	typedef talk_to_svr self_type;
	typedef unsigned int uint;

	talk_to_svr(boost::asio::io_service& service);
	talk_to_svr(const talk_to_svr&) = delete;

	void log(const std::string& msg, const boost::system::error_code& error);
	void log(const std::string& msg, bool is_important = false);
	void f_log(const std::string& msg); // force put message into the log

	void handle_timer(const std::string& error_msg, const boost::system::error_code& e);

public:
	typedef boost::system::error_code error_code;
	typedef std::shared_ptr<talk_to_svr> ptr;

	static ptr start(boost::asio::io_service&
			, boost::asio::ip::tcp::endpoint
			, const std::string & username);

	static void setEnableUnimportantMessages(bool enable);
	static bool getEnableUnimportantMessages();
	//static void enableLog(bool enable);

	void start(boost::asio::ip::tcp::endpoint ep, const std::string & username);
	void stop();
	
	bool incoming(std::vector<char>& msg);
	bool post(const std::vector<char>& msg);

	static void set_emulate_ping(uint min, uint max);
	static void get_emulate_ping(uint& min, uint& max);

	unsigned int get_ping() const { return m_Ping; }

	uint getSendedBytesCount()		{ return m_Sended;	}
	uint getRecievedBytesCount()	{ return m_Recieved;}

private:
	void on_connect(const error_code & err);
	void on_read_size(const error_code & err);
	void on_read(const error_code & err, size_t bytes);

//	void on_login(std::vector<char>& msg);
	inline void on_ping(short ping);//std::vector<char>& msg);
//	void on_clients(const std::vector<char>& msg);
	void add_incoming_message(std::vector<char>& msg);
	void check_out_message();
	void server_is_bastard();
	void ping();
	void login();
	//void do_ping();
	void postpone_ping();
//	void do_ask_clients();	// не нужен

	void write(const std::vector<char>& msg, bool ping = false);
	void on_write(const error_code & err, size_t bytes);
	void do_read(bool only_size = true);
	inline void do_write(const std::vector<char>& msg);
	size_t read_complete(const boost::system::error_code & err, size_t bytes);

	inline void do_delete_incoming_msg(void);
	inline void do_add_outcoming_msg(const std::vector<char>& msg);

public:
	bool reading()		{ return m_bIsReading;	}
	bool writing()		{ return m_bIsWriting;	}
	bool connected()	{ return m_bConnected;	}
	bool started()		{ return m_bStarted;	}
private:
	bool m_bIsReading;
	bool m_bIsWriting;
	bool m_bConnected;
	bool m_bStarted;
	//bool m_bLoggedIn;

	static uint m_MinPing;
	static uint m_MaxPing;
	uint m_Ping;
	uint m_Sended;
	uint m_Recieved;

	static bool m_bOnlyImportant;
	//static bool m_bEnableLog;

	boost::asio::ip::tcp::socket sock_;

	enum { 
		max_in_msg = 1024*10,
		max_out_msg = 1024*10
	};
	char read_buffer_[max_in_msg];
	char write_buffer_[max_out_msg];

	std::string username_;

	boost::asio::deadline_timer timer_;
	boost::asio::deadline_timer m_timer;

	std::string last_error;
	std::string disconnect_error;

	std::deque<std::vector<char>> m_IncomingMsgs;
	std::deque<std::vector<char>> m_OutcomingMsgs;

	std::mutex m_Mutex;
};

#endif
