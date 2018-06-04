#pragma once
#ifndef _TCP_SERVER_H_
#define _TCP_SERVER_H_

#define _SERVER tcp_server::getInstance()

#define ENABLE_LOG
#define ENABLE_CONSOLE_NET_LOG

#include <vector>
#include <map>

#include "tcp_connection.h"
#include "ClientDataBase.h"
#include "Client.h"

class tcp_server
{
	//friend talk_to_client;
	//friend Client;

	typedef boost::shared_ptr<talk_to_client> client_ptr;
	typedef tcp_server self_type;
//	typedef std::vector<client_ptr> array;
	typedef talk_to_client client;

	void handle_timer(const boost::system::error_code&);	
	void handle_accept(client::ptr, const boost::system::error_code&);
	
	tcp_server(unsigned short Port);
	~tcp_server();

	bool do_compare(char registered_user[], const std::string& connected_user);

	std::map<int, Client::ptr> m_Clients;
	std::map<int, Client::ptr>& clients() { return m_Clients; }

	void Log(std::string msg);
	//bool createGameLogFile();

public:

	static tcp_server& getInstance();

	void start();

	Client::ptr get_client(int client_id);

	void post_all(std::vector<char> msg, client_ptr except = nullptr);
	void post(int client_id, std::vector<char> msg);
	void post(int client_id, char command);
	void post(Client::ptr, std::vector<char> msg);
	void post(Client::ptr, char command);

	void add_client(client_ptr);
	void drop_client(client_ptr);
	void update_clients_changed();
	//std::string getClientListStr();
	bool client_registered(std::string& login);
	bool already_logged_in(const std::string& login);

private:
	boost::asio::deadline_timer m_Timer;
	//boost::asio::deadline_timer m_TurnTimer;
	boost::asio::ip::tcp::acceptor m_Acceptor;
	boost::mutex m_Mutex;
	//boost::mutex m_LogMutex;
//	array clients;
	//bool m_bLogFileCreated;
	//std::ofstream log_file;
};

#endif
