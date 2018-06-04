#include "tcp_server.h"
#include "Logger.h"
#include <boost/lexical_cast.hpp>

using namespace boost::asio;
using namespace boost::posix_time;

#define TIME_TO_ACCEPT boost::posix_time::seconds(6000) // при отсутствии запросов на подключение вырубается
#define DEFAULT_PORT 27015

#define INT(x) *( (int*)(&x) )
#define UD(x) *( (UnitDef*)(&x) )

boost::asio::io_service g_Service;

tcp_server::tcp_server(unsigned short port)
	: m_Timer(g_Service, TIME_TO_ACCEPT)
	//, m_TurnTimer(g_Service)
	, m_Acceptor(g_Service, ip::tcp::endpoint(ip::tcp::v4(), port))
	//, online_clients(_CLDATA.getClientsCount() + 1, false)
	//, m_bLogFileCreated(false)
{}
tcp_server::~tcp_server() {}

void tcp_server::handle_timer(const boost::system::error_code& err) {
	if (err) {
		return;
	}
	m_Acceptor.cancel();
}
void tcp_server::handle_accept(talk_to_client::ptr client, const boost::system::error_code & err) {
	if (err) {
		std::cout << "acception failed" << std::endl;
		return;
	}

	m_Timer.cancel();

	client->start();
	talk_to_client::ptr new_client = talk_to_client::new_(g_Service);

	m_Timer.expires_from_now(TIME_TO_ACCEPT);
	m_Acceptor.async_accept(new_client->sock(), boost::bind(&tcp_server::handle_accept,this,new_client,_1));
	m_Timer.async_wait( boost::bind( &tcp_server::handle_timer, this, _1) );
}

void tcp_server::update_clients_changed() {
	//for( auto b = clients.begin(), e = clients.end(); b != e; ++b)
	//	b->second->set_clients_changed();
}

tcp_server& tcp_server::getInstance() {
	static tcp_server instance(DEFAULT_PORT);
	return instance;
}

void tcp_server::start() {
	talk_to_client::ptr client = talk_to_client::new_(g_Service);

	m_Acceptor.async_accept(client->sock(), boost::bind(&tcp_server::handle_accept,this,client,_1));
	m_Timer.async_wait( boost::bind(&tcp_server::handle_timer, this, _1) );
		
	g_Service.run();
}

void tcp_server::Log(std::string msg) {
#ifdef ENABLE_LOG
//	if (!m_bLogFileCreated) {
//		return;
//	}
//	std::time_t t = time(nullptr);
//	std::tm* aTm = localtime(&t);
//	log_file << std::setw(2) << std::setfill('0') << aTm->tm_hour << ":"
//		<< std::setw(2) << std::setfill('0') << aTm->tm_min <<":"
//		<< std::setw(2) << std::setfill('0') << aTm->tm_sec << ":" << " " << msg << std::endl;
////	delete aTm;
	_LOG(msg);
#endif
}

//std::string tcp_server::getClientListStr() {
//	std::vector<char> msg;
//	for( auto b = clients.begin(), e = clients.end() ; b != e; ++b)
//		msg += b->second->username() + " ";
//	return msg;
//}

void tcp_server::add_client(client_ptr cl) {
	std::string name = cl->username();
	int client_id = _CLDATA.getClientID(name);
	//cl->set_id(client_id);

	Client::ptr player = get_client(client_id);

	if (player.get() == nullptr) {

		player =  Client::create(client_id, cl);

		{
			boost::mutex::scoped_lock lk(m_Mutex);
			//clients[cl->get_id()] = cl;
			//online_clients[cl_id] = true;
			m_Clients[client_id] = player;
		}

		cl->set_player(player);

	} else {
		player->set_client(cl);
		player->set_online(true);

		cl->set_player(player);
	}

	std::vector<char> msg(sizeof(char)*2 + sizeof(int) + name.size(), 0);
	int i = 0;
	msg[i++] = static_cast<char>(SERVER_ANSWER::NEW_CLIENT);

	INT(msg[i]) = client_id;
	i += sizeof(int);

	msg[i++] = name.size();
	memcpy_s(&msg[i], name.size(), name.c_str(), name.size());
	i += name.size();

	if (i != sizeof(char)*2 + sizeof(int) + name.size()) {
		int a = 5;
	}

	post_all(msg, cl);

	Log("Client " + cl->get_address_string() + " was logged in as "
		+ cl->username() + " (id " + boost::lexical_cast<std::string>(client_id) + ")");
	//if (!client_reconnected(cl)) create_start_unit(cl);
}
void tcp_server::drop_client(client_ptr cl) {

	Log("Droping client " + cl->get_address_string() + " ("
		+ cl->username() + ", id " + boost::lexical_cast<std::string>(cl->get_id()) + ")");

	//int cl_id = _CLDATA.getClientID(cl->username());
	Client::ptr p = cl->get_player();

	//{
	//	boost::mutex::scoped_lock lk(m_Mutex);

	////auto it = std::find(clients.begin(), clients.end(), cl);
	////if (it != clients.end()) clients.erase(it);

	////for(auto iter = clients.begin(); iter != clients.end(); ) {
	////	if (iter->second == cl) {
	////		online_clients[_CLDATA.getClientID(cl->username())] = false;
	////		iter = clients.erase(iter);
	////	} else {
	////		++iter;
	////	}
	////}

	//	//online_clients[cl->get_id()] = false;
	//	p.set_online(false);
	//	p.set_client(nullptr);
	//}

	p->set_online(false);
	p->set_client(nullptr);

	if (cl->started()) {
		cl->stop();
	}

	const int N = sizeof(char) + sizeof(int);
	std::vector<char> msg(N, 0);;
	int i = 0;
	msg[0] = static_cast<char>(SERVER_ANSWER::DROP_PLAYER);
	INT(msg[1]) = p->get_id();

	post_all(msg, cl);

	//update_clients_changed();
}

void tcp_server::post_all(std::vector<char> msg, client_ptr except) {
	if (except) {
		int except_id = except->get_id();
		// надо заменить на client_id и смотреть по нему
		//for(auto b = clients.begin(), e = clients.end(); b != e; ++b) {
		//	if (b->second != except) b->second->post(msg);
		//}
		for(auto b = m_Clients.begin(), e = m_Clients.end(); b != e; ++b) {
			if (b->second->is_online() && b->first != except_id) {
				b->second->get_client()->post(msg);
			}
		}
	} else {
		for(auto b = m_Clients.begin(), e = m_Clients.end(); b != e; ++b) {
			if (b->second->is_online()) {
				b->second->get_client()->post(msg);
			}
		}
	}
}
void tcp_server::post(int client_id, std::vector<char> msg) {
	post(m_Clients[client_id], msg);
}
void tcp_server::post(int client_id, char command) {
	post(m_Clients[client_id], command);
}
void tcp_server::post(Client::ptr player, std::vector<char> msg) {
	if (player->is_online()) player->get_client()->post(msg);
}
void tcp_server::post(Client::ptr player, char command) {
	if (player->is_online()) player->get_client()->post(command);
}
bool tcp_server::do_compare(char reg[], const std::string& con) {
	int sz = strlen(reg);
	if (sz != con.size()) {
		return false;
	}
	for (char* ch = reg, k = 0; *ch; ++ch, ++k) {
		if (std::tolower(con[k]) != std::tolower(*ch)) {
			return false;
		}
	}
	return true;
}

bool tcp_server::client_registered(std::string& login) {
	auto id = _CLDATA.getClientID(login);

	if (id == -1) return false;

	login = _CLDATA.getClientName(id);

	return true;
}
bool tcp_server::already_logged_in(const std::string& login) {
	//for( auto b = clients.begin(), e = clients.end(); b != e; ++b)
	//	if (b->second->username() == login) {
	//		return true;
	//	}

	int cl_id = _CLDATA.getClientID(login);

	if (m_Clients.find(cl_id) == m_Clients.end()) {
		return false;
	}

	return m_Clients[cl_id]->is_online();

/*
	Если сделать vector<bool> logged_in_clients размерности базы все существующих клиентов,
	а это в случае с bool в любом случае будет крайне мало по занимаемому объему данных,
	то можно просто проверять logged_in_clients[client_id]
*/
	//return false;
}

Client::ptr tcp_server::get_client(int client_id) {
	//return m_Clients[client_id];
	auto it = m_Clients.find(client_id);
	if (it == m_Clients.end()) {
		return nullptr;
	}
	return it->second;
}