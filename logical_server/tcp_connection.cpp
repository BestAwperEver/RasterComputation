#include "tcp_connection.h"
#include "tcp_server.h"
#include "Logger.h"
#include "Client.h"
#include <boost/lexical_cast.hpp>

using namespace boost::asio;
using namespace boost::posix_time;
typedef tcp_server Server;

#define INT(x) *( (int*)(&x) )
#define UD(x) *( (UnitDef*)(&x) )
#define SHORT(x) *( (short*)(&x) )

std::vector<char> operator + (std::vector<char> v, std::string s) {
	std::vector<char> res(v.size() + s.size(), 0);
	memcpy(&res[0], &v[0], v.size());
	memcpy(&res[v.size()], &s[0], s.size());
	return std::move( res );
}
void operator += (std::vector<char>& v, std::string s) {
	int old_size(v.size());
	v.resize(old_size + s.size());
	memcpy(&v[old_size], &s[0], s.size());
}

talk_to_client::talk_to_client(io_service& service) : sock_(service), started_(false), 
	m_DisconnectTimer(service), 
	m_Timer(service),
	//clients_changed_(false),
	loggedin_(false)
	//id_(-1)
	, m_Client(0)
	//m_bIsActive(false)
{}
talk_to_client::~talk_to_client() {

}
void talk_to_client:: log(const std::string& msg, const boost::system::error_code& e) const {
	//std::cout << msg << " (" << username_ << ") " << e << std::endl;
	//boost::mutex::scoped_lock(m_Mutex);
	if (m_Client) {
		_LOG(msg + " (" + username_ + ", id " + boost::lexical_cast<std::string>(m_Client->get_id())
			+ ") error: " + e.message());
	} else {
		_LOG(msg + " (" + username_ + ") error: " + e.message());
	}
}
void talk_to_client:: log(const std::string& msg, bool is_important) const {
#ifdef ONLY_IMPORTANT_LOG
	if (is_important) {
#endif
		//std::cout << msg << " (" << username_ << ")" << std::endl;
		//boost::mutex::scoped_lock(m_Mutex);
		if (m_Client) {
		_LOG(msg + " (" + username_ + ", id "
			+ boost::lexical_cast<std::string>(m_Client->get_id()) + ")");
		} else {
			_LOG(msg + " (" + username_ + ")");
		}
#ifdef ONLY_IMPORTANT_LOG
	}
#endif
}

talk_to_client::ptr talk_to_client::new_(io_service& service) {
	ptr new_(new talk_to_client(service));
	return new_;
}

void talk_to_client:: start() {
	started_ = true;
	last_sended = clock();//boost::posix_time::microsec_clock::local_time();
	// first, we wait for client to login
	//post_check_ping();
	do_read();
}
void talk_to_client:: stop() {
	if ( !started_) {
		log("stop: returned, not started", true);
		return;
	}
	log("stop", true);
	started_ = false;
	if (loggedin_) {
		_SERVER.drop_client( shared_from_this() );
	}
	sock_.close();

	//ptr self = shared_from_this();
	//_SERVER.drop_client(shared_from_this());
}

std::string talk_to_client::get_address_string() const {
	return sock_.remote_endpoint().address().to_string();
}

bool talk_to_client:: post(std::vector<char>&& msg) {
	if (started_) {
		do_add_outcoming_msg(msg);
		return true;
	}
	return false;
}
bool talk_to_client:: post(const std::vector<char>& msg) {
	if (started_) {
		do_add_outcoming_msg(msg);
		return true;
	}
	return false;
}
bool talk_to_client:: post(char command) {
	if (started_) {
		do_add_outcoming_msg(std::vector<char>(1, command));
		return true;
	}
	return false;
}

void talk_to_client:: on_read_size(const error_code & err) {
	m_DisconnectTimer.cancel();
	if ( err) {
		log("on_read_size", err);
		stop();
		return;
	}
	if ( !started_ ) return;
	if (*((short*)read_buffer_) > max_msg) {
		log("Too big message size: " + boost::to_string(*((short*)read_buffer_)), true);
		client_is_bastard();
		return;
	}
	do_read(false);
}
void talk_to_client:: on_read(const error_code & err, size_t bytes) {
	m_DisconnectTimer.cancel();
	if ( err) {
		log("on_read",err);
		stop();
		return;
	}
	if ( !started_ ) return;

	if (check_ping() == CP_TOO_BIG) {
		log("stopping client due to a high ping", true);
		//write("ping high");
		do_ping();
		stop();
		return;
	}

	if (*read_buffer_ == static_cast<char>(REQUEST_TYPE::PING)) {
		on_ping();
		return;
	}

	// process the msg
	if (read_buffer_[bytes-1] != static_cast<char>(REQUEST_TYPE::END_MESSAGE)) {
		log("Message is not ended by REQUEST_TYPE::END_MESSAGE", true);
		client_is_bastard();
	}
	std::vector<char> msg(read_buffer_, read_buffer_ + bytes);
	if ( msg[0] == static_cast<char>(REQUEST_TYPE::LOGIN) ) on_login(msg);
	else {
		if (!loggedin_) {
			log("Client is not logged in", true);
			client_is_bastard();
			return;
		}
		//if ( msg[0] == static_cast<char>(REQUEST_TYPE::PING) ) on_ping();
		//else if ( msg.find("ask_clients") == 0) on_clients();
		//else 
		parse_message(msg);//std::cerr << "invalid msg " << msg << std::endl;
	}
}
void talk_to_client:: on_login(std::vector<char>& msg) {
	if (msg.size() < sizeof(short)) {
		client_is_bastard();
		return;
	}

	char len = msg[1];
	
	if (msg.size() != sizeof(short) + len + 1) {
		client_is_bastard();
		return;
	}

	username_ = std::string(&msg[sizeof(short)], &msg[sizeof(short)] + len);

	if (_SERVER.client_registered(username_)) {
		if (_SERVER.already_logged_in(username_)) {
			log("Client already logged in", true);
			//char ans[3] = {REQUEST_TYPE::LOGIN, SERVER_ANSWER::LOGIN_ALREADY, 0};
			std::vector<char> ans(2, 0);
			ans[0] = static_cast<char>(REQUEST_TYPE::LOGIN);
			ans[1] = static_cast<char>(SERVER_ANSWER::LOGIN_ALREADY);
			write(ans);
			return;
		} else if (username_ == "Radagast"
			&& sock_.remote_endpoint().address().to_string() != "127.0.0.1")
		{
			log("Detected trying to log as Radagast from " + sock_.remote_endpoint().address().to_string(), true);
			//char ans[3] = {REQUEST_TYPE::LOGIN, LOGIN_FAILED, 0};
			std::vector<char> ans(2, 0);
			ans[0] = static_cast<char>(REQUEST_TYPE::LOGIN);
			ans[1] = static_cast<char>(SERVER_ANSWER::LOGIN_ALREADY);
			write(ans);
			return;
		}

		_SERVER.add_client( shared_from_this() );

		loggedin_ = true;

		//id_ = _CLDATA.getClientID(username_);

		const int N = 3*sizeof(char) + sizeof(int);
		//char ans[N];
		std::vector<char> ans(N, 0);
		int i = 0;
		ans[i++] = static_cast<char>(REQUEST_TYPE::LOGIN);
		ans[i++] = static_cast<char>(SERVER_ANSWER::LOGIN_OK);
		//*( (int*)(ans+i) ) = get_id();
		INT(ans[i]) = get_id();
		i += sizeof(int);
		ans[i++] = username_.size();
		//ans[i] = 0;
		//std::string ans;
		//ans.push_back(REQUEST_TYPE::LOGIN);
		//ans.push_back(SERVER_ANSWER::LOGIN_OK);
		//ans.push_back(username_.size());
		//ans += username_;
		//ans.push_back(_SERVER.get_team(username_)); // теперь в лобби

		//// костыльчик =(^_^)=
		//ans.push_back(LOAD_MAP);
		//ans.push_back(7);
		//ans += "testmap";

		write(ans + username_);

		_SERVER.update_clients_changed();

		//if (_SERVER.game_started()) do_add_outcoming_msg(_SERVER.get_map_create_msg());

	} else {
		log("Client dropped: unregistered", true);
		//char ans[3] = {REQUEST_TYPE::LOGIN, LOGIN_FAILED, 0};
		std::vector<char> ans(2, 0);
		ans[0] = static_cast<char>(REQUEST_TYPE::LOGIN);
		ans[1] = static_cast<char>(SERVER_ANSWER::LOGIN_FAILED);
		write(ans);
	}
}
void talk_to_client:: on_ping() {
	log("on_ping");
	//do_write(clients_changed_ ? "ping client_list_changed\n" : "ping ok\n");
	check_out_message();
	//clients_changed_ = false;
}
void talk_to_client:: on_clients() {
#ifdef _USE_ASSERTS_
	assert(false && "wrong request");
#endif
	log("on_clients");
	//std::vector<char> msg = _SERVER.getClientListStr();
	//do_write("clients " + msg);
}
void talk_to_client:: on_write(const error_code & err, size_t bytes, bool disconnect_after_writing) {
	last_sended = clock();
	//last_sended = boost::posix_time::microsec_clock::local_time();
	if (err) {
		log("on_write", err);
		stop();
		return;
	}
	log("on_write");
	if (disconnect_after_writing) {
		stop();
	} else {
		do_read();
	}
}

void talk_to_client:: write(std::vector<char> msg) {
	//std::string len_str(sizeof(short),0);
	short size = msg.size() + 1;
	//std::memcpy(&len_str[0], &size, sizeof(short));

	std::vector<char> final_msg(sizeof(short) + msg.size() + sizeof(char), 0);
	memcpy(&final_msg[0], &size, sizeof(short));
	memcpy(&final_msg[sizeof(short)], &msg[0], msg.size());
	final_msg[sizeof(short) + msg.size()] = static_cast<char>(REQUEST_TYPE::END_MESSAGE);

	//do_write(len_str + msg + char(REQUEST_TYPE::END_MESSAGE));
	do_write(std::move( final_msg ));
}
//void talk_to_client:: write(std::vector<char>&& msg) {
//	do_write(char(msg.size()+1) + msg + char(REQUEST_TYPE::END_MESSAGE));
//}
void talk_to_client:: ping() {
	log("ping");
	if (last_ping < TIME_TO_PING - 10) {
		m_Timer.expires_from_now(boost::posix_time::milliseconds(TIME_TO_PING - last_ping));
		m_Timer.async_wait( MEM_FN(do_ping));
	} else {
		do_ping();
	}
}
void talk_to_client:: disconnect_timer_handler(const boost::system::error_code& e) {
	if (e) {
		if (e.value() == 995) {
			log("disconnect_timer calcelled", true);
			return;
		}
	log("disconnect_timer_handler", e);
	return;
	}
	log("Client is not responding", true);
	stop();
}
void talk_to_client:: post_check_ping(int time_to_disconnect) {
	log("post_check_ping");
#ifdef CHECK_PING
	m_DisconnectTimer.expires_from_now(boost::posix_time::seconds(time_to_disconnect));
	m_DisconnectTimer.async_wait( MEM_FN1(disconnect_timer_handler, _1));
#endif
}
talk_to_client::CP_RESULT talk_to_client::check_ping() {
	//boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
	last_ping = clock() - last_sended;//(now - last_sended).total_microseconds() / 1000;
#ifdef CHECK_PING
	if ( last_ping > CP_TOO_BIG ) {
		log("check ping: " + boost::to_string(last_ping), true);
		return CP_TOO_BIG;
	} else {
		log("check_ping: " + boost::to_string(last_ping));
	}
#endif
	return CP_OK;
}

void talk_to_client:: parse_message(const std::vector<char>& msg) {
	bool res = false;
	if (msg[0] == static_cast<char>(REQUEST_TYPE::WANT_TO_DISCONNECT)) {
		res = true;
		//stop();
		_SERVER.drop_client(shared_from_this());
	}
	//if (res) _SERVER.post(msg.substr(0, msg.size()-1) + char(UNBLOCK));
	//else {
	if (!res) {
		_LOG("Sending GTFO to " + username_
			+ " (id " + boost::lexical_cast<std::string>(get_id()) + ")");
		std::vector<char> ans(2, 0);
		ans[0] = static_cast<char>(SERVER_ANSWER::DENIED); // GTFO
		post(std::move( ans ));
	}
	check_out_message();
}

void talk_to_client:: check_out_message() {
#ifdef ENABLE_CONSOLE_NET_LOG
	log("check_out_message");
#endif
	if (!started()) {
#ifdef ENABLE_CONSOLE_NET_LOG
	log("can't check outcoming messages: connection isn't started", true);
#endif
		return;
	}
	boost::mutex::scoped_lock lk(m_Mutex);
	if (m_OutcomingMsgs.size()) {
		write(m_OutcomingMsgs.front());
		m_OutcomingMsgs.pop_front();
	} else {
		ping();
	}
}
void talk_to_client:: client_is_bastard() {
	log("Client " + sock_.remote_endpoint().address().to_string() + " is damn bastard", true);
	stop();
}

void talk_to_client::	do_read(bool only_size) {
	if ( !started_) {
		log("do_read: returned, not started", true);
		return;
	}
	log("do_read");
	if (only_size) {
		post_check_ping();
		async_read(sock_, buffer(read_buffer_), 
			//MEM_FN2(read_complete,_1,_2), MEM_FN2(on_read,_1,_2));
			transfer_exactly(sizeof(short)), MEM_FN1(on_read_size, _1));
	} else {
		post_check_ping(5);
		async_read(sock_, buffer(read_buffer_), 
			transfer_exactly(*((short*)(read_buffer_))), MEM_FN2(on_read,_1,_2));
	}
}
void talk_to_client::	do_write(std::vector<char> msg, bool disconnect_after) {
	if ( !started_) {
		log("do_write: returned, not started", true);
		return;
	}
	log("do_write");
#ifdef _USE_ASSERTS_
	assert( msg.size() < max_msg );
#endif
	std::copy(msg.begin(), msg.end(), write_buffer_);
//	sock_.async_write_some( buffer(write_buffer_, msg.size()), 
//		MEM_FN3(on_write,_1,_2,disconnect_after));
	async_write(sock_, buffer(write_buffer_, msg.size()), transfer_exactly(msg.size()), 
		MEM_FN3(on_write,_1,_2,disconnect_after));
}
//void talk_to_client::	do_write(std::string && msg, bool disconnect_after) {
//	if ( !started_) {
//		log("do_write: returned, not started", true);
//		return;
//	}
//	log("do_write");
//#ifdef _USE_ASSERTS_
//	assert( msg.size() < max_msg );
//#endif
//	std::copy(msg.begin(), msg.end(), write_buffer_);
////	sock_.async_write_some( buffer(write_buffer_, msg.size()), 
////		MEM_FN3(on_write,_1,_2,disconnect_after));
//	async_write(sock_, buffer(write_buffer_, msg.size()), transfer_exactly(msg.size()), 
//		MEM_FN3(on_write,_1,_2,disconnect_after));
//}
void talk_to_client::	do_ping() {
	log("do_ping " + boost::to_string(last_ping));
	//write("ping " + boost::to_string(last_ping));
	//char ans[sizeof(short)*2 + 1];
	std::vector<char> ans(sizeof(short)*2 + sizeof(char), 0);
	SHORT(ans[0]) = sizeof(short) + sizeof(char);
	ans[sizeof(short)] = static_cast<char>(REQUEST_TYPE::PING);
	SHORT(ans[sizeof(short) + sizeof(char)]) = short(last_ping);
	do_write(std::move( ans ));
}
void talk_to_client::	do_ask_clients() {
	assert(false && "do_ask_clients");
	return;
	//log("do_ask_clients");
	//do_write("ask_clients\n");
}
void talk_to_client::	do_add_outcoming_msg(const std::vector<char>& msg) {
	boost::mutex::scoped_lock lk(m_Mutex);
#ifdef _USE_ASSERTS_
	assert(!msg.empty() && "Try to add empty message");
#endif
	m_OutcomingMsgs.push_back(msg);
}
void talk_to_client::	do_add_outcoming_msg(std::vector<char>&& msg) {
	boost::mutex::scoped_lock lk(m_Mutex);
#ifdef _USE_ASSERTS_
	assert(!msg.empty() && "Try to add empty message");
#endif
	m_OutcomingMsgs.push_back(msg);
}

size_t talk_to_client::read_complete(const boost::system::error_code & err, size_t bytes) {
	if ( err) return 0;
	bool found = std::find(read_buffer_, read_buffer_ + bytes, '\n') < read_buffer_ + bytes;
	// we read one-by-one until we get to enter, no buffering
	return found ? 0 : 1;
}

int talk_to_client::get_id() const {
	return m_Client->get_id();
}
void talk_to_client::set_id(int id) {
	m_Client = _SERVER.get_client(id);
}
Client::ptr talk_to_client::get_player() const {
	return m_Client;
}
void talk_to_client::set_player(Client::ptr p) {
	m_Client = p;
}

std::string talk_to_client::username() const {
	return username_;
}
//void talk_to_client::set_active(bool active) {
//	m_bIsActive = active;
//	//do_add_outcoming_msg(active ? TURN_BEGIN : TURN_END);
//}

//void talk_to_client:: on_check_ping() {
//	boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
//	int last_ping = (now - last_sended).total_milliseconds();
//	log("on_check_ping");
//	if ( last_ping > 5000) {
//		std::cout << "stopping " << username_ << " due to a high ping" << std::endl;
//		stop();
//	}
//	else if (last_ping > 1000) {
//		std::cout << "ping " << ping << " from " << username_ << std::endl;
//	}
//	last_sended = boost::posix_time::microsec_clock::local_time();
//}