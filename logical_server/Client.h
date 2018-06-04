#pragma once
#ifndef _CLIENT_H_
#define _CLIENT_H_

class talk_to_client;

class Client: public std::enable_shared_from_this<Client>, public boost::noncopyable {

	typedef boost::shared_ptr<talk_to_client> talk_to_client_ptr;

	int			m_ID;
	talk_to_client_ptr	m_cl;

	bool	m_bIsOnline;

	Client(int client_id, talk_to_client_ptr cl, bool online);

public:

	typedef boost::shared_ptr<Client> ptr;

	static Client::ptr Client::create(int client_id, talk_to_client_ptr cl, bool online = true);

	bool is_online() const {return m_bIsOnline;}
	void set_online(bool online);

	void set_client(talk_to_client_ptr cl) { m_cl = cl; }
	talk_to_client_ptr get_client() const {return m_cl;}

	void set_id(int id) { m_ID = id; }
	int get_id() const {return m_ID;}

};

#endif

