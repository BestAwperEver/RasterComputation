#include "StdAfx.h"
#include "Client.h"
#include "tcp_connection.h"
#include "tcp_server.h"

#define INT(x) *( (int*)(&x) )

Client::ptr Client::create(int client_id, talk_to_client_ptr cl, bool online) {
	Client::ptr new_client(new Client(client_id, cl, online));
	return new_client;
}
Client::Client(int client_id, talk_to_client_ptr cl, bool online)
	: m_ID(client_id)
	, m_cl(cl) 
	, m_bIsOnline(online)
{
}

void Client::set_online(bool online) {
	m_bIsOnline = online;
}