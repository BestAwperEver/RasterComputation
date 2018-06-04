#include "ClientDataBase.h"

const int ClientsCount = 6;
char* registered_clients[] = {"Radagast", "Patamaniack", "Dart", "Imran", "Lanin", "Redgar", 0};

ClientDataBase::ClientDataBase() {}
ClientDataBase::~ClientDataBase() {}

ClientDataBase& ClientDataBase::getInstance() {
	static ClientDataBase instance;
	return instance;
}

bool ClientDataBase::do_compare(char reg[], const std::string& con) const {
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

int ClientDataBase::getClientID(const ClientDataBase::String& client_name) const {
	for (char **ch = registered_clients, k = 0; *ch; ++ch, ++k) {
		if (do_compare(*ch,client_name)) {
			return k+1;
		}
	}
	return -1;
}

ClientDataBase::String ClientDataBase::getClientName(int client_id) const {
	return registered_clients[client_id-1];
}
unsigned int ClientDataBase::getClientsCount() const {
	return ClientsCount;
}