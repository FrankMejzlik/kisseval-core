#pragma once

#include "common.h"

#if USE_SQLITE

#include <string>
#include <vector>

#include <sqlite3.h>

namespace image_ranker
{
class Database
{
 public:
  Database() = delete;
  Database(const std::string& host, size_t port, const std::string& username, const std::string& password,
           const std::string& db_name);
  ~Database() noexcept;

  size_t GetLastId() const;
  std::string GetErrorDescription() const;
  size_t GetErrorCode() const;
  const std::string& GetDbName() const { return _db_name; };

  size_t EstablishConnection() { return 0_z; };

  size_t NoResultQuery(const std::string& query) const;
  std::pair<size_t, std::vector<std::vector<std::string> > > ResultQuery(const std::string& query) const;

  std::string EscapeString(const std::string& stringToEscape) const;

 private:
  sqlite3* _db;

  std::string_view _host;
  size_t _port;
  std::string _username;
  std::string _password;
  std::string _db_name;
};

}  // namespace image_ranker

#else

#include <string.h>
#include <iostream>
#include <vector>

#include <mysql.h>
#include "config.h"

namespace image_ranker
{
class Database
{
  // Structs
 public:
  enum Type
  {
    cPrimary,
    cSecondary
  };

 public:
  Database() = delete;
  Database(std::string_view host, size_t port, std::string_view username, std::string_view password,
           std::string_view dbName);
  ~Database() noexcept;

  size_t GetLastId() const;
  std::string GetErrorDescription() const;
  size_t GetErrorCode() const;
  const std::string& GetDbName() const { return _dbName; };

  std::string EscapeString(const std::string& stringToEscape) const;
  size_t EstablishConnection();
  void CloseConnection();

  size_t NoResultQuery(std::string_view query) const;
  std::pair<size_t, std::vector<std::vector<std::string> > > ResultQuery(std::string_view query) const;

 private:
  MYSQL* _mysqlConnection;

  std::string_view _host;
  size_t _port;
  std::string _username;
  std::string _password;
  std::string _dbName;
};

}  // namespace image_ranker

#endif