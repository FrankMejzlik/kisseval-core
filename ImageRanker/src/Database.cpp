#include "Database.h"

using namespace image_ranker;

#if USE_SQLITE

Database::Database(const std::string& host, size_t port, const std::string& username, const std::string& password,
                   const std::string& db_name)
    : _host(host), _port(port), _username(username), _db_name(db_name), _password(password)
{
  int rc;

  rc = sqlite3_open(DB_FILENAME, &_db);
  if (rc)
  {
    sqlite3_close(_db);
    std::string msg{ "Can't open database: " + std::string{ sqlite3_errmsg(_db) } };
    LOGE(msg);
  }
};

Database::~Database() noexcept { sqlite3_close(_db); }

size_t Database::GetErrorCode() const { return static_cast<size_t>(sqlite3_errcode(_db)); }

size_t Database::NoResultQuery(const std::string& query) const
{
  sqlite3_stmt* stmt;
  // compile sql statement to binary
  if (sqlite3_prepare_v2(_db, query.c_str(), -1, &stmt, NULL) != SQLITE_OK)
  {
    printf("ERROR: while compiling sql: %s\n", sqlite3_errmsg(_db));
    sqlite3_close(_db);
    sqlite3_finalize(stmt);
    return GetErrorCode();
  }

  int rc{ sqlite3_step(stmt) };
  if (rc != SQLITE_DONE)
  {
    auto msg{ "SQL statement `"s + std::string(sqlite3_sql(stmt)) +
              "` failed with error: " + std::string(sqlite3_errmsg(_db)) };
    LOGE(msg);
  }

  // release resources
  sqlite3_finalize(stmt);

  return GetErrorCode();
}

std::string Database::EscapeString(const std::string& stringToEscape) const
{
  char* zSQL = sqlite3_mprintf("%q", stringToEscape.c_str());

  return std::string(zSQL);
}

size_t Database::GetLastId() const { return static_cast<size_t>(sqlite3_last_insert_rowid(_db)); }

std::pair<size_t, std::vector<std::vector<std::string>>> Database::ResultQuery(const std::string& query) const
{
  // Send query to DB and get result
  std::vector<std::vector<std::string>> result;

  sqlite3_stmt* stmt;
  // compile sql statement to binary
  if (sqlite3_prepare_v2(_db, query.c_str(), -1, &stmt, NULL) != SQLITE_OK)
  {
    printf("ERROR: while compiling sql: %s\n", sqlite3_errmsg(_db));
    sqlite3_close(_db);
    sqlite3_finalize(stmt);
    return std::pair(sqlite3_errcode(_db), result);
  }

  int ret_code = 0;
  while ((ret_code = sqlite3_step(stmt)) == SQLITE_ROW)
  {
    // execute sql statement, and while there are rows returned, print ID
    auto cnt = sqlite3_data_count(stmt);

    std::vector<std::string> r;
    for (size_t i{ 0_z }; i < cnt; ++i)
    {
      auto c = (char*)sqlite3_column_text(stmt, i);
      r.emplace_back(std::string(c));
    }
    result.emplace_back(std::move(r));
  }
  if (ret_code != SQLITE_DONE)
  {
    // this error handling could be done better, but it works
    printf("ERROR: while performing sql: %s\n", sqlite3_errmsg(_db));
    printf("ret_code = %d\n", ret_code);
  }

  // release resources
  sqlite3_finalize(stmt);

  return std::make_pair(GetErrorCode(), result);
}

#else

Database::Database(std::string_view host, size_t port, std::string_view username, std::string_view password,
                   std::string_view dbName)
    : _host(host),
      _port(port),
      _username(username),
      _dbName(dbName),
      _password(password),
      _mysqlConnection(mysql_init(NULL)){};

Database::~Database() noexcept { mysql_close(_mysqlConnection); }

std::string Database::GetErrorDescription() const
{
  return std::string{ "Error(" + mysql_errno(_mysqlConnection) + std::string(") [") + mysql_sqlstate(_mysqlConnection) +
                      std::string("]\n ") + mysql_error(_mysqlConnection) };
}

size_t Database::GetErrorCode() const { return static_cast<size_t>(mysql_errno(_mysqlConnection)); }

size_t Database::EstablishConnection()
{
  _mysqlConnection = mysql_real_connect(_mysqlConnection, _host.data(), _username.data(), _password.data(),
                                        _dbName.data(), static_cast<unsigned long>(_port), "NULL", 0);

  if (!_mysqlConnection)
  {
    fprintf(stderr, "Failed to connect to database: Error: %s\n", mysql_error(_mysqlConnection));
  }

  bool reconnect = true;
  mysql_options(_mysqlConnection, MYSQL_OPT_RECONNECT, &reconnect);

  // If connection failed
  if (!_mysqlConnection)
  {
    // Close connection
    CloseConnection();

    return GetErrorCode();
  }

  // Connected to database
  return GetErrorCode();
}

void Database::CloseConnection() { mysql_close(_mysqlConnection); }

std::string Database::EscapeString(const std::string& stringToEscape) const
{
  char buffer[1024];

  mysql_real_escape_string(_mysqlConnection, buffer, stringToEscape.data(),
                           static_cast<unsigned long>(stringToEscape.size()));

  return std::string{ buffer };
}

size_t Database::NoResultQuery(std::string_view query) const
{
  // Send query to DB and get result
  auto result{ mysql_real_query(_mysqlConnection, query.data(), static_cast<unsigned long>(query.length())) };

  // If error executing query
  if (result != 0)
  {
    return GetErrorCode();
  }

  //(unsigned long) mysql_affected_rows(mysql));

  return GetErrorCode();
}

size_t Database::GetLastId() const { return mysql_insert_id(_mysqlConnection); }

std::pair<size_t, std::vector<std::vector<std::string>>> Database::ResultQuery(std::string_view query) const
{
  // Send query to DB and get result
  auto result{ mysql_real_query(_mysqlConnection, query.data(), static_cast<unsigned long>(query.length())) };

  // If error executing query
  if (result != 0)
  {
    return std::make_pair(GetErrorCode(), std::vector<std::vector<std::string>>());
  }

  MYSQL_RES* data = mysql_store_result(_mysqlConnection);
  size_t numRows = (size_t)mysql_num_rows(data);
  size_t numCols = (size_t)mysql_num_fields(data);

  std::vector<std::vector<std::string>> retData;
  retData.reserve(numRows);

  MYSQL_ROW rawRow;

  // Process all rows
  while ((rawRow = mysql_fetch_row(data)))
  {
    std::vector<std::string> row;
    row.reserve(numCols);

    for (size_t i = 0ULL; i < numCols; ++i)
    {
      // If null value
      if (!rawRow[i])
      {
        row.push_back("");
        continue;
      }

      row.push_back(rawRow[i]);
    }

    retData.push_back(row);
  }

  mysql_free_result(data);

  return std::make_pair(GetErrorCode(), retData);
}

#endif