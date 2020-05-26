#include "Database.h"

using namespace image_ranker;

Database::Database(const std::string& db_filepath) : _db_fpth(db_filepath)
{
  int rc;

  rc = sqlite3_open(_db_fpth.c_str(), &_db);
  if (rc)
  {
    sqlite3_close(_db);
    std::string msg{ "Can't open database: " + std::string{ sqlite3_errmsg(_db) } };
    LOGE(msg);
    THROW_PROD("An error occured!");
  }
};

Database::~Database() noexcept { sqlite3_close(_db); }

std::string Database::GetErrorDescription() const { return std::string(sqlite3_errmsg(_db)); }

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
    THROW_PROD("An error occured!");
  }

  int rc{ sqlite3_step(stmt) };
  if (rc != SQLITE_DONE)
  {
    auto msg{ "SQL statement `"s + std::string(sqlite3_sql(stmt)) +
              "` failed with error: " + std::string(sqlite3_errmsg(_db)) };
    LOGE(msg);
    THROW_PROD("An error occured!");
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
      auto c = (char*)sqlite3_column_text(stmt, static_cast<int>(i));
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
