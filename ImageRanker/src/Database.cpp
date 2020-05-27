#include "Database.h"

using namespace image_ranker;

Database::Database(const std::string& db_filepath)
    : _db_fpth(db_filepath),
      // Add custom deleter for this pointer
      _p_db(nullptr, [](sqlite3* ptr) { sqlite3_close(ptr); })
{
  int rc{ ERR_VAL<int>() };

  // Instantiate database instance
  sqlite3* p_db{ nullptr };
  rc = sqlite3_open(_db_fpth.c_str(), &p_db);
  _p_db.reset(p_db);

  if (rc > 0)
  {
    std::string msg{ "Can't open database: " + std::string{ sqlite3_errmsg(_p_db.get()) } };
    LOGE(msg);
    PROD_THROW("An error occured!");
  }
};

std::string Database::get_last_error_msg() const { return std::string{ sqlite3_errmsg(_p_db.get()) }; }

size_t Database::get_last_err_code() const { return static_cast<size_t>(sqlite3_errcode(_p_db.get())); }

size_t Database::no_result_query(const std::string& query) const
{
  sqlite3_stmt* stmt;
  // compile sql statement to binary
  if (sqlite3_prepare_v2(_p_db.get(), query.c_str(), -1, &stmt, NULL) != SQLITE_OK)
  {
    printf("ERROR: while compiling sql: %s\n", sqlite3_errmsg(_p_db.get()));
    sqlite3_close(_p_db.get());
    sqlite3_finalize(stmt);
    return get_last_err_code();
    PROD_THROW("An error occured!");
  }

  int rc{ sqlite3_step(stmt) };
  if (rc != SQLITE_DONE)
  {
    auto msg{ "SQL statement `"s + std::string(sqlite3_sql(stmt)) +
              "` failed with error: " + std::string(sqlite3_errmsg(_p_db.get())) };
    LOGE(msg);
    PROD_THROW("An error occured!");
  }

  // release resources
  sqlite3_finalize(stmt);

  return get_last_err_code();
}

std::string Database::escape_str(const std::string& stringToEscape) const
{
  char* zSQL = sqlite3_mprintf("%q", stringToEscape.c_str());

  return std::string(zSQL);
}

size_t Database::get_last_inserted_ID() const { return static_cast<size_t>(sqlite3_last_insert_rowid(_p_db.get())); }

std::pair<size_t, std::vector<std::vector<std::string>>> Database::result_query(const std::string& query) const
{
  // Send query to DB and get result
  std::vector<std::vector<std::string>> result;

  sqlite3_stmt* stmt;
  // compile sql statement to binary
  if (sqlite3_prepare_v2(_p_db.get(), query.c_str(), -1, &stmt, NULL) != SQLITE_OK)
  {
    printf("ERROR: while compiling sql: %s\n", sqlite3_errmsg(_p_db.get()));
    sqlite3_close(_p_db.get());
    sqlite3_finalize(stmt);
    return std::pair(sqlite3_errcode(_p_db.get()), result);
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
    printf("ERROR: while performing sql: %s\n", sqlite3_errmsg(_p_db.get()));
    printf("ret_code = %d\n", ret_code);
  }

  // release resources
  sqlite3_finalize(stmt);

  return std::make_pair(get_last_err_code(), result);
}
