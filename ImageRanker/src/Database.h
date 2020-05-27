#pragma once

#include <string>
#include <vector>

#include <sqlite3.h>

#include "common.h"

namespace image_ranker
{
/**
 * Class abstracting database for collected data storage.
 *
 * It is implemented using SQLite database engine:
 * https://www.sqlite.org/
 */
class [[nodiscard]] Database
{
  /*
   * Methods
   */
 public:
  /**
   * Escapes the given string and returns new copy of escaped one.
   *
   * \param   String to escape.
   * \return  std::string   New copy of escaped string.
   */
  [[nodiscard]] static std::string escape_str(const std::string& raw_str);

  // This is non-copyable due to `std::unique_ptr`.
  Database() = delete;
  Database(const Database& other) = delete;
  Database(Database && other) = default;
  Database& operator=(const Database& other) = delete;
  Database& operator=(Database&& other) = default;
  ~Database() noexcept = default;

  /**
   * Main construcotr.
   *
   * \param Filepath to database file.
   */
  Database(const std::string& db_filepath);

  /**
   * Returns the ID of last row that was inserted into the database.
   *
   * \return  size_t Last inserted ID.
   */
  [[nodiscard]] size_t get_last_inserted_ID() const;

  /**
   * Gets message describin last database error that occured.
   *
   * \return  std::string String with error message.
   */
  [[nodiscard]] std::string get_last_error_msg() const;

  /**
   * Gets code of the last error that occured.
   *
   * \return  size_t Error code.
   */
  [[nodiscard]] size_t get_last_err_code() const;

  /**
   * Runs SQL against the database and returns return code.
   *
   * \param   SQL query to run.
   * \return  size_t Return code (0 means success).
   */
  [[nodiscard]] size_t no_result_query(const std::string& query) const;

  /**
   * Runs SQL against the database and returns return results.
   *
   * \param   SQL query to run.
   * \return  std::pair<size_t, std::vector<std::vector<std::string>>>
   *          Pair with return code and rows itself.
   */
  [[nodiscard]] std::pair<size_t, std::vector<std::vector<std::string>>> result_query(const std::string& query) const;

  /*
   * Member variables
   */
 private:
  /** Ptr to the database instance */
  std::unique_ptr<sqlite3, void (*)(sqlite3*)> _p_db;

  /** Filepath of source database file */
  std::string _db_fpth;
};

}  // namespace image_ranker
