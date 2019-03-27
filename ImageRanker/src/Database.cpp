#include "Database.h"



Database::Database() :
  _mysqlConnection(mysql_init(NULL))
{};


Database::~Database() noexcept
{
  mysql_close(_mysqlConnection);
}

std::string Database::GetErrorDescription() const
{
  return std::string{
    "Error(" + mysql_errno(_mysqlConnection) + std::string(") [") 
    + mysql_sqlstate(_mysqlConnection) + std::string("]\n ") +  mysql_error(_mysqlConnection)
  };
}

size_t Database::GetErrorCode() const
{
  return static_cast<size_t>(mysql_errno(_mysqlConnection));
}

size_t Database::EstablishConnection()
{ 
#if USE_REMOTE_DB

  _mysqlConnection = mysql_real_connect(
    _mysqlConnection,
    "li1504-72.members.linode.com",
    "image-ranker", "s5XurJ3uS3E52Gzm",
    "image-ranker-collector-data", 3306,
    "NULL", 0
  );

#else

  _mysqlConnection = mysql_real_connect(
    _mysqlConnection,
    "localhost",
    "image-ranker", "s5XurJ3uS3E52Gzm",
    "image-ranker-collector-data", 3306,
    "NULL", 0
  );

#endif

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


void Database::CloseConnection()
{
  mysql_close(_mysqlConnection);
}


std::string Database::EscapeString(const std::string& stringToEscape) const
{
  char buffer[1024];

  mysql_real_escape_string(_mysqlConnection, buffer, stringToEscape.data(), stringToEscape.size());

  return std::string{ buffer };
}

size_t Database::NoResultQuery(std::string_view query)
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

std::pair< size_t, std::vector< std::vector<std::string>>> Database::ResultQuery(std::string_view query)
{
  // Send query to DB and get result
  auto result{ mysql_real_query(_mysqlConnection, query.data(), static_cast<unsigned long>(query.length())) };

  // If error executing query
  if (result != 0)
  {
    return std::make_pair(GetErrorCode(), std::vector< std::vector<std::string>>());
  }


  MYSQL_RES* data = mysql_store_result(_mysqlConnection);
  size_t numCols = (size_t)mysql_num_fields(data);

  std::vector< std::vector<std::string>> retData;

  MYSQL_ROW rawRow;

  // Process all rows
  while ((rawRow = mysql_fetch_row(data)))
  {
    std::vector<std::string> row;
    row.resize(numCols);

    for(size_t i = 0ULL; i < numCols; ++i)
    {
      row.push_back(row[i]);
    }

    retData.push_back(row);
  }

  mysql_free_result(data);

  return std::make_pair(GetErrorCode(), retData);
}