#pragma once

#include <string.h>
#include <iostream>
#include <vector>

#include "config.h"
#include <mysql.h>


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
  Database(
    std::string_view host,
    size_t port,
    std::string_view username,
    std::string_view password,
    std::string_view dbName
  );
  ~Database() noexcept;


  std::string GetErrorDescription() const;
  size_t GetErrorCode() const;

  std::string EscapeString(const std::string& stringToEscape) const;
  size_t EstablishConnection();
  void CloseConnection();

  size_t NoResultQuery(std::string_view query);
  std::pair<size_t, std::vector<std::vector<std::string>>> ResultQuery(std::string_view query);

  private:
    MYSQL* _mysqlConnection;

    std::string_view _host;
    size_t _port;
    std::string _username;
    std::string _password;
    std::string _dbName;

};




/*query = "DROP TABLE IF EXISTS affected_rows";
if (mysql_real_query(mysql, query, strlen(query)))
show_error(mysql);*/

//query= "CREATE TABLE affected_rows (id int not null, my_name varchar(50),"
//  "PRIMARY KEY(id))";

/*query = "INSERT INTO `test`.`images` (id, src) VALUES (16, \"src5\");";
if (mysql_real_query(mysql, query, strlen(query)))
{
show_error(mysql);
}*/


///* Affected rows with INSERT statement */
//query= "INSERT INTO affected_rows VALUES (1, \"First value\"),"
//  "(2, \"Second value\")";
//if (mysql_real_query(mysql, query, strlen(query)))
//  show_error(mysql);
//printf("Affected_rows after INSERT: %lu\n",
//  (unsigned long) mysql_affected_rows(mysql));

///* Affected rows with REPLACE statement */
//query= "REPLACE INTO affected_rows VALUES (1, \"First value\"),"
//  "(2, \"Second value\")";
//if (mysql_real_query(mysql, query, strlen(query)))
//  show_error(mysql);
//printf("Affected_rows after REPLACE: %lu\n",
//  (unsigned long) mysql_affected_rows(mysql));

///* Affected rows with UPDATE statement */
//query= "UPDATE affected_rows SET id=1 WHERE id=1";
//if (mysql_real_query(mysql, query, strlen(query)))
//  show_error(mysql);
//printf("Affected_rows after UPDATE: %lu\n",
//  (unsigned long) mysql_affected_rows(mysql));

//query= "UPDATE affected_rows SET my_name=\"Monty\" WHERE id=1";
//if (mysql_real_query(mysql, query, strlen(query)))
//  show_error(mysql);
//printf("Affected_rows after UPDATE: %lu\n",
//  (unsigned long) mysql_affected_rows(mysql));

///* Affected rows after select */
//query= "SELECT id, my_name FROM affected_rows";
//if (mysql_real_query(mysql, query, strlen(query)))
//  show_error(mysql);
//result= mysql_store_result(mysql);
//printf("Affected_rows after SELECT and storing result set: %lu\n",
//  (unsigned long) mysql_affected_rows(mysql));
//mysql_free_result(result);

///* Affected rows with DELETE statement */
//query= "DELETE FROM affected_rows";
//if (mysql_real_query(mysql, query, strlen(query)))
//  show_error(mysql);
//printf("Affected_rows after DELETE: %lu\n",
//  (unsigned long) mysql_affected_rows(mysql));