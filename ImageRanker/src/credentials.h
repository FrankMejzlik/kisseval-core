#pragma once

/*!
 * PRIMARY DB
 */
//! 0: localhost - testing
#if PRIMARY_DB_ID == 0

#define PRIMARY_DB_HOST "localhost"
#define PRIMARY_DB_PORT 3306ULL
#define PRIMARY_DB_USERNAME "appImageRanker1"
#define PRIMARY_DB_PASSWORD "6zb9OhzCyojh0Ld4h20C"
#define PRIMARY_DB_DB_NAME "db_ir"

//! 1: localhost - production copy
#elif PRIMARY_DB_ID == 1

#define PRIMARY_DB_HOST "localhost"
#define PRIMARY_DB_PORT 3306ULL
#define PRIMARY_DB_USERNAME "appImageRanker1"
#define PRIMARY_DB_PASSWORD "6zb9OhzCyojh0Ld4h20C"
#define PRIMARY_DB_DB_NAME "image-ranker-collector-data2"

//! 2: localhost 2
#elif PRIMARY_DB_ID == 2

#define PRIMARY_DB_HOST "localhost"
#define PRIMARY_DB_PORT 3306ULL
#define PRIMARY_DB_USERNAME "appImageRanker1"
#define PRIMARY_DB_PASSWORD "s5XurJ3uS3E52Gzm"
#define PRIMARY_DB_DB_NAME "image-ranker-collector-data2"

//! 3: linode data2
#elif PRIMARY_DB_ID == 3

#define PRIMARY_DB_HOST "herkules.ms.mff.cuni.cz"
#define PRIMARY_DB_PORT 8082ULL
#define PRIMARY_DB_USERNAME "appImageRanker1"
#define PRIMARY_DB_PASSWORD "6zb9OhzCyojh0Ld4h20C"
#define PRIMARY_DB_DB_NAME "ir_cikm_rev1"

//! 4: linode data2
#elif PRIMARY_DB_ID == 4

#define PRIMARY_DB_HOST "herkules.ms.mff.cuni.cz"
#define PRIMARY_DB_PORT 8082ULL
#define PRIMARY_DB_USERNAME "appImageRanker1"
#define PRIMARY_DB_PASSWORD "6zb9OhzCyojh0Ld4h20C"
#define PRIMARY_DB_DB_NAME "image-ranker-collector-data2"

#endif
