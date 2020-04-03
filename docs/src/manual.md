@page manual Manual

@subpage overview "Overview"  
@subpage build "How to build"  
@subpage db_data "Database data"

@page overview Overview

# TBA

@page building_syengine Building SYEngine

@note
    TBA

@section msvc MSVC


Project folder structure should look something like: 
@code
SYEngine/ 
├── Diagnostics/ 
│   └── ...  
├── Engine/  
│   └── ...  
@endcode

@page db_data Database data
Database uses simple structure that is mainly designed to be human readable during the debugging process therefore is avoids cryptic integral codes with speacial meaning and uses human readable strings as values whenever possible e.g. `V3C1_20k_BoW_Jan_2020`.

# Tables
## `user_queries`
Stores all queries that are gathered from users in the web UI.

|Column name|Column type| Column description
|---|---|---|
|ID             |INT(11)        |Unique AI ID of this record.|
|user_query     |LONGTEXT       |Encoded version of the query. Encoded version depends on vocabulary it is created upon.|
|readable_user_query     |LONGTEXT       |Human readable version of the query.|
|vocabulary_ID     |VARCHAR(45)       |String ID of vocabulary this query was collected for.|
|data_pack_ID    |VARCHAR(45)       |Data pack that was used for autocomplete results during query creation.|
|model_options   |LONGTEXT       | String describing options of model that was used for autocomplete results during query creation.|
|imageset_ID   |VARCHAR(45)      | String ID identifying imageset this frame belongs to.|
|target_frame_ID   |INT(11)       |Frame that was used as the target.|
|with_example_images   |TINYINT(1)       |Flag indicating that user was presented with example images for each autocomplete keyword in autocomplete popup.|
|user_level   |TINYINT(4)       | Describes level of user that created this query, the higher the more experienced user in video browsing.|
|manually_validated   |TINYINT(1)       |True if this query has been manually checked that it's not a garbage query.|
|session_ID   |VARCHAR(255)       | Session ID of the user.|
|created   |TIMESTAMP       | Timestamp this record was created.|



## `search_sessions`
Stores all search sessions done by users in the web UI.

|Column name|Column type| Column description
|---|---|---|
|ID             |INT(11)        |Unique AI ID of this record.|
|target_frame_ID   |INT(11)       |Frame that was used as the target.|
|vocabulary_ID     |VARCHAR(45)       |String ID of vocabulary this query was collected for.|
|data_pack_ID    |VARCHAR(45)       |Data pack that was used for autocomplete results during query creation.|
|model_options   |LONGTEXT       | String describing options of model that was used for autocomplete results during query creation.|
|duration   |INT(11)       |Duration (in seconds) of this search session.|
|user_level   |TINYINT(4)       | Describes level of user that created this query, the higher the more experienced user in video browsing.|
|result   |TINYINT(1)       |True if this session ended with user finding the target.|
|manually_validated   |TINYINT(1)       |True if this query has been manually checked that it's not a garbage query.|
|session_ID   |VARCHAR(255)       | Session ID of the user.|
|created   |TIMESTAMP       | Timestamp this record was created.|

## `search_sessions_actions`
Stores all metadata for search sessions stored in `search_sessions` table.

|Column name|Column type| Column description
|---|---|---|
|search_session_ID             |INT(11)        |Unique ID of the search session this action belongs to.|
|action_index             |INT(11)        |Index of this action in this search session.|
|query_index   |INT(11)       |Index of query that this action happened at (for temporal queries).|
|action   |VARCHAR(45)       |String describing action that happened.|
|operand   |INT(11)       |Object that this action was done on.|
|result_target_rank   |INT(11)       |Rank of the target frame that this action resulted in.|
|time   |INT(255)       | Time elapsed since the start of this search session (in seconds). |
