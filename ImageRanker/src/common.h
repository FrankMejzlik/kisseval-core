#pragma once

#include <vector>

#include "config.h"

using Clause = std::vector<std::pair<bool, size_t>>;
using CnfFormula = std::vector<Clause>;


enum class RankingModelId
{
  cBooleanBucket = 1,
  cBooleanExtended = 2,
  cViretBase = 3
};

enum class AggregationId
{
  cSoftmax = 100,
  cXToTheP = 200,
  cSine = 300
};

enum class QueryOriginId
{
  cDeveloper = 0,
  cPublic = 1,
  cManaged = 2
};

/*!
* FORMAT:
*  1: cBooleanBucket:
*    0 => trueTreshold
*    1 => inBucketSorting
*  2: cBooleanExtended:
*  3: cViretBase:
*    0 => ignoreTreshold
*    1 => rankCalcMethod
*      0: Multiply & (Add |)
*      1: Add only
*/
using ModelSettings = std::vector<std::string>;

/*!
* FORMAT:
*  1: cSoftmax
*  2: cXToTheP:
*    0 => m_exponent
*  3: cSine:
*/
using AggregationSettings = std::vector<std::string>;

using AggregationVector = std::vector<float>;
using TestSettings = std::tuple<AggregationId, RankingModelId, QueryOriginId, ModelSettings, AggregationSettings>;

using Buffer = std::vector<std::byte>;

//! This is returned to front end app when some quesries are submited
//! <SessionID, image filename, user keywords, net <keyword, probability> >
using GameSessionQueryResult = std::tuple<std::string, std::string, std::vector<std::string>, std::vector<std::pair<std::string, float>>>;

//! Array of those is submited from front-end app game
using GameSessionInputQuery = std::tuple<std::string, size_t, std::string>;

using ImageReference = std::pair<size_t, std::string>;

/*! <wordnetID, keyword, description> */
using KeywordReferences = std::vector<std::tuple<size_t, std::string, std::string>>;

/*! <wordnetID, keyword, description> */
using KeywordData = std::tuple<size_t, std::string, std::string>;

/*!
 * Output data for drawing chart
 */
using ChartData = std::vector <std::pair<uint32_t, uint32_t>>;

using UserImgQuery = std::tuple<size_t, CnfFormula>;

//! Structure for returning results of queries
struct QueryResult
{
  QueryResult() :
    m_targetImageRank(0ULL)
  {
  }

  size_t m_targetImageRank;
};