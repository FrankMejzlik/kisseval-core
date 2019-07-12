#pragma once

#include <vector>
#include <cstddef>

#include "config.h"

using size_t = std::size_t;


enum class eTempQueryOpOutter
{
  cSum,
  cProduct
};

enum class eTempQueryOpInner
{
  cSum,
  cProduct,
  cMax
};

/*
 * User defined literals
 */
constexpr size_t operator ""_z(unsigned long long int x)
{
  return static_cast<size_t>(x);
}

using Clause = std::vector<std::pair<bool, size_t>>;
using CnfFormula = std::vector<Clause>;
using InteractiveSearchAction = std::tuple<size_t, size_t, size_t>;

enum class InteractiveSearchOrigin
{
  cDeveloper,
  cPublic
};

enum class RankingModelId
{
  cBooleanBucket = 1,
  cBooleanExtended = 2,
  cViretBase = 3
};

enum class NetDataTransformation
{
  cSoftmax = 100,
  cXToTheP = 200,
  cSine = 300
};

enum class QueryOriginId
{
  cDeveloper = 0,
  cPublic = 1,
  cManaged = 2,

  cAll = 999,

  // Simulated variants
  cDeveloperSimulated = 10000,
  cPublicSimulated = 10001,
  cManagedSimulated = 10002,
  cAllSimulated = 10999,

  // Extended variants
  cDeveloperExtended = 20000,
  cPublicExtended = 20001,
  cManagedExtended = 20002,
  cAllExtended = 20999

};



/*!
*  FORMAT:
*  ========================================================
* (x) cBooleanBucket
* --- 
* 
*    0 => keyword frequency handling
*    1 => trueTreshold
*    2 => inBucketSorting
*    3 => outter temporal query operation (\ref eTempQueryOpOutter)
*       0: sum
*       1: product
*    4 => inner temporal query operation  (\ref eTempQueryOpInner)
*       0: sum
*       1: product
*       2: max
* 
*  ========================================================
*  (x) cBooleanExtended:
*  ---
* 
*  ========================================================
*  (x) cViretBase
*  ----
*    0 => keyword frequency handling
*    1 => ignoreTreshold
*    2 => rankCalcMethod
*       cMultSum = 0,
*       cMultMax = 1,
*       cSumSum = 2,
*       cSumMax = 3,
*       cMaxMax= 4
*    3 => outter temporal query operation (\ref eTempQueryOpOutter)
*       0: sum
*       1: product
*    4 => inner temporal query operation (\ref eTempQueryOpInner)
*       0: sum
*       1: product
*       2: max
*/
using AggModelSettings = std::vector<std::string>;


/*!
* FORMAT:
*  1: cSoftmax
*  2: cXToTheP:
*    0 => Exponent fot x^p postscale transformation
*    1 => 
*       0: SUM accumulated precoputed hypernyms
*       1: MAX accumulated precoputed hypernyms
*/
using NetDataTransformSettings = std::vector<std::string>;


/*!
* FORMAT:
*  0 => Simulated user exponent
*/
using SimulatedUserSettings = std::vector<std::string>;





using AggregationVector = std::vector<float>;
using TestSettings = std::tuple<NetDataTransformation, RankingModelId, QueryOriginId, AggModelSettings, NetDataTransformSettings>;

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
using UserImgQueryRaw = std::tuple<size_t, std::vector<size_t>>;


//! (datasourceID, percentageofAll)
using UserAccuracyChartDataMisc = std::tuple<size_t, float>;

using UserAccuracyChartData = std::pair<UserAccuracyChartDataMisc, ChartData>;
//! Structure for returning results of queries
struct QueryResult
{
  QueryResult() :
    m_targetImageRank(0ULL)
  {
  }

  size_t m_targetImageRank;
};


class SimulatedUser
{
public:
  SimulatedUser():
    m_exponent(1)
  { }

public:
  int m_exponent;
};