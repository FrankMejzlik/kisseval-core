
#include "ImageRanker.h"



/*********************
  Ranking Models
 *********************/

float BooleanBucketModel::m_trueTresholdFrom{ 0.01f };
float BooleanBucketModel::m_trueTresholdTo{ 0.9f };
float BooleanBucketModel::m_trueTresholdStep{ 0.1f };
std::vector<float> BooleanBucketModel::m_trueTresholds;
std::vector<uint8_t> BooleanBucketModel::m_inBucketOrders{ {0,1,2} };

float ViretModel::m_trueTresholdFrom{ 0.01f };
float ViretModel::m_trueTresholdTo{ 0.9f };
float ViretModel::m_trueTresholdStep{ 0.1f };
std::vector<float> ViretModel::m_trueTresholds;
std::vector<uint8_t> ViretModel::m_queryOperations{ {0,1} };


ImageRanker::ImageRanker(
  const std::string& imagesPath,
  const std::string& rawNetRankingFilepath,
  const std::string& keywordClassesFilepath,
  const std::string& softmaxFilepath,
  const std::string& deepFeaturesFilepath,
  const std::string& imageToIdMapFilepath,
  size_t idOffset,
  Mode mode
) :
  _primaryDb(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME),
  _secondaryDb(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME),

  _mainAggregation(DEFAULT_AGG_FUNCTION),
  _mainRankingModel(DEFAULT_RANKING_MODEL),
  _mainSettings(DEFAULT_MODEL_SETTINGS),
  _isReinitNeeded(true),
  _mode(mode),
  _imageIdStride(idOffset),
  _imagesPath(imagesPath),
  _rawNetRankingFilepath(rawNetRankingFilepath),
  _softmaxFilepath(softmaxFilepath),
  _deepFeaturesFilepath(deepFeaturesFilepath),
  _imageToIdMap(imageToIdMapFilepath),

  _keywords(keywordClassesFilepath),
  _images(ParseRawNetRankingBinFile())
{
  // Insert all desired aggregations
  _aggregations.emplace(AggregationId::cSoftmax, std::make_unique<AggregationSoftmax>());
  _aggregations.emplace(AggregationId::cXToTheP, std::make_unique<AggregationXToTheP>());

  // Insert all desired ranking models
  _models.emplace(RankingModelId::cViretBase, std::make_unique<ViretModel>());
  _models.emplace(RankingModelId::cBooleanBucket, std::make_unique<BooleanBucketModel>());


  

  // Connect to database
  auto result{ _primaryDb.EstablishConnection() };
  if (result != 0ULL)
  {
    LOG_ERROR("Connecting to primary DB failed.");
  }

}

bool ImageRanker::SpitImagesIntoCsv() const
{
  std::ofstream presoftmaxFileHandle("presoftmax_20k_subset.csv", std::ios::out);

  for (auto&&[imgId, pImg] : _images)
  {
    presoftmaxFileHandle << imgId << ",";

    size_t size{ pImg->m_rawNetRanking.size() };
    size_t i{ 0ULL };
    for (auto&& bin : pImg->m_rawNetRanking)
    {
      presoftmaxFileHandle << std::fixed<< std::setprecision(std::numeric_limits<long double>::digits10 + 1) << bin;
      

      if (i < size - 1) 
      {
        presoftmaxFileHandle << ",";
      }

      ++i;
    }
    presoftmaxFileHandle << std::endl;
  }

  presoftmaxFileHandle.close();

  std::ofstream softmaxFileHandle("softmax_20k_subset.csv", std::ios::out);

  
  for (auto&&[imgId, pImg] : _images)
  {
    softmaxFileHandle << imgId << ",";

    size_t size{ pImg->m_aggVectors[100].size() };
    size_t i{ 0ULL };

    for (auto&& bin : pImg->m_aggVectors[100])
    {
      
      softmaxFileHandle << std::fixed << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << bin;

      if (i < size - 1)
      {
        softmaxFileHandle << ",";
      }

      ++i;
    }
    softmaxFileHandle << std::endl;
  }

  softmaxFileHandle.close();


  return true;
}

bool ImageRanker::Initialize()
{
  if (_mode == Mode::cCollector)
  {
    return InitializeCollectorMode();
  }
  else
  {
    return InitializeFullMode();
  }

  

}

bool ImageRanker::Reinitialize()
{
  // If reinit not needed
  if (!_isReinitNeeded) 
  {
    return true;
  }

  // Clear app
  Clear();

  // Initialize with current settings
  return Initialize();
}

void ImageRanker::Clear()
{

}

void ImageRanker::SetMode(Mode value)
{ 
  _isReinitNeeded = true; 
  _mode = value; 
}

bool ImageRanker::InitializeCollectorMode()
{
  // Parse softmax file if available
  ParseSoftmaxBinFile();

  return true;
}

bool ImageRanker::InitializeFullMode()
{
  // Parse softmax file if available
  ParseSoftmaxBinFile();

  // Load and process all aggregations
  for (auto&& agg : _aggregations)
  {
    agg.second->CalculateTransformedVectors(_images);
  }

  // Initialize gridtests
  InitializeGridTests();


  //SpitImagesIntoCsv();

  return true;
}


void ImageRanker::InitializeGridTests()
{
  // ==========================================
  // Iterate through all desired configurations we want to test
  // ==========================================

  // Aggregations
  //for (auto&& agg : GridTest::m_aggregations)
  //{
  //  // Query origins
  //  for (auto&& queryOrigin : GridTest::m_queryOrigins)
  //  {
  //    // Ranking models
  //    for (auto&& model : GridTest::m_rankingModels)
  //    {


  //      switch (model)
  //      {
  //        // BooleanBucketModel
  //      case RankingModelId::cBooleanBucket:
  //        
  //        // True treshold probability values
  //        for (float fi{ BooleanBucketModel::m_trueTresholdFrom }; fi <= BooleanBucketModel::m_trueTresholdTo; fi += BooleanBucketModel::m_trueTresholdStep)
  //        {
  //          // In bucket ordering options
  //          for (auto&& qo : BooleanBucketModel::m_inBucketOrders)
  //          {
  //            std::vector<std::string> modSettings{ std::to_string(fi), std::to_string((uint8_t)qo) };

  //            GridTest::m_testSettings.emplace_back(agg, model, queryOrigin, modSettings);
  //          }
  //        }
  //        break;

  //        // BooleanViretModel
  //      case RankingModelId::cViretBase:
  //        // True treshold probability values
  //        for (float fi{ ViretModel::m_trueTresholdFrom }; fi <= ViretModel::m_trueTresholdTo; fi += ViretModel::m_trueTresholdStep)
  //        {
  //          // Query operation options
  //          for (auto&& qo : ViretModel::m_queryOperations) 
  //          {
  //            std::vector<std::string> modSettings{std::to_string(fi), std::to_string((uint8_t)qo)};

  //            GridTest::m_testSettings.emplace_back(agg, model, queryOrigin, modSettings);
  //          }
  //        }
  //        break;
  //      }


  //    }
  //  }
  //}
  LOG("GridTests initialized.");
}


AggregationFunctionBase* ImageRanker::GetAggregationById(AggregationId id) const
{
  // Try to get this aggregation
  if (
    auto result{ _aggregations.find(static_cast<AggregationId>(id)) };
    result != _aggregations.end()
    ) {
    return result->second.get();
  }
  else {
    LOG_ERROR("Aggregation not found!");
    return nullptr;
  }
}

RankingModelBase* ImageRanker::GetRankingModelById(RankingModelId id) const
{
  // Try to get this aggregation
  if (
    auto result{ _models.find(static_cast<RankingModelId>(id)) };
    result != _models.end()
    ) {
    return result->second.get();
  }
  else {
    LOG_ERROR("Ranking model not found!");
    return nullptr;
  }
}


void ImageRanker::SetMainSettings(AggregationId agg, RankingModelId rankingModel, ModelSettings settings)
{
  _mainAggregation = agg;
  _mainRankingModel = rankingModel;

  // \todo Avoid string -> type parsing
  _mainSettings = settings;
}


std::vector<std::pair<TestSettings, ChartData>> ImageRanker::RunGridTest(
  const std::vector<TestSettings>& userTestsSettings
)
{
  // Final result set
  std::vector<std::pair<TestSettings, ChartData>> results;

  std::vector<std::thread> tPool;
  tPool.reserve(10);

  constexpr unsigned int numThreads{8};
  size_t numTests{ GridTest::m_testSettings.size() };

  size_t numTestsPerThread{(numTests / numThreads) + 1};

  size_t from{ 0ULL };
  size_t to{ 0ULL };

  // Cache all queries to avoid data races
  GetCachedQueries(QueryOriginId::cDeveloper);
  GetCachedQueries(QueryOriginId::cPublic);

  std::thread tProgress(&GridTest::ReportTestProgress);

  // Start threads
  for (size_t i{ 0ULL }; i < numThreads; ++i)
  {
    to = from + numTestsPerThread;

    tPool.emplace_back(&ImageRanker::RunGridTestsFromTo, this, &results, from, to);

    from = to;
  }

  // Wait for threads
  for (auto&& t : tPool)
  {
    t.join();
  }
  tProgress.join();

  // xoxo

  auto cmp = [](const std::pair<size_t, size_t>& left, const std::pair<size_t, size_t>& right)
  {
    return left.second < right.second;
  };

  std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(cmp)> maxHeap(cmp);

  // Choose the winner
  size_t i{0ULL};
  for (auto&& test : results)
  {
    size_t score{0ULL};
    size_t j{ 0ULL };

    size_t numHalf{ test.second.size() / 2 };
    for (auto&& val : test.second)
    {
      // Only calculate first half
      if (j >= numHalf)
      {
        break;
      }
      score += val.second;

      ++j;
    }

    maxHeap.push(std::pair(i, score));

    ++i;
  }

  LOG("All tests complete.");

  std::vector<std::pair<TestSettings, ChartData>> resultsReal;

  for (size_t i{ 0ULL }; i < 15; ++i)
  {
    auto item = maxHeap.top();
    maxHeap.pop();

    resultsReal.emplace_back(results[item.first]);
  }
  
  LOG("Winner models and settings selected.");

  return resultsReal;
}


void ImageRanker::RunGridTestsFromTo(
  std::vector<std::pair<TestSettings, ChartData>>* pDest, 
  size_t fromIndex, size_t toIndex
)
{
  // If to index out of bounds
  if (toIndex > GridTest::m_testSettings.size()) 
  {
    toIndex = GridTest::m_testSettings.size();
  }

  auto to = GridTest::m_testSettings.begin() + toIndex;

  // Itarate over that interval
  size_t i{0ULL};
  for (auto it = GridTest::m_testSettings.begin() + fromIndex; it != to; ++it)
  {
    auto testSet{ (*it) };

    // Run test
    auto chartData{ RunModelTestWrapper(std::get<0>(testSet), std::get<1>(testSet), std::get<2>(testSet), std::get<3>(testSet), std::get<4>(testSet)) };

    pDest->emplace_back(testSet, std::move(chartData));

    GridTest::ProgressCallback();

    ++i;
  }
  
}

size_t ImageRanker::GetRandomImageId() const
{
  // Get random index
  return static_cast<size_t>(GetRandomInteger(0, (int)GetNumImages()) * _imageIdStride);
}

ImageReference ImageRanker::GetRandomImage() const
{
  size_t imageId{GetRandomImageId()};

  return ImageReference{ imageId, GetImageFilenameById(imageId) };
}

KeywordReferences ImageRanker::GetNearKeywords(const std::string& prefix)
{
  // Force lowercase
  std::locale loc;
  std::string lower;

  for (auto elem : prefix)
  {
    lower.push_back(std::tolower(elem, loc));
  }

  return _keywords.GetNearKeywords(lower);
}


KeywordData ImageRanker::GetKeywordByVectorIndex(size_t index)
{
  return _keywords.GetKeywordByVectorIndex(index);
}

std::vector<std::string> ImageRanker::GetImageFilenames() const
{
  // If no map file provided, use directory layout
  if (_imageToIdMap.empty()) 
  {
    return GetImageFilenamesFromDirectoryStructure();
  }

  // Open file with list of files in images dir
  std::ifstream inFile(_imageToIdMap, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR(std::string("Error opening file :") + _imageToIdMap);
  }

  std::vector<std::string> result;

  std::string line;

  // While there are lines in file
  while (std::getline(inFile, line))
  {
    // Extract file name
    std::stringstream ss(line);

    // FILE FORMAT: filename   imageId
    size_t imageId;
    std::string filename;

    ss >> filename;
    ss >> imageId;

    result.emplace_back(filename);
  }

  return result;
}

std::vector<std::string> ImageRanker::GetImageFilenamesFromDirectoryStructure() const
{
  LOG("Not implemented!"s);

  return std::vector<std::string>();
}

const std::vector<float>& ImageRanker::GetMainRankingVector(const Image& image) const
{
  switch (_mainAggregation) 
  {
  case AggregationId::cSoftmax:
    return image.m_softmaxVector;
    break;

  }
}

std::vector<float>& ImageRanker::GetMainRankingVector(Image& image)
{
  switch (_mainAggregation)
  {
  case AggregationId::cSoftmax:
    return image.m_softmaxVector;
    break;

  }
}


std::unordered_map<size_t, std::unique_ptr<Image>> ImageRanker::ParseRawNetRankingBinFile()
{
  // Get image filenames
  std::vector<std::string> imageFilenames{ GetImageFilenames() };

  // Open file for reading as binary from the end side
  std::ifstream ifs(_rawNetRankingFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file: "s + _rawNetRankingFilepath);
  }

  // Get end of file
  auto end = ifs.tellg();

  // Get iterator to begining
  ifs.seekg(0, std::ios::beg);

  // Compute size of file
  auto size = std::size_t(end - ifs.tellg());

  // If emtpy file
  if (size == 0)
  {
    LOG_ERROR("Empty file opened!");
  }


  // Create 4B buffer
  std::array<std::byte, sizeof(int32_t)>  smallBuffer;

  // Discard first 36B of data
  ifs.ignore(36ULL);

  // Read number of items in each vector per image
  ifs.read((char*)smallBuffer.data(), sizeof(int32_t));

  // If something happened
  if (!ifs)
  {
    LOG_ERROR("Error reading file: "s + _rawNetRankingFilepath);
  }

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(smallBuffer.data());

  // Calculate byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Where rows data start
  size_t currOffset = 40ULL;



  // Declare result vector
  std::unordered_map<size_t, std::unique_ptr<Image>> images;

  // Create line buffer
  std::vector<std::byte>  lineBuffer;
  lineBuffer.resize(byteRowLengths);

  // Iterate until there is something to read from file
  while (ifs.read((char*)lineBuffer.data(), byteRowLengths))
  {
    // Get picture ID of this row
    size_t id = ParseIntegerLE(lineBuffer.data());

    // Stride in bytes
    currOffset = sizeof(float);

    // Initialize vector of floats for this row
    std::vector<float> rawRankData;
    

    // Reserve exact capacitys
    rawRankData.reserve(numFloats);
    
    float sum{ 0.0f };
    float min{ std::numeric_limits<float>::max() };
    float max{ -std::numeric_limits<float>::max() };

    // Iterate through all floats in row
    for (size_t i = 0ULL; i < numFloats; ++i)
    {
      float rankValue{ ParseFloatLE(&lineBuffer[currOffset]) };

      // Update min
      if (rankValue < min) 
      {
        min = rankValue;
      }
      // Update max
      if (rankValue > max) 
      {
        max = rankValue;
      }

      // Add to sum
      sum += rankValue;

      // Push float value in
      rawRankData.emplace_back(rankValue);

      // Stride in bytes
      currOffset += sizeof(float);
    }

    // Calculate mean value
    float mean{ sum / numFloats };

    // Calculate variance
    float varSum{0.0f};
    for (auto&& val : rawRankData) 
    {
      float tmp{ val - mean };
      varSum += (tmp * tmp);
    }
    float variance = sqrtf((float)1 / (numFloats - 1) * varSum);

    // Get image filename 
    std::string filename{ imageFilenames[id / _imageIdStride] };


    // Push final row
    auto newIt = images.emplace(std::pair(
      id, 
      std::make_unique<Image>(
        id, 
        std::move(filename), std::move(rawRankData),
        min, max, 
        mean, variance
      )
    ));
  }

  return images;
}


bool ImageRanker::ParseSoftmaxBinFile()
{
  if (_softmaxFilepath.empty()) 
  {
    return false;
  }

  // Open file for reading as binary from the end side
  std::ifstream ifs(_softmaxFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file: "s + _softmaxFilepath);
    return false;
  }

  // Get end of file
  auto end = ifs.tellg();

  // Get iterator to begining
  ifs.seekg(0, std::ios::beg);

  // Compute size of file
  auto size = std::size_t(end - ifs.tellg());

  // If emtpy file
  if (size == 0)
  {
    LOG_ERROR("Empty file opened!");
    return false;
  }


  // Create 4B buffer
  std::array<std::byte, sizeof(int32_t)>  smallBuffer;

  // Discard first 36B of data
  ifs.ignore(36ULL);

  // Read number of items in each vector per image
  ifs.read((char*)smallBuffer.data(), sizeof(int32_t));

  // If something happened
  if (!ifs)
  {
    LOG_ERROR("Error reading file: "s + _softmaxFilepath);
    return false;
  }

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(smallBuffer.data());

  // Calculate byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Where rows data start
  size_t currOffset = 40ULL;



  // Create line buffer
  std::vector<std::byte>  lineBuffer;
  lineBuffer.resize(byteRowLengths);

  // Iterate until there is something to read from file
  while (ifs.read((char*)lineBuffer.data(), byteRowLengths))
  {
    // Get picture ID of this row
    size_t id = ParseIntegerLE(lineBuffer.data());

    // Get this image
    auto imageIt = _images.find(id);

    // Stride in bytes
    currOffset = sizeof(float);

    std::vector<float> softmaxVector; 
    softmaxVector.reserve(numFloats);

    // Iterate through all floats in row
    for (size_t i = 0ULL; i < numFloats; ++i)
    {
      float rankValue{ ParseFloatLE(&lineBuffer[currOffset]) };

      softmaxVector.emplace_back(rankValue);

      // Stride in bytes
      currOffset += sizeof(float);
    }
    // Store  vector of floats
    imageIt->second->m_aggVectors.insert(std::pair(static_cast<size_t>(AggregationId::cSoftmax), std::move(softmaxVector)));
  }

  return true;
}


//void ImageRanker::CalculateMinMaxClampAgg()
//{
//  // Itarate through all images
//  for (auto&& imgPair : _images)
//  {
//    Image& img{ imgPair.second };
//
//    img.m_minMaxLinearVector.reserve(img.m_rawNetRanking.size());
//
//    float amplitude{ img.m_max - img.m_min };
//
//    size_t i{ 0ULL };
//    for (auto&& prob : img.m_rawNetRanking)
//    {
//      float newValue{ (prob - img.m_min) / (float)amplitude };
//
//      // Calculate the MAGIC!!
//      img.m_minMaxLinearVector[i] = newValue;
//
//      ++i;
//    }
//  }
//}


std::string ImageRanker::EncodeAndQuery(const std::string& query) const
{
  auto wordIds{ TokenizeAndQuery(query) };

  std::string resultEncodedGraph{ "&"s };

  for (auto&& wordId : wordIds)
  {
    resultEncodedGraph += "-"s + wordId + "+"s;
  }

  return resultEncodedGraph;
}

std::vector<GameSessionQueryResult> ImageRanker::SubmitUserQueriesWithResults(std::vector<GameSessionInputQuery> inputQueries, QueryOriginId origin)
{
  /******************************
    Save query to database
  *******************************/
  // Input format:
  // <SessionID, ImageID, User query - "wID1&wID1& ... &wIDn">

  // Resolve query origin
  size_t originNumber{ static_cast<size_t>(origin) };

  // Store it into database
  std::string sqlQuery{ "INSERT INTO `queries` (query, image_id, type, sessionId) VALUES " };

  for (auto&& query : inputQueries)
  {
    // Get image ID
    size_t imageId = std::get<1>(query);
    std::string queryString = std::get<2>(query);



    sqlQuery += "('"s + EncodeAndQuery(queryString) + "', " + std::to_string(imageId) + ", "s + std::to_string(originNumber) + ", '"s + std::get<0>(query) + "'),"s;
  }

  sqlQuery.pop_back();
  sqlQuery += ";";

  auto result = _primaryDb.NoResultQuery(sqlQuery);
  if (result != 0)
  {
    LOG_ERROR("Inserting queries into DB failed");
  }

  /******************************
    Construct result for user
  *******************************/
  std::vector<GameSessionQueryResult> userResult;
  userResult.reserve(inputQueries.size());

  for (auto&& query : inputQueries)
  {
    // Get user keywords tokens
    std::vector<std::string> userKeywords{ StringenizeAndQuery(std::get<2>(query)) };

    // Get image ID
    size_t imageId = std::get<1>(query);

    // Get image filename
    std::string imageFilename{ GetImageFilenameById(imageId) };

    std::vector<std::pair<std::string, float>> netKeywordsProbs{};

    userResult.emplace_back(std::get<0>(query), std::move(imageFilename), std::move(userKeywords), GetHighestProbKeywords(imageId, 10ULL));
  }

  return userResult;
}

std::vector<std::pair<std::string, float>> ImageRanker::GetHighestProbKeywords(size_t imageId, size_t N) const
{
  // Find image in map
  std::unordered_map<size_t, std::unique_ptr<Image>>::const_iterator imagePair = _images.find(imageId);

  // If no such image
  if (imagePair == _images.end())
  {
    LOG_ERROR("No image found!");
    return std::vector<std::pair<std::string, float>>();
  }

  // Construct new subvector
  std::vector<std::pair<std::string, float>> result;
  result.reserve(N);

  auto ranking = imagePair->second->m_rawNetRankingSorted;

  // Get first N highest probabilites
  for (size_t i = 0ULL; i < N; ++i)
  {
    float probability{ ranking[i].second };

    // Get keyword string
    std::string keyword{ std::get<1>(_keywords.GetKeywordByVectorIndex(ranking[i].first)) }; ;

    // Place it into result vector
    result.emplace_back(std::pair(keyword, probability));
  }

  return result;
}

std::vector<std::string> ImageRanker::TokenizeAndQuery(std::string_view query) const
{
  // Create sstram from query
  std::stringstream querySs{ query.data() };

  std::vector<std::string> resultTokens;
  std::string tokenString;

  while (std::getline(querySs, tokenString, '&'))
  {
    // If empty string
    if (tokenString.empty())
    {
      continue;
    }

    // Push new token into result
    resultTokens.emplace_back(std::move(tokenString));
  }

  return resultTokens;
}

std::vector<std::string> ImageRanker::StringenizeAndQuery(const std::string& query) const
{
  // Create sstram from query
  std::stringstream querySs{ query.data() };

  std::vector<std::string> resultTokens;
  std::string tokenString;

  while (std::getline(querySs, tokenString, '&'))
  {
    // If empty string
    if (tokenString.empty())
    {
      continue;
    }

    std::stringstream tokenSs{ tokenString };
    size_t wordnetId;
    tokenSs >> wordnetId;

    // Push new token into result
    resultTokens.emplace_back(GetKeywordByWordnetId(wordnetId));
  }

  return resultTokens;
}


ChartData ImageRanker::RunModelTestWrapper(
  AggregationId aggId, RankingModelId modelId, QueryOriginId dataSource,
  const ModelSettings& modelSettings, const AggregationSettings& aggSettings
) const
{
  // Get queries
  auto testQueries = GetCachedQueries(dataSource);

  // Get desired aggregation
  auto pAggFn = GetAggregationById(aggId);
  // Setup model correctly
  pAggFn->SetAggregationSettings(aggSettings);

  // Get disired model
  auto pRankingModel = GetRankingModelById(modelId);
  // Setup model correctly
  pRankingModel->SetModelSettings(modelSettings);

  // Run test
  return pRankingModel->RunModelTest(pAggFn, testQueries, _images);
}


std::vector<UserImgQuery>& ImageRanker::GetCachedQueries(QueryOriginId dataSource) const
{
  static std::vector<UserImgQuery> cachedData0;
  static std::vector<UserImgQuery> cachedData1;
  static std::vector<UserImgQuery> cachedData2;

  static std::chrono::steady_clock::time_point cachedData0Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData1Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData2Ts = std::chrono::steady_clock::now();

  auto currentTime = std::chrono::steady_clock::now();

  switch (dataSource) 
  {
  case QueryOriginId::cDeveloper:

    if (cachedData0.empty() || cachedData0Ts < currentTime)
    {
      // Fetch pairs of <Q, Img>
      std::string query("SELECT image_id, query FROM `image-ranker-collector-data2`.queries WHERE type = " + std::to_string((int)dataSource) + ";");
      
      auto dbData = _primaryDb.ResultQuery(query);

      if (dbData.first != 0)
      {
        LOG_ERROR("Error getting queries from database."s);
      }

      // Parse DB results
      for (auto&& idQueryRow : dbData.second)
      {
        size_t imageId{ static_cast<size_t>(strToInt(idQueryRow[0].data())) };
        CnfFormula queryFormula{ _keywords.GetCanonicalQuery(EncodeAndQuery(idQueryRow[1])) };
        
        cachedData0.emplace_back(std::move(imageId), std::move(queryFormula));
      }

      cachedData0Ts = std::chrono::steady_clock::now();
      cachedData0Ts += std::chrono::hours(2);

    }

    return cachedData0;

    break;

  case QueryOriginId::cPublic:

    if (cachedData1.empty() || cachedData1Ts < currentTime)
    {
      // Fetch pairs of <Q, Img>
      std::string query("SELECT image_id, query FROM `image-ranker-collector-data2`.queries WHERE type = " + std::to_string((int)dataSource) + ";");
      auto dbData = _primaryDb.ResultQuery(query);

      if (dbData.first != 0)
      {
        LOG_ERROR("Error getting queries from database."s);
      }

      // Parse DB results
      for (auto&& idQueryRow : dbData.second)
      {
        size_t imageId{ static_cast<size_t>(strToInt(idQueryRow[0].data())) };
        CnfFormula queryFormula{ _keywords.GetCanonicalQuery(EncodeAndQuery(idQueryRow[1])) };

        cachedData1.emplace_back(std::move(imageId), std::move(queryFormula));
      }

      cachedData1Ts = std::chrono::steady_clock::now();
      cachedData1Ts += std::chrono::hours(2);
    }

    return cachedData1;

    break;
  }

  return cachedData0;
}


std::pair<std::vector<ImageReference>, QueryResult> ImageRanker::GetRelevantImagesWrapper(
  const std::string& queryEncodedPlaintext, size_t numResults,
  AggregationId aggId, RankingModelId modelId,
  const ModelSettings& modelSettings, const AggregationSettings& aggSettings,
  size_t imageId
) const
{
  // Decode query to logical CNF formula
  CnfFormula queryFormula{ _keywords.GetCanonicalQuery(EncodeAndQuery(queryEncodedPlaintext)) };

  // Get desired aggregation
  auto pAggFn = GetAggregationById(aggId);
  // Setup model correctly
  pAggFn->SetAggregationSettings(aggSettings);
  
  // Get disired model
  auto pRankingModel = GetRankingModelById(modelId);
  // Setup model correctly
  pRankingModel->SetModelSettings(modelSettings);

  // Rank it
  auto [imgOrder, targetImgRank] {pRankingModel->GetRankedImages(queryFormula, pAggFn, _images, numResults, imageId)};


  std::pair<std::vector<ImageReference>, QueryResult> resultResponse;

  // Prepare final result to return
  for (auto&& imgId : imgOrder)
  {
    resultResponse.first.emplace_back(ImageReference(imgId, GetImageFilenameById(imgId)));
  }

  // Fill in QueryResult
  resultResponse.second.m_targetImageRank = targetImgRank;

  return resultResponse;
}



const Image* ImageRanker::GetImageDataById(size_t imageId) const
{
  auto imgIdImgPair = _images.find(imageId);

  if (imgIdImgPair == _images.end())
  {
    LOG_ERROR("Image not found.")
      return nullptr;
  }

  // Return copy to this Image instance
  return imgIdImgPair->second.get();
}


std::unordered_map<size_t, std::pair<size_t, std::string> > ImageRanker::ParseKeywordClassesTextFile(std::string_view filepath) const
{
  // Open file with list of files in images dir
  std::ifstream inFile(filepath.data(), std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR("Error opening file :"s + filepath.data());
  }

  // Result variable
  std::unordered_map<size_t, std::pair<size_t, std::string> > keywordTable;


  std::string lineBuffer;

  // While there is something to read
  while (std::getline(inFile, lineBuffer))
  {
    if (lineBuffer.at(0) == 72)
    {
      continue;
    }

    // Extract file name
    std::stringstream lineBufferStream(lineBuffer);

    std::vector<std::string> tokens;
    std::string token;
    size_t i = 0ULL;

    while (std::getline(lineBufferStream, token, '~')) 
    {
      tokens.push_back(token);

      ++i;
    }


    // Index of vector
    std::stringstream vectIndSs(tokens[0]);
    std::stringstream wordnetIdSs(tokens[1]);

    size_t vectorIndex;
    size_t wordnetId;
    std::string indexClassname = tokens[2];

    vectIndSs >> vectorIndex;
    wordnetIdSs >> wordnetId;

    // Insert this record into table
    keywordTable.insert(std::make_pair(vectorIndex, std::make_pair(wordnetId, indexClassname)));
  }

  // Return result filepath 
  return keywordTable;
}


std::unordered_map<size_t, std::pair<size_t, std::string> > ImageRanker::ParseHypernymKeywordClassesTextFile(std::string_view filepath) const
{
  // Open file with list of files in images dir
  std::ifstream inFile(filepath.data(), std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR("Error opening file :"s + filepath.data());
  }

  // Result variable
  std::unordered_map<size_t, std::pair<size_t, std::string> > keywordTable;

  size_t idCounter = 0ULL;
  std::string lineBuffer;

  // While there is something to read
  while (std::getline(inFile, lineBuffer))
  {
    ++idCounter;

    // If not 'H' line, just continue
    if (lineBuffer.at(0) != 72)
    {
      continue;
    }

    // Extract file name
    std::stringstream lineBufferStream(lineBuffer);

    std::vector<std::string> tokens;
    std::string token;
    size_t i = 0ULL;

    while (std::getline(lineBufferStream, token, '~')) 
    {
      tokens.push_back(token);

      ++i;
    }


    // Index of vector
    //std::stringstream vectIndSs(tokens[0]);
    std::stringstream wordnetIdSs(tokens[1]);

    size_t vectorIndex;
    size_t wordnetId;
    std::string indexClassname = tokens[2];

    vectorIndex = idCounter;
    wordnetIdSs >> wordnetId;

    // Insert this record into table
    keywordTable.insert(std::make_pair(vectorIndex, std::make_pair(wordnetId, indexClassname)));


  }

  // Return result filepath 
  return keywordTable;
}

std::string ImageRanker::GetImageFilenameById(size_t imageId) const
{
  auto imgPair = _images.find(imageId);

  if (imgPair == _images.end())
  {
    LOG_ERROR("Image not found");
  }

  return imgPair->second->m_filename;
}


std::vector<std::byte> ImageRanker::LoadFileToBuffer(std::string_view filepath) const
{
  // Open file for reading as binary
  std::ifstream ifs(filepath.data(), std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file :"s + filepath.data());
  }

  // Get end of file
  auto end = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // Compute size of file
  auto size = std::size_t(end - ifs.tellg());

  // If emtpy file
  if (size == 0)
  {
    return std::vector<std::byte>();
  }

  // Declare vector with enough capacity
  std::vector<std::byte> buffer(size);

  // If error during reading
  if (!ifs.read((char*)buffer.data(), buffer.size()))
  {
    LOG_ERROR("Error opening file :"s + filepath.data());
  }

  // Return (move) final buffer
  return buffer;
}


bool ImageRanker::LoadKeywordsFromDatabase(Database::Type type)
{
  Database* pDb{nullptr};

  if (type == Database::cPrimary)
  {
    pDb = &_primaryDb;
  }
  else
  {
    LOG_ERROR("NOT IMPLEMENTED!");
    return false;
  }
  
  std::string query{"SELECT `keywords`.`wordnet_id`, `keywords`.`vector_index`, `words`.`word`, `keywords`.`description` FROM `keywords` INNER JOIN `keyword_word` ON `keywords`.`wordnet_id` = `keyword_word`.`keyword_id` INNER JOIN `words` ON `keyword_word`.`word_id` = `words`.`id`;"};

  auto result = pDb->ResultQuery(query);

  // Add hypernym and hyponym data to id
  for (auto&& row : result.second)
  {
    // Hypernyms
    std::string queryHypernyms{ "SELECT `hypernym_id` FROM `keywords_hypernyms` WHERE `keyword_id` = " + row[0] + ";" };
    auto resultHyper = pDb->ResultQuery(queryHypernyms);
    std::string hypernyms{ "" };
    for (auto&& hyper : resultHyper.second)
    {
      hypernyms += hyper.front();
      hypernyms += ";";
    }

    row.push_back(hypernyms);
    

    // Hyponyms
    std::string queryHyponyms{ "SELECT `hyponyms_id` FROM `keywords_hyponyms` WHERE `keyword_id` = " + row[0] + ";" };
    auto resultHypo = pDb->ResultQuery(queryHyponyms);
    std::string hyponyms{ "" };
    for (auto&& hypo : resultHypo.second)
    {
      hyponyms += hypo.front();
      hyponyms += ";";
    }

    row.push_back(hyponyms);
  }

  // Load Keywords into data structures
  _keywords = KeywordsContainer(std::move(result.second));


  return true;
}

bool ImageRanker::LoadImagesFromDatabase(Database::Type type)
{
  //Database* pDb{ nullptr };

  //if (type == Database::cPrimary)
  //{
  //  pDb = &_primaryDb;
  //}
  //else
  //{
  //  LOG_ERROR("NOT IMPLEMENTED!");
  //  return false;
  //}

  //// Fetch data from db
  //std::string query{ "SELECT * FROM `images`;" };
  //auto result = pDb->ResultQuery(query);

  //// Iterate through all images
  //for (auto&& row : result.second)
  //{
  //  std::stringstream imageIdSs{ row[0] };
  //  size_t imageId;
  //  imageIdSs >> imageId;
  //    
  //  std::string filename{ row[1] };

  //  std::vector<std::pair<size_t, float>> probabilityVector;

  //  std::string queryProVec{ "SELECT `probability` FROM `probability_vectors` WHERE `image_id` = " + std::to_string(imageId) + " ORDER BY `vector_index`;" };
  //  auto resultProbVec = pDb->ResultQuery(queryProVec);

  //  // Construct probability vector
  //  size_t i = 0ULL;
  //  for (auto&& prob : resultProbVec.second)
  //  {
  //    std::stringstream probSs{ prob.front() };

  //    float probability;
  //    probSs >> probability;

  //    size_t i = 0ULL;
  //    probabilityVector.push_back(std::pair(i, probability));
  //    
  //    ++i;
  //  }

  //  // Sort probabilities
  //  std::sort(
  //    probabilityVector.begin(), probabilityVector.end(), 
  //    [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b)
  //  {
  //    return a.second > b.second;
  //  }
  //  );

  //  _images.insert(std::make_pair(imageId, Image{ imageId, std::move(filename), std::move(probabilityVector) }));

  //}
  return false;
}


std::pair<uint8_t, uint8_t> ImageRanker::GetGridTestProgress() const
{ 
  return GridTest::GetGridTestProgress(); 
}

int32_t ImageRanker::ParseIntegerLE(const std::byte* pFirstByte) const
{
  // Initialize value
  int32_t signedInteger = 0;

  // Construct final BE integer
  signedInteger = 
    static_cast<uint32_t>(pFirstByte[3]) << 24 | 
    static_cast<uint32_t>(pFirstByte[2]) << 16 | 
    static_cast<uint32_t>(pFirstByte[1]) << 8 | 
    static_cast<uint32_t>(pFirstByte[0]);

  // Return parsed integer
  return signedInteger;
}


float ImageRanker::ParseFloatLE(const std::byte* pFirstByte) const
{
  // Initialize temp value
  uint32_t byteFloat = 0;

  // Get correct unsigned value of float data
  byteFloat = 
    static_cast<uint32_t>(pFirstByte[3]) << 24 | 
    static_cast<uint32_t>(pFirstByte[2]) << 16 | 
    static_cast<uint32_t>(pFirstByte[1]) << 8 | 
    static_cast<uint32_t>(pFirstByte[0]);

  // Return reinterpreted data
  return *(reinterpret_cast<float*>(&byteFloat));
}


#if PUSH_DATA_TO_DB

bool ImageRanker::PushImagesToDatabase()
{
  /*===========================
  Push into `images` & `probability_vectors` table
  ===========================*/
  {
    // Start query
    std::string queryImages{ "INSERT IGNORE INTO images (`id`, `filename`) VALUES" };
    

    // Keywords then
    for (auto&& idImagePair : _images)
    {
      std::string filename{ _primaryDb.EscapeString(idImagePair.second._filename) };

      queryImages.append("(");
      queryImages.append(std::to_string(idImagePair.second._imageId));
      queryImages.append(", '");
      queryImages.append(filename);
      queryImages.append("'),");
    }

    // Delete last comma
    queryImages.pop_back();
    // Add semicolon
    queryImages.append(";");

    // Send query
    _primaryDb.NoResultQuery(queryImages);


    for (auto&& idImagePair : _images)
    {
      std::string filename{ _primaryDb.EscapeString(idImagePair.second._filename) };

      std::string queryProbs{ "INSERT IGNORE INTO probability_vectors (`image_id`, `vector_index`, `probability`) VALUES" };

      // Iterate through probability vector
      size_t i = 0ULL;
      for (auto&& prob : idImagePair.second._probabilityVector)
      {
        queryProbs.append("(");
        queryProbs.append(std::to_string(idImagePair.second._imageId));
        queryProbs.append(", ");
        queryProbs.append(std::to_string(prob.first));
        queryProbs.append(", ");
        queryProbs.append(std::to_string(prob.second));
        queryProbs.append("),");

        ++i;
      }
      // Delete last comma
      queryProbs.pop_back();
      // Add semicolon
      queryProbs.append(";");
      _primaryDb.NoResultQuery(queryProbs);
    }
  }

  return true;
}


bool ImageRanker::PushDataToDatabase()
{
  bool result{true};

  result = PushKeywordsToDatabase();
  result = PushImagesToDatabase();

  return result;
}

bool ImageRanker::PushKeywordsToDatabase()
{
  bool result = _keywords.PushKeywordsToDatabase(_primaryDb);

  return false;
}

#endif