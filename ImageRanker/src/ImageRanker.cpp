
#include "ImageRanker.h"

ImageRanker::ImageRanker(
  const std::string& imagesPath,
  const std::vector<KeywordsFileRef>& keywordsFileRefs,
  const std::vector<ScoringDataFileRef>& imageScoringFileRefs,
  const std::vector<ScoringDataFileRef>& imageSoftmaxScoringFileRefs,
  const std::vector<ScoringDataFileRef>& deepFeaturesFileRefs,
  const std::string& imageToIdMapFilepath,
  size_t idStride,
  eMode mode
) :
  _primaryDb(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME),
  _secondaryDb(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME),

  _mode(mode),
  _imageIdStride(idStride),
  _imagesPath(imagesPath),
  _imageToIdMapFilepath(imageToIdMapFilepath),
  _fileParser(this)
{
  // Construct all desired keyword containers
  for (auto&&[id, filepath] : keywordsFileRefs)
  {
    auto result = _keywordContainers.insert(
      std::pair(id, KeywordsContainer(this, eKeywordsDataType(id), filepath))
    );

    // Save shortcuts
    switch (id)
    {
    case eKeywordsDataType::cViret1:
      _pViretKws = &(result.first->second);
      break;

    case eKeywordsDataType::cGoogleAI:
      _pGoogleKws = &(result.first->second);
      break;
    }
  }

  //
  // Store initial scoring filepaths
  //
  for (auto&&[kwId, netId, filepath] : imageScoringFileRefs)
  {
    _imageScoringFileRefs.emplace(std::pair(kwId, netId), filepath);
  }

  //
  // Store initial Softmax scoring filepaths
  //
  for (auto&&[kwId, netId, filepath] : imageSoftmaxScoringFileRefs)
  {
    _imageSoftmaxScoringFileRefs.emplace(std::pair(kwId, netId), filepath);
  }

  //
  // Store initial deep features filepaths
  //
  for (auto&&[kwId, netId, filepath] : deepFeaturesFileRefs)
  {
    _deepFeaturesFileRefs.emplace(std::pair(kwId, netId), filepath);
  }


  // Connect to the database
  auto result{ _primaryDb.EstablishConnection() };
  if (result != 0ULL)
  {
    LOG_ERROR("Connecting to primary DB failed.");
  }

}

bool ImageRanker::Initialize()
{
  bool res{ true };

  // Initialize keyword containers
  for (auto&& [id, kwCont] : _keywordContainers)
  {
    res &= kwCont.Initialize();
  }

  // Collector only mode
  if (_mode == eMode::cCollector)
  {
    res &= InitializeCollectorMode();
  }
  // Search tool mode
  else if (_mode == eMode::cSearchTool)
  {
    res &= InitializeSearchToolMode();
  }
  // Full analytical mode
  else
  {
    res &= InitializeFullMode();
  }

  return res;
}

void ImageRanker::ClearData()
{
  _imageScoringFileRefs.clear();
  _imageSoftmaxScoringFileRefs.clear();
  _deepFeaturesFileRefs.clear();
  _keywordContainers.clear();
  _pViretKws = nullptr;
  _pGoogleKws = nullptr;

  _indexKwFrequency.clear();
  
  _transformations.clear();
  _models.clear();
  
  _images.clear();
}

bool ImageRanker::Reinitialize()
{ 
  // Clear all current working data
  ClearData();

  // Initialize with current settings
  return Initialize();
}

ImageRanker::eMode ImageRanker::GetMode() const
{
  return _mode;
}

void ImageRanker::SetMode(eMode value)
{ 
  _mode = value; 
}

const FileParser* ImageRanker::GetFileParser() const
{
  return &_fileParser;
}

bool ImageRanker::InitializeCollectorMode()
{
  // \todo Implement this if needed
  LOG_ERROR("Not implemented")

  return false;
}

bool ImageRanker::InitializeSearchToolMode()
{
  LOG_ERROR("Not implemented!"s);
  return false;
  /*
  // Parse binary images data with low memory version of parsing
  //_images = std::move(LowMem_ParseRawNetRankingBinFile());


#if TRECVID_MAPPING

  // Parse TRECVID shot reference
  _trecvidShotReferenceMap = ParseTrecvidShotReferencesFromDirectory(SHOT_REFERENCE_PATH);

  // Parse file containing dropped shots
  _tvDroppedShots = ParseTrecvidDroppedShotsFile(DROPPED_SHOTS_FILEPATH);

#endif

  // In Search tool mode only one transformation and one model should be used because of memory savings
  {
    // Insert all desired transformations
    //_aggregations.emplace(NetDataTransformation::cSoftmax, std::make_unique<TransformationSoftmax>());
    _transformations.emplace(InputDataTransformId::cXToTheP, std::make_unique<TransformationLinearXToTheP>());

    // Insert all desired ranking models
    _models.emplace(RankingModelId::cViretBase, std::make_unique<ViretModel>());
    _models.emplace(RankingModelId::cBooleanBucket, std::make_unique<BooleanBucketModel>());
  }

  // Load and process all transformations
  for (auto&& transf : _transformations)
  {
    // Send in 1 for MAX based precalculations
    transf.second->LowMem_CalculateTransformedVectors(_images, PRECOMPUTE_MAX_BASED_DATA);
  }


  // Apply hypernym recalculation on all transformed vectors
  for (auto&&[imageId, pImg] : _images)
  {
    for (auto&&[transformId, binVec] : pImg->_transformedImageScoringData)
    {
      // If is summ based aggregation precalculation
      if (((transformId / 10) % 10) == 0)
      {
        LowMem_RecalculateHypernymsInVectorUsingSum(binVec);
      }
      else
      {
        LowMem_RecalculateHypernymsInVectorUsingMax(binVec);
      }

    }
  }


  LOG("ImageRanker initialized in mode 'SearchTool'!");
  return true;
  */
}

bool ImageRanker::InitializeFullMode()
{
  //
  // Setup supported transformations and models
  //
  {
    // Insert all desired transformations
    _transformations.emplace(InputDataTransformId::cSoftmax, std::make_unique<TransformationSoftmax>());
    _transformations.emplace(InputDataTransformId::cXToTheP, std::make_unique<TransformationLinearXToTheP>());

    // Insert all desired ranking models
    _models.emplace(RankingModelId::cViretBase, std::make_unique<ViretModel>());
    _models.emplace(RankingModelId::cBooleanBucket, std::make_unique<BooleanBucketModel>());
  }

  // Initialize all images
  _images = _fileParser.ParseImagesMetaData(_imageToIdMapFilepath, _imageIdStride);

  // Fill in scoring data to images
  for (auto&&[kwScDataId, filepath] : _imageScoringFileRefs)
  {
    // Choose correct parsing method
    switch (std::get<0>(kwScDataId))
    {
    case eKeywordsDataType::cViret1:
      _fileParser.ParseRawScoringData_ViretFormat(_images, kwScDataId, filepath);
      break;

    case eKeywordsDataType::cGoogleAI:
      _fileParser.ParseRawScoringData_GoogleAiVisionFormat(_images, kwScDataId, filepath);
      break;

    default:
      LOG_ERROR("Invalid keyword data type.");
    }
  }

  // Fill in Softmax data
  for (auto&&[kwScDataId, filepath] : _imageSoftmaxScoringFileRefs)
  {
    // Choose correct parsing method
    switch (std::get<0>(kwScDataId))
    {
    case eKeywordsDataType::cViret1:
      _fileParser.ParseSoftmaxBinFile_ViretFormat(_images, kwScDataId, filepath);
      break;

    case eKeywordsDataType::cGoogleAI:
      _fileParser.ParseSoftmaxBinFile_GoogleAiVisionFormat(_images, kwScDataId, filepath);
      break;

    default:
      LOG_ERROR("Invalid scoring data type.");
    }
  }

  // Load and process all supported transformations
  for (auto&& transFn : _transformations)
  {
    transFn.second->CalculateTransformedVectors(_images);
  }
  
  // Apply hypernym recalculation on all transformed vectors
  for (auto&& pImg : _images)
  {
    for (auto&&[ksScDataId, transformedData] : pImg->_transformedImageScoringData)
    {
      // \todo implement propperly

      // Copy untouched vector for simulating user from [0, 1] linear scale transformation
      pImg->_rawSimUserData.emplace(ksScDataId, pImg->GetScoringVectorsConstPtr(ksScDataId)->at(200));

      if (std::get<0>(ksScDataId) == eKeywordsDataType::cGoogleAI)
      {
        continue;
      }

      for (auto&&[transformId, binVec] : transformedData)
      {
        // If is summ based aggregation precalculation
        if (((transformId / 10) % 10) == 0)
        {
          RecalculateHypernymsInVectorUsingSum(binVec);
        }
        else
        {
          RecalculateHypernymsInVectorUsingMax(binVec);
        }

      }
    }
  }

  // Calculate approx document frequency
  //ComputeApproxDocFrequency(200, TRUE_TRESHOLD_FOR_KW_FREQUENCY);

  // Initialize gridtests
  InitializeGridTests();

  LOG("Aplication initialized in FULL MODE.");
  return true;
}

TransformationFunctionBase* ImageRanker::GetAggregationById(InputDataTransformId id) const
{
  // Try to get this aggregation
  if (
    auto result{ _transformations.find(static_cast<InputDataTransformId>(id)) };
    result != _transformations.end()
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

std::vector<std::pair<TestSettings, ChartData>> ImageRanker::RunGridTest(
  const std::vector<TestSettings>& userTestsSettings
)
{
  LOG_ERROR("ImageRanker::RunGridTest() NOT IMPLEMENTED");
  return std::vector<std::pair<TestSettings, ChartData>>();
  //// Final result set
  //std::vector<std::pair<TestSettings, ChartData>> results;

  //std::vector<std::thread> tPool;
  //tPool.reserve(10);

  //constexpr unsigned int numThreads{8};
  //size_t numTests{ GridTest::m_testSettings.size() };

  //size_t numTestsPerThread{(numTests / numThreads) + 1};

  //size_t from{ 0ULL };
  //size_t to{ 0ULL };

  //// Cache all queries to avoid data races
  //GetCachedQueries(QueryOriginId::cDeveloper);
  //GetCachedQueries(QueryOriginId::cPublic);

  //std::thread tProgress(&GridTest::ReportTestProgress);

  //// Start threads
  //for (size_t i{ 0ULL }; i < numThreads; ++i)
  //{
  //  to = from + numTestsPerThread;

  //  tPool.emplace_back(&ImageRanker::RunGridTestsFromTo, this, &results, from, to);

  //  from = to;
  //}

  //// Wait for threads
  //for (auto&& t : tPool)
  //{
  //  t.join();
  //}
  //tProgress.join();

  //// xoxo

  //auto cmp = [](const std::pair<size_t, size_t>& left, const std::pair<size_t, size_t>& right)
  //{
  //  return left.second < right.second;
  //};

  //std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(cmp)> maxHeap(cmp);

  //// Choose the winner
  //size_t i{0ULL};
  //for (auto&& test : results)
  //{
  //  size_t score{0ULL};
  //  size_t j{ 0ULL };

  //  size_t numHalf{ test.second.size() / 2 };
  //  for (auto&& val : test.second)
  //  {
  //    // Only calculate first half
  //    if (j >= numHalf)
  //    {
  //      break;
  //    }
  //    score += val.second;

  //    ++j;
  //  }

  //  maxHeap.push(std::pair(i, score));

  //  ++i;
  //}

  //LOG("All tests complete.");

  //std::vector<std::pair<TestSettings, ChartData>> resultsReal;

  //for (size_t i{ 0ULL }; i < 15; ++i)
  //{
  //  auto item = maxHeap.top();
  //  maxHeap.pop();

  //  resultsReal.emplace_back(results[item.first]);
  //}
  //
  //LOG("Winner models and settings selected.");

  //return resultsReal;
}

void ImageRanker::RunGridTestsFromTo(
  std::vector<std::pair<TestSettings, ChartData>>* pDest, 
  size_t fromIndex, size_t toIndex
)
{
  LOG_ERROR("Not implemented : RunGridTestsFromTo()!");
  return;

  //// If to index out of bounds
  //if (toIndex > GridTest::m_testSettings.size()) 
  //{
  //  toIndex = GridTest::m_testSettings.size();
  //}

  //auto to = GridTest::m_testSettings.begin() + toIndex;

  //// Itarate over that interval
  //size_t i{0ULL};
  //for (auto it = GridTest::m_testSettings.begin() + fromIndex; it != to; ++it)
  //{
  //  auto testSet{ (*it) };

  //  // Run test
  //  auto chartData{ RunModelTestWrapper(std::get<0>(testSet), std::get<1>(testSet), std::get<2>(testSet), std::vector<std::string>({ "1"s }), std::get<3>(testSet), std::get<4>(testSet)) };

  //  pDest->emplace_back(testSet, std::move(chartData));

  //  GridTest::ProgressCallback();

  //  ++i;
  //}
  
}

size_t ImageRanker::GetRandomImageId() const
{
  // Get random index
  return static_cast<size_t>(GetRandomInteger(0, (int)GetNumImages()) * _imageIdStride);
}

const Image* ImageRanker::GetRandomImage() const
{
  size_t imageId{ GetRandomImageId() };

  return GetImageDataById(imageId);
}

std::tuple<const Image*, bool, size_t> ImageRanker::GetCouplingImage() const
{
  // Get Viret KW data
  auto queriesViret{ GetCachedQueries(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet), QueryOriginId::cAll) };

  auto queriesGoogle{ GetCachedQueries(std::tuple(eKeywordsDataType::cGoogleAI, eImageScoringDataType::cGoogleAI), QueryOriginId::cAll) };

  // image ID -> (Number of annotations left for this ID, Number of them without examples)
  std::map<size_t, std::pair<size_t, size_t>> imageIdOccuranceMap;
  
  size_t totalCounter{ 0_z };

  // Add pluses there first
  for (auto&& v : queriesViret)
  {
    auto&&[imageId, fml, withExamples] = v[0];

    auto countRes{ imageIdOccuranceMap.count(imageId) };

    // If not existent Image ID add it
    if (countRes <= 0)
    {
      imageIdOccuranceMap.emplace(imageId, std::pair{ 0_z, 0_z });
    }

    // Increment count
    ++imageIdOccuranceMap[imageId].first;
    ++totalCounter;

    // If should be without examples
    if (!withExamples)
    {
      ++imageIdOccuranceMap[imageId].second;
    }
  }


  // Subtract already created Google mathing ones
  for (auto&& v : queriesGoogle)
  {
    auto&&[imageId, fml, withExamples] = v[0];

    auto countRes{ imageIdOccuranceMap.count(imageId) };

    // Skip Google ones that are not created for viret
    if (countRes <= 0)
    {
      continue;
    }

    // Increment count
    auto& i{ imageIdOccuranceMap[imageId].first };
    auto& ii{ imageIdOccuranceMap[imageId].second };

    // If without examples
    if (!withExamples)
    {
      if (i > 0)
      {
        if (ii > 0)
        {
          --i;
          --ii;
          --totalCounter;
        }
        else
        {
          continue;
        }
      }
    }
    else 
    {
      if (i > 0)
      {
        --i;
        --totalCounter;
      }
    }

    // If this record paired
    if (totalCounter <= 0)
    {
      imageIdOccuranceMap.erase(imageId);
    }
  }

  // Get random item from map
  auto it = imageIdOccuranceMap.begin();

  if (totalCounter <= 0)
  {
    return std::tuple(GetRandomImage(), true, 0_z);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0_z, totalCounter - 1_z);

  std::advance(it, dis(gen));

  auto pImg{GetImageDataById(it->first)};

  return std::tuple(pImg, !((bool)it->second.second), totalCounter);
}

std::vector<const Image*> ImageRanker::GetRandomImageSequence(size_t seqLength) const
{
  std::vector<const Image*> resultImagePtrs;

  for (size_t i{ 0_z }; i < seqLength; ++i)
  {
    // \todo Implement to return images from the same video
    resultImagePtrs.emplace_back(GetRandomImage());
  }

  return resultImagePtrs;
}

NearKeywordsResponse ImageRanker::GetNearKeywords(
  KwScoringDataId kwScDataId, const std::string& prefix, size_t numResults, bool withExampleImages
)
{
  // Force lowercase
  std::locale loc;
  std::string lower;

  // Convert to lowercase
  for (auto elem : prefix)
  {
    lower.push_back(std::tolower(elem, loc));
  }

  KeywordsContainer* pkws{nullptr};

  // Save shortcuts
  switch (std::get<0>(kwScDataId))
  {
  case eKeywordsDataType::cViret1:
    pkws = _pViretKws;
    break;

  case eKeywordsDataType::cGoogleAI:
    pkws = _pGoogleKws;
    break;
  }

  auto suggestedKeywordsPtrs{ pkws->GetNearKeywordsPtrs(lower, numResults) };

  if (withExampleImages)
  {
    // Load representative images for keywords
    for (auto&& pKw : suggestedKeywordsPtrs)
    {
      LoadRepresentativeImages(kwScDataId, pKw);
    }
  }

  return suggestedKeywordsPtrs;
}

bool ImageRanker::LoadRepresentativeImages(KwScoringDataId kwScDataId, Keyword* pKw)
{
  // If examples already loaded
  if (!pKw->m_exampleImageFilenames.empty())
  {
    return true;
  }

  std::vector<std::string> queries;
  queries.push_back(std::to_string(pKw->m_wordnetId));

  // Get first results for this keyword
  std::vector<const Image*> relevantImages{ std::get<0>(GetRelevantImages(
    kwScDataId,
    queries, 10,
    DEFAULT_AGG_FUNCTION, DEFAULT_RANKING_MODEL,
    DEFAULT_MODEL_SETTINGS, DEFAULT_TRANSFORM_SETTINGS
  )) };


  // Push those into examples
  for (auto pImg : relevantImages)
  {
    pKw->m_exampleImageFilenames.emplace_back(pImg->m_filename);
  }

  return true;
}

Keyword* ImageRanker::GetKeywordByVectorIndex(KwScoringDataId kwScDataId, size_t index)
{
  return GetCorrectKwContainerPtr(kwScDataId)->GetKeywordPtrByVectorIndex(index);
}

std::vector<std::string> ImageRanker::GetImageFilenamesTrecvid() const
{
  // If no map file provided, use directory layout
  if (_imageToIdMapFilepath.empty())
  {
    LOG_ERROR("Image filename file not provided.");
    return std::vector<std::string>();
  }

  // Open file with list of files in images dir
  std::ifstream inFile(_imageToIdMapFilepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR(std::string("Error opening file :") + _imageToIdMapFilepath);
  }

  std::vector<std::string> result;

  std::string line;

  size_t i{ 0_z };

  // While there are lines in file
  while (std::getline(inFile, line))
  {
    //auto [videoId, shotId, frameNumber] { ParseVideoFilename(line) };

    line = line.substr(6, line.length() - 6);

    result.emplace_back(line);

    ++i;
  }

  return result;
}

std::tuple<
  std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
  std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>,
  std::vector<std::pair<size_t, std::string>>
> 
ImageRanker::GetImageKeywordsForInteractiveSearch(
  size_t imageId, size_t numResults, KwScoringDataId kwScDataId, bool withExampleImages)
{
  std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>  hypernyms;
  std::vector<std::tuple<size_t, std::string, float, std::vector<std::string>>>  nonHypernyms;
  nonHypernyms.reserve(numResults);

  auto img{ GetImageDataById(imageId) };
  if (img != nullptr)
  {
    LOG_ERROR("Image not found.");
  }

  // Get kws
  size_t i{ 0ULL };
  for (auto&& [kwPtr, kwScore] : img->_topKeywords[kwScDataId])
  {
    if (!(i < numResults))
    {
      break;
    }

    std::string word{ kwPtr->m_word };
    std::vector<std::string> exampleImagesFilepaths;

    // Get example images
    if (withExampleImages)
    {
      LoadRepresentativeImages(kwScDataId, kwPtr);

      exampleImagesFilepaths = kwPtr->m_exampleImageFilenames;
    }

    nonHypernyms.emplace_back(std::tuple(kwPtr->m_wordnetId, std::move(word), kwScore, std::move(exampleImagesFilepaths)));

    ++i;
  }

  // Get video/shot images
  std::vector<std::pair<size_t, std::string>> succs;

  size_t numSucc{ img->m_numSuccessorFrames };
  for (size_t i{ 0_z }; i <= numSucc; ++i)
  {
    size_t nextId{img->m_imageId + (i * _imageIdStride) };

    auto pImg{ GetImageDataById(nextId) };

    succs.emplace_back(nextId, pImg->m_filename);
  }

  return std::tuple(std::move(hypernyms), std::move(nonHypernyms), std::move(succs));
}

void ImageRanker::RecalculateHypernymsInVectorUsingSum(std::vector<float>& binVectorRef)
{
  std::vector<float> newBinVector;
  newBinVector.reserve(binVectorRef.size());

  // Iterate over all bins in this vector
  for (auto&& [it, i] { std::tuple(binVectorRef.begin(), size_t{ 0 }) }; it != binVectorRef.end(); ++it, ++i)
  {
    auto&& bin{*it};

    auto pKw{ _pViretKws->GetKeywordConstPtrByVectorIndex(i) };

    float binValue{0.0f};

    // Iterate over all indices this keyword interjoins
    for (auto&& kwIndex : pKw->m_hyponymBinIndices)
    {
      binValue += binVectorRef[kwIndex];
    }

    newBinVector.emplace_back(binValue);
  }

#if LOG_DEBUG_HYPERNYMS_EXPANSION

  /*for (size_t i{ 0ULL }; i < newBinVector.size(); ++i)
  {
    LOG(std::to_string(binVectorRef[i]) +" => "s + std::to_string(newBinVector[i]));
  }*/

#endif

  // Replace old values with new
  binVectorRef = std::move(newBinVector);
}

void ImageRanker::RecalculateHypernymsInVectorUsingMax(std::vector<float>& binVectorRef)
{
  std::vector<float> newBinVector;
  newBinVector.reserve(binVectorRef.size());

  // Iterate over all bins in this vector
  for (auto&& [it, i] { std::tuple(binVectorRef.begin(), size_t{ 0 }) }; it != binVectorRef.end(); ++it, ++i)
  {
    auto&& bin{ *it };

    auto pKw{ _pViretKws->GetKeywordConstPtrByVectorIndex(i) };

    float binValue{ 0.0f };

    // Iterate over all indices this keyword interjoins
    for (auto&& kwIndex : pKw->m_hyponymBinIndices)
    {
      binValue = std::max(binValue, binVectorRef[kwIndex]);
    }

    newBinVector.emplace_back(binValue);
  }

#if LOG_DEBUG_HYPERNYMS_EXPANSION

  /*for (size_t i{ 0ULL }; i < newBinVector.size(); ++i)
  {
    LOG(std::to_string(binVectorRef[i]) +" => "s + std::to_string(newBinVector[i]));
  }*/

#endif

  // Replace old values with new
  binVectorRef = std::move(newBinVector);
}

void ImageRanker::LowMem_RecalculateHypernymsInVectorUsingSum(std::vector<float>& binVectorRef)
{
  // Iterate over all bins in this vector
  for (auto&& [it, i] { std::tuple(binVectorRef.begin(), size_t{ 0 }) }; it != binVectorRef.end(); ++it, ++i)
  {
    auto& bin{ *it };
    auto pKw{ _pViretKws->GetKeywordConstPtrByVectorIndex(i) };

    float binValue{ 0.0f };

    // Iterate over all indices this keyword interjoins
    for (auto&& kwIndex : pKw->m_hyponymBinIndices)
    {
      binValue += binVectorRef[kwIndex];
    }

    // Write in new value
    bin = binValue;
  }
}

void ImageRanker::LowMem_RecalculateHypernymsInVectorUsingMax(std::vector<float>& binVectorRef)
{
  // Iterate over all bins in this vector
  for (auto&& [it, i] { std::tuple(binVectorRef.begin(), size_t{ 0 }) }; it != binVectorRef.end(); ++it, ++i)
  {
    auto&& bin{ *it };

    auto pKw{ _pViretKws->GetKeywordConstPtrByVectorIndex(i) };

    float binValue{ 0.0f };

    // Iterate over all indices this keyword interjoins
    for (auto&& kwIndex : pKw->m_hyponymBinIndices)
    {
      binValue = std::max(binValue, binVectorRef[kwIndex]);
    }

    bin = binValue;
  }
}

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

std::vector<GameSessionQueryResult> ImageRanker::SubmitUserQueriesWithResults(KwScoringDataId kwScDataId, std::vector<GameSessionInputQuery> inputQueries, QueryOriginId origin)
{
  /******************************
    Save query to database
  *******************************/
  // Input format:
  // <SessionID, ImageID, User query - "wID1&wID1& ... &wIDn">

  // Resolve query origin
  size_t originNumber{ static_cast<size_t>(origin) };

  // Store it into database
  std::string sqlQuery{ "INSERT INTO `queries` (query, keyword_data_type, scoring_data_type, image_id, type, sessionId) VALUES " };

  for (auto&& query : inputQueries)
  {
    // Get image ID
    size_t imageId = std::get<1>(query);
    std::string queryString = std::get<2>(query);

    sqlQuery += "('"s + EncodeAndQuery(queryString) + "', "s + std::to_string((size_t)std::get<0>(kwScDataId)) + ", "s 
      + std::to_string((size_t)std::get<1>(kwScDataId)) + ", " + std::to_string(imageId) + ", "s 
      + std::to_string(originNumber) + ", '"s + std::get<0>(query) + "'),"s;
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
    std::vector<std::string> userKeywords{ StringenizeAndQuery(kwScDataId, std::get<2>(query)) };

    // Get image ID
    size_t imageId = std::get<1>(query);

    // Get image filename
    std::string imageFilename{ GetImageFilenameById(imageId) };

    std::vector<std::pair<std::string, float>> netKeywordsProbs{};

    userResult.emplace_back(
      std::get<0>(query), std::move(imageFilename), std::move(userKeywords), 
      GetHighestProbKeywords(std::tuple(DEFAULT_KEYWORD_DATA_TYPE, DEFAULT_SCORING_DATA_TYPE), imageId, 10ULL)
    );
  }

  return userResult;
}

size_t ImageRanker::MapIdToVectorIndex(size_t id) const
{
  return id / _imageIdStride;
}

std::vector<std::pair<std::string, float>> ImageRanker::GetHighestProbKeywords(KwScoringDataId kwScDataId, size_t imageId, size_t N) const
{
  N = std::min(N, NUM_TOP_KEYWORDS);

  // Find image in map
  Image* pImg = _images[MapIdToVectorIndex(imageId)].get();

  
  // Construct new subvector
  std::vector<std::pair<std::string, float>> result;
  result.reserve(N);

  auto kwScorePairs = pImg->_topKeywords.at(kwScDataId);

  // Get first N highest probabilites
  for (size_t i = 0ULL; i < N; ++i)
  {
    float kwScore{ std::get<1>(kwScorePairs[i]) };

    // Get keyword string
    std::string keyword{ std::get<0>(kwScorePairs[i])->m_word }; ;

    // Place it into result vector
    result.emplace_back(std::pair(keyword, kwScore));
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

std::vector<std::string> ImageRanker::StringenizeAndQuery(KwScoringDataId kwScDataId, const std::string& query) const
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
    resultTokens.emplace_back(GetKeywordByWordnetId(kwScDataId, wordnetId));
  }

  return resultTokens;
}

ChartData ImageRanker::RunModelTestWrapper(
  KwScoringDataId kwScDataId,
  InputDataTransformId aggId, RankingModelId modelId, QueryOriginId dataSource,
  const SimulatedUserSettings& simulatedUserSettings, const RankingModelSettings& aggModelSettings, 
  const InputDataTransformSettings& netDataTransformSettings
) const
{
  std::vector<std::vector<UserImgQuery>> testQueries;

  // If data source should be simulated
  if (static_cast<int>(dataSource) >= SIMULATED_QUERIES_ENUM_OFSET)
  {
    LOG_ERROR("RunModelTestWrapper() - NOT IMPLEMENTED");

    // Parse simulated user settings
    auto simUser{ GetSimUserSettings(simulatedUserSettings) };

    if (static_cast<int>(dataSource) >= USER_PLUS_SIMULATED_QUERIES_ENUM_OFSET)
    {
      // xoxo
      //
      // Generate temporal queries for real user queries
      //

      // Get real user queries with simulated queries added
      testQueries = GetExtendedRealQueries(kwScDataId, dataSource, simUser);
    }
    else 
    {
      // Get simulated queries
      testQueries = GetSimulatedQueries(kwScDataId, dataSource, simUser);
    }
  }
  else
  {
    // Get queries
    testQueries = GetCachedQueries(kwScDataId, dataSource);
  }
  
  // Get desired transformation
  auto pNetDataTransformFn = GetAggregationById(aggId);
  // Setup transformation correctly
  pNetDataTransformFn->SetTransformationSettings(netDataTransformSettings);

  // Get disired model
  auto pRankingModel = GetRankingModelById(modelId);
  // Setup model correctly
  pRankingModel->SetModelSettings(aggModelSettings);

  // Run test
  return pRankingModel->RunModelTest(kwScDataId, pNetDataTransformFn, &_indexKwFrequency, testQueries, _images);
}

std::vector<UserImgQueryRaw>& ImageRanker::GetCachedQueriesRaw(QueryOriginId dataSource) const
{
  static std::vector<UserImgQueryRaw> cachedData0;
  static std::vector<UserImgQueryRaw> cachedData1;
  static std::vector<UserImgQueryRaw> cachedData2;

  static std::chrono::steady_clock::time_point cachedData0Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData1Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData2Ts = std::chrono::steady_clock::now();

  auto currentTime = std::chrono::steady_clock::now();

  switch (dataSource)
  {
  case QueryOriginId::cDeveloper:

    if (cachedData0.empty() || cachedData0Ts < currentTime)
    {
      cachedData0.clear();

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
        std::vector<size_t> queryWordnetIds{ _pViretKws->GetCanonicalQueryNoRecur(idQueryRow[1]) };

        cachedData0.emplace_back(std::move(imageId), std::move(queryWordnetIds));
      }

      cachedData0Ts = std::chrono::steady_clock::now();
      cachedData0Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);

    }

    return cachedData0;

    break;

  case QueryOriginId::cPublic:

    if (cachedData1.empty() || cachedData1Ts < currentTime)
    {
      cachedData1.clear();

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
        std::vector<size_t> queryWordnetIds{ _pViretKws->GetCanonicalQueryNoRecur(idQueryRow[1]) };

        cachedData1.emplace_back(std::move(imageId), std::move(queryWordnetIds));
      }

      cachedData1Ts = std::chrono::steady_clock::now();
      cachedData1Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
    }

    return cachedData1;

    break;
  }

  return cachedData0;
}

UserImgQuery ImageRanker::GetSimulatedQueryForImage(size_t imageId, const SimulatedUser& simUser) const
{
  constexpr size_t from{ 2_z };
  constexpr size_t to{ 6_z };

  auto pImgData{ GetImageDataById(imageId) };

  const auto& linBinVector{ pImgData->_rawSimUserData.at(std::tuple(eKeywordsDataType::cViret1, eImageScoringDataType::cNasNet)) };

  // Calculate transformed vector
  float totalSum{ 0.0f };
  std::vector<float> transformedData;
  for (auto&& value : linBinVector)
  {
    float newValue{ pow(value, simUser.m_exponent) };
    transformedData.push_back(newValue);

    totalSum += newValue;
  }

  // Get scaling coef
  float scaleCoef{ 1 / totalSum };

  // Normalize
  size_t i{ 0_z };
  float cummulSum{ 0.0f };
  for (auto&& value : transformedData)
  {
    cummulSum += value * scaleCoef;
    
    transformedData[i] = cummulSum;

    ++i;
  }

  std::random_device randDev;
  std::mt19937 generator(randDev());
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  auto randLabel{ static_cast<float>(distribution(generator)) };

  size_t numLabels{ static_cast<size_t>((randLabel * (to - from)) + from) };
  std::vector<size_t> queryLabels;
  for (size_t i{ 0_z }; i < numLabels; ++i)
  {
    // Get random number between [0, 1] from uniform distribution
    float rand{static_cast<float>(distribution(generator)) };
    size_t labelIndex{ 0 };

    // Iterate through discrete points while we haven't found correct point
    while (transformedData[labelIndex] < rand)
    {
      ++labelIndex;
    }

    queryLabels.push_back(labelIndex);
  }

  std::vector<Clause> queryFormula;
  queryFormula.reserve(numLabels);

  // Create final formula with wordnet IDs
  for (auto&& index : queryLabels)
  {
    auto a = _pViretKws->GetKeywordPtrByVectorIndex(index);
    Clause meta;
    meta.emplace_back(false, a->m_vectorIndex);

    queryFormula.emplace_back(std::move(meta));
  }

  return std::tuple(imageId, queryFormula, true);
}

std::vector< std::vector<UserImgQuery>> ImageRanker::GetSimulatedQueries(KwScoringDataId kwScDataId, QueryOriginId dataSource, const SimulatedUser& simUser) const
{
  // Determine what id would that be if not simulated user
  QueryOriginId dataSourceNotSimulated{ static_cast<QueryOriginId>(static_cast<int>(dataSource) - SIMULATED_QUERIES_ENUM_OFSET) };

  // Fetch real user queries to immitate them
  std::vector< std::vector<UserImgQuery>> realUsersQueries{ GetCachedQueries(kwScDataId, dataSourceNotSimulated) };

  // Prepare result structure
  std::vector< std::vector<UserImgQuery>> resultSimulatedQueries;
  // Reserve space properly
  resultSimulatedQueries.reserve(realUsersQueries.size());

  for (auto&& queries : realUsersQueries)
  {
    std::vector<UserImgQuery> singleQuery;

    for (auto&&[imageId, formula, withExamples] : queries)
    {
      auto simulatedQuery{ GetSimulatedQueryForImage(imageId, simUser) };

      singleQuery.push_back(std::move(simulatedQuery));
    }
    resultSimulatedQueries.push_back(std::move(singleQuery));
  }


  return resultSimulatedQueries;
}

std::vector< std::vector<UserImgQuery>> ImageRanker::GetExtendedRealQueries(KwScoringDataId kwScDataId, QueryOriginId dataSource, const SimulatedUser& simUser) const
{
  // Determine what id would that be if not simulated user
  QueryOriginId dataSourceNotSimulated{ static_cast<QueryOriginId>(static_cast<int>(dataSource) - USER_PLUS_SIMULATED_QUERIES_ENUM_OFSET) };

  // Fetch real user queries to immitate them
  std::vector< std::vector<UserImgQuery>> realUsersQueries{ GetCachedQueries(kwScDataId, dataSourceNotSimulated) };

  // Prepare result structure - copy of real user queries
  std::vector< std::vector<UserImgQuery>> resultSimulatedQueries{ realUsersQueries };

  std::random_device rd;     // only used once to initialise (seed) engine
  std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
  
  size_t iterator{ 0_z };
  for (auto&& queries : realUsersQueries)
  {
    for (auto&&[imageId, formula, withExamples] : queries)
    {
      auto imgIt{ _images.begin() + MapIdToVectorIndex(imageId) };

      if (imgIt == _images.end())
      {
        LOG_ERROR("aaa");
      }

      // Get image ptr
      Image* pImg{ imgIt->get() };

      size_t numSuccs{ pImg->m_numSuccessorFrames };
      if (numSuccs <= 0)
      {
        break;
      }
      // Get how much we will offset from this image
      std::uniform_int_distribution<size_t> uni(1, std::min((size_t)MAX_TEMP_QUERY_OFFSET, numSuccs));
      size_t offset{ uni(rng) };

      // Offset iterator
      for (size_t i{ 0_z }; i < offset; ++i)
      {
        ++imgIt;
      }

      auto simulatedQuery{ GetSimulatedQueryForImage(pImg->m_imageId, simUser) };

      resultSimulatedQueries[iterator].push_back(std::move(simulatedQuery));

      // Only the first one
      break;
    }
    
    ++iterator;
  }

  return resultSimulatedQueries;
}

std::vector< std::vector<UserImgQuery>>& ImageRanker::GetCachedQueries(KwScoringDataId kwScDataId, QueryOriginId dataSource) const
{
  static std::vector < std::vector<UserImgQuery>> cachedAllData;
  static std::vector < std::vector<UserImgQuery>> cachedData0;
  static std::vector < std::vector<UserImgQuery>> cachedData1;
  static std::vector < std::vector<UserImgQuery>> cachedData2;

  static std::chrono::steady_clock::time_point cachedAllDataTs = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData0Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData1Ts = std::chrono::steady_clock::now();
  static std::chrono::steady_clock::time_point cachedData2Ts = std::chrono::steady_clock::now();

  auto currentTime = std::chrono::steady_clock::now();

  if (dataSource == QueryOriginId::cAll)
  {
    if (cachedAllData.empty() || cachedAllDataTs < currentTime)
    {
      cachedAllData.clear();

      const auto& queries1 = GetCachedQueries(kwScDataId, QueryOriginId::cDeveloper);
      const auto& queries2 = GetCachedQueries(kwScDataId, QueryOriginId::cPublic);

      // Merge them
      std::copy(queries1.begin(), queries1.end(), back_inserter(cachedAllData));
      std::copy(queries2.begin(), queries2.end(), back_inserter(cachedAllData));

      cachedAllDataTs = std::chrono::steady_clock::now();
      cachedAllDataTs += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
    }

    return cachedAllData;
  }

  switch (dataSource) 
  {
  case QueryOriginId::cDeveloper:

    if (cachedData0.empty() || cachedData0Ts < currentTime)
    {
      cachedData0.clear();

      // Fetch pairs of <Q, Img>
      std::string query("\
        SELECT image_id, query, type FROM `" + _primaryDb.GetDbName() +"`.queries \
          WHERE ( type = " + std::to_string((int)dataSource) + " OR type =  " + std::to_string(((int)dataSource + 10)) + ") AND \
            keyword_data_type = " + std::to_string((int)std::get<0>(kwScDataId)) + " AND \
            scoring_data_type = " + std::to_string((int)std::get<0>(kwScDataId)) + ";");
      
      auto dbData = _primaryDb.ResultQuery(query);

      if (dbData.first != 0)
      {
        LOG_ERROR("Error getting queries from database."s);
      }

      // Parse DB results
      for (auto&& idQueryRow : dbData.second)
      {
        
        size_t imageId{ static_cast<size_t>(strToInt(idQueryRow[0].data())) * TEST_QUERIES_ID_MULTIPLIER };

        CnfFormula queryFormula{ GetCorrectKwContainerPtr(kwScDataId)->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), IGNORE_CONSTRUCTED_HYPERNYMS) };
        bool wasWithExamples{ (bool)((strToInt(idQueryRow[2]) / 10) % 2 ) };

#if RUN_TESTS_ONLY_ON_NON_EMPTY_POSTREMOVE_HYPERNYM

        CnfFormula queryFormulaTest{ GetCorrectKwContainerPtr(kwScDataId)->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), true) };
        if (!queryFormulaTest.empty())

#else 

        if (!queryFormula.empty())

#endif
        {
          std::vector<UserImgQuery> tmp;
          tmp.emplace_back(std::move(imageId), std::move(queryFormula), wasWithExamples);

          cachedData0.emplace_back(std::move(tmp));
        }
      }

      cachedData0Ts = std::chrono::steady_clock::now();
      cachedData0Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);

    }

    return cachedData0;

    break;

  case QueryOriginId::cPublic:

    if (cachedData1.empty() || cachedData1Ts < currentTime)
    {
      cachedData1.clear();

      // Fetch pairs of <Q, Img>
      std::string query("\
        SELECT image_id, query, type FROM `" + _primaryDb.GetDbName() + "`.queries \
          WHERE (type = " + std::to_string((int)dataSource) + " OR type = " + std::to_string(((int)dataSource + 10)) + ") AND \
            keyword_data_type = " + std::to_string((int)std::get<0>(kwScDataId)) + " AND \
            scoring_data_type = " + std::to_string((int)std::get<0>(kwScDataId)) + ";");

      auto dbData = _primaryDb.ResultQuery(query);

      if (dbData.first != 0)
      {
        LOG_ERROR("Error getting queries from database."s);
      }

      // Parse DB results
      for (auto&& idQueryRow : dbData.second)
      {
        size_t imageId{ static_cast<size_t>(strToInt(idQueryRow[0].data())) * TEST_QUERIES_ID_MULTIPLIER };

        CnfFormula queryFormula{ GetCorrectKwContainerPtr(kwScDataId)->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), IGNORE_CONSTRUCTED_HYPERNYMS) };
        bool wasWithExamples{ (bool)((strToInt(idQueryRow[2]) / 10) % 2) };

#if RUN_TESTS_ONLY_ON_NON_EMPTY_POSTREMOVE_HYPERNYM

        CnfFormula queryFormulaTest{ GetCorrectKwContainerPtr(kwScDataId)->GetCanonicalQuery(EncodeAndQuery(idQueryRow[1]), true) };
        if (!queryFormulaTest.empty())

#else 
        if (!queryFormula.empty())
#endif
        {
          std::vector<UserImgQuery> tmp;
          tmp.emplace_back(std::move(imageId), std::move(queryFormula), wasWithExamples);

          cachedData1.emplace_back(std::move(tmp));
        }
      }

      cachedData1Ts = std::chrono::steady_clock::now();
      cachedData1Ts += std::chrono::seconds(QUERIES_CACHE_LIFETIME);
    }

    return cachedData1;

    break;
  }

  return cachedData0;
}

void ImageRanker::SubmitInteractiveSearchSubmit(
  InteractiveSearchOrigin originType, size_t imageId, RankingModelId modelId, InputDataTransformId transformId,
  std::vector<std::string> modelSettings, std::vector<std::string> transformSettings,
  std::string sessionId, size_t searchSessionIndex, int endStatus, size_t sessionDuration,
  std::vector<InteractiveSearchAction> actions,
  size_t userId
)
{
  size_t isEmpty{ 0_z };

  size_t countIn{ 0_z };
  size_t countOut{ 0_z };

  for (auto&& a : actions)
  {
    if (std::get<0>(a) == 0)
    {
      ++countOut;
    }
    else if (std::get<0>(a) == 1 || std::get<0>(a) == 2)
    {
      ++countIn;
    }
  }

  if (countIn == countOut)
  {
    isEmpty = 1_z;
  }

  std::stringstream query1Ss;
  query1Ss << "INSERT INTO `interactive_searches`(`type`, `target_image_id`, `model_id`, `transformation_id`, `model_settings`, `transformation_settings`, `session_id`, `user_id`, `search_session_index`, `end_status`, `session_duration`,`is_empty`)";
  query1Ss << "VALUES(" << (int)originType << "," << imageId << "," << (int)modelId << "," << (int)transformId << ",";

  query1Ss << "\"";
  for (auto&& s : modelSettings)
  {
    query1Ss << s << ";";
  }
  query1Ss << "\"";
  query1Ss << ",";
  query1Ss << "\"";
  for (auto&& s : transformSettings)
  {
    query1Ss << s << ";";
  }
  query1Ss << "\"";
  query1Ss << ",\"" << sessionId << "\"," << userId << "," << searchSessionIndex << "," << endStatus << "," << sessionDuration << "," << isEmpty << ");";  

  std::string query1{query1Ss.str()};
  size_t result1{ _primaryDb.NoResultQuery(query1) };

  size_t id{ _primaryDb.GetLastId() };


  std::stringstream query2Ss;
  query2Ss << "INSERT INTO `interactive_searches_actions`(`interactive_search_id`, `index`, `action`, `score`, `operand`)";
  query2Ss << "VALUES";
  {
    size_t i{ 0_z };
    for (auto&& action : actions)
    {
      query2Ss << "(" << id << "," << i << "," << std::get<0>(action) << "," << std::get<1>(action) << "," << std::get<2>(action) << ")";

      if (i < actions.size() - 1)
      {
        query2Ss << ",";
      }
      ++i;
    }
    query2Ss << ";";
  }
  

 
  std::string query2{ query2Ss.str() };
  size_t result2{ _primaryDb.NoResultQuery(query2) };

  if (result1 != 0 || result2 != 0)
  {
    LOG_ERROR("Failed to insert search session result.");
  }
}

std::tuple<UserAccuracyChartData, UserAccuracyChartData> ImageRanker::GetStatisticsUserKeywordAccuracy(QueryOriginId queriesSource) const
{
  LOG_ERROR("Not implemented: GetStatisticsUserKeywordAccuracy()");
  return std::tuple<UserAccuracyChartData, UserAccuracyChartData>();
  /*
  std::vector<UserImgQueryRaw> queries;

  if (queriesSource == QueryOriginId::cAll)
  {
    queries = GetCachedQueriesRaw(QueryOriginId::cDeveloper);
    auto queries2 = GetCachedQueriesRaw(QueryOriginId::cPublic);

    // Merge them
    std::copy(queries2.begin(), queries2.end(), back_inserter(queries));
  }

  std::vector<size_t> hitsNonHyper;
  size_t hitsNonHyperTotal{0ULL};
  hitsNonHyper.resize(_pViretKws->GetNetVectorSize());

  std::vector<size_t> hitsHyper;
  size_t hitsHyperTotal{ 0ULL };
  auto pImg{ GetImageDataById(0) };
  hitsHyper.resize(pImg->m_hypernymsRankingSorted.size());

  for (auto&&[imgId, wnIds] : queries)
  {
    auto pImg{ GetImageDataById(imgId) };


    for (auto&& wordnetId : wnIds)
    {
      // Get keyword
      auto kw{ _pViretKws->GetWholeKeywordByWordnetId(wordnetId) };

      // If is non-hyper
      if (kw->m_vectorIndex != SIZE_T_ERROR_VALUE)
      {
        
        const auto& rankVec{ pImg->_rawImageScoringDataSorted };

        for (size_t i{ 0ULL }; i < rankVec.size(); ++i)
        {
          // If this is the word
          if (rankVec[i].first == kw->m_vectorIndex)
          {
            ++hitsNonHyper[i];
            ++hitsNonHyperTotal;
          }
        }
      }
      // Is hypernym
      else
      {
        const auto& rankVec{ pImg->m_hypernymsRankingSorted };

        for (size_t i{ 0ULL }; i < rankVec.size(); ++i)
        {
          // If this is the word
          if (rankVec[i].first == wordnetId)
          {
            ++hitsHyper[i];
            ++hitsHyperTotal;
          }
        }
      }
    }
  }

  float percNonHyper{ (float)hitsNonHyperTotal / (hitsHyperTotal + hitsNonHyperTotal) };
  float percHyper{ 1 - percNonHyper };


  UserAccuracyChartDataMisc nonHyperMisc{ (size_t)queriesSource, percNonHyper };
  UserAccuracyChartDataMisc hyperMisc{ (size_t)queriesSource, percHyper };

  float scaleDownNonHyper{ 10 };
  float scaleDownHyper{ 10 };

  ChartData nonHyperChartData;
  ChartData hyperChartData;


  nonHyperChartData.emplace_back(0, 0);
  {
    size_t chIndex{ 1ULL };
    for (auto it = hitsNonHyper.begin(); it != hitsNonHyper.end(); )
    {
      size_t locMax{ 0ULL };
      for (size_t i{ 0ULL }; i < scaleDownNonHyper; ++i, ++it)
      {
        if (it == hitsNonHyper.end())
        {
          break;
        }

        if (*it > locMax)
        {
          locMax = *it;
        }
      }

      nonHyperChartData.emplace_back(static_cast<uint32_t>(chIndex * scaleDownNonHyper), static_cast<uint32_t>(locMax));
      ++chIndex;
    }
  }

  hyperChartData.emplace_back(0, 0);

  {
    size_t chIndex{ 1ULL };
    for (auto it = hitsHyper.begin(); it != hitsHyper.end();)
    {
      size_t locMax{ 0ULL };
      for (size_t i{ 0ULL }; i < scaleDownHyper; ++i, ++it)
      {
        if (it == hitsHyper.end())
        {
          break;

        }
        if (*it > locMax)
        {
          locMax = *it;
        }
      }

      hyperChartData.emplace_back(static_cast<uint32_t>(chIndex * scaleDownNonHyper), static_cast<uint32_t>(locMax));
      ++chIndex;
    }
  }

  UserAccuracyChartData nonHyperData{ std::pair(std::move(nonHyperMisc), std::move(nonHyperChartData)) };
  UserAccuracyChartData hyperData{ std::pair(std::move(hyperMisc), std::move(hyperChartData)) };

  return std::tuple(std::move(nonHyperData), std::move(hyperData));
  */

}


KeywordsContainer* ImageRanker::GetCorrectKwContainerPtr(KwScoringDataId kwScDataId) const
{
  KeywordsContainer* ptr{nullptr};

  switch (std::get<0>(kwScDataId))
  {
  case eKeywordsDataType::cViret1:
    ptr = _pViretKws;
    break;

  case eKeywordsDataType::cGoogleAI:
    ptr = _pGoogleKws;
    break;

  default:
    LOG_ERROR("ImageRanker::GetCorrectKwContainerPtr(): Incorrect KW type!");
  }

  return ptr;
}

RelevantImagesResponse ImageRanker::GetRelevantImages(
  KwScoringDataId kwScDataId,
  const std::vector<std::string>& queriesEncodedPlaintext, size_t numResults,
  InputDataTransformId aggId, RankingModelId modelId,
  const RankingModelSettings& modelSettings, const InputDataTransformSettings& aggSettings,
  size_t imageId, 
  bool withOccuranceValue
) const
{
  std::vector<CnfFormula> formulae;

  KeywordsContainer* pKws{ GetCorrectKwContainerPtr(kwScDataId)};
  
  for (auto&& queryString : queriesEncodedPlaintext)
  {
    // Decode query to logical CNF formula
    CnfFormula queryFormula{ pKws->GetCanonicalQuery(EncodeAndQuery(queryString)) };

    formulae.push_back(queryFormula);
  }

  // Get desired aggregation
  auto pAggFn = GetAggregationById(aggId);
  // Setup model correctly
  pAggFn->SetTransformationSettings(aggSettings);

  // Get disired model
  auto pRankingModel = GetRankingModelById(modelId);
  // Setup model correctly
  pRankingModel->SetModelSettings(modelSettings);

  // Rank it
  auto [imgOrder, targetImgRank] {pRankingModel->GetRankedImages(formulae, kwScDataId, pAggFn, &_indexKwFrequency, _images, numResults, imageId)};


  RelevantImagesResponse resultResponse;

  std::vector<std::tuple<size_t, std::string, float>> occuranceHistogram;
  occuranceHistogram.reserve(pKws->GetNetVectorSize());


  if (withOccuranceValue)
  {
    //// Prefil keyword wordnetIDs
    //for (size_t i{ 0ULL }; i < _pViretKws->GetNetVectorSize(); ++i)
    //{
    //  occuranceHistogram.emplace_back(std::get<0>(_pViretKws->GetKeywordByVectorIndex(i)), std::get<1>(_pViretKws->GetKeywordByVectorIndex(i)), 0.0f);
    //}

    //for (auto&& imgId : imgOrder)
    //{
    //  auto imagePtr = GetImageDataById(imgId);
    //  auto min = imagePtr->m_min;

    //  const auto& linBinVector{ imagePtr->_transformedImageScoringData.at(200) };

    //  // Add ranking to histogram
    //  for (auto&& r : imagePtr->_rawImageScoringDataSorted)
    //  {

    //    std::get<2>(occuranceHistogram[r.first]) += linBinVector[r.first];
    //  }
    //}

    //// Sort suggested list
    //std::sort(occuranceHistogram.begin(), occuranceHistogram.end(),
    //  [](const std::tuple<size_t, std::string, float>& l, std::tuple<size_t, std::string, float>& r)
    //  {
    //    return std::get<2>(l) > std::get<2>(r);
    //  }
    //);
  }

  // Prepare final result to return
  {
    size_t i{0ULL};
    for (auto&& imgId : imgOrder)
    {
      std::get<0>(resultResponse).emplace_back(GetImageDataById(imgId));
      //std::get<1>(resultResponse).emplace_back(std::move(occuranceHistogram[i]));

      ++i;
    }
  }

  // Fill in QueryResult
  std::get<2>(resultResponse) = targetImgRank;

  return resultResponse;
}

const Image* ImageRanker::GetImageDataById(size_t imageId) const
{
  // Get correct image index
  size_t index{imageId / _imageIdStride};

  if (index >= _images.size())
  {
    LOG_ERROR("Out of bounds image index."s);
    return nullptr;
  }

  return _images[index].get();
}

Image* ImageRanker::GetImageDataById(size_t imageId)
{
  // Get correct image index
  size_t index{ imageId / _imageIdStride };

  if (index >= _images.size())
  {
    LOG_ERROR("Out of bounds image index."s);
    return nullptr;
  }

  return _images[index].get();
}

std::string ImageRanker::GetImageFilenameById(size_t imageId) const
{
  auto imgPtr{ GetImageDataById(imageId) };

  if (imgPtr == nullptr)
  {
    LOG_ERROR("Incorrect image ID."s);
    return ""s;
  }

  return imgPtr->m_filename;
}




// =======================================================================================
// =======================================================================================
// =======================================================================================
// =======================================================================================
// =======================================================================================
// =======================================================================================



SimulatedUser ImageRanker::GetSimUserSettings(const SimulatedUserSettings& stringSettings) const
{
  SimulatedUser newSimUser;

  // If setting 0 (Simulated user exponent) is set
  if (stringSettings.size() >= 1 && stringSettings[0].size() >= 0)
  {
    newSimUser.m_exponent = strToInt(stringSettings[0]);
  }

  return newSimUser;
}

void ImageRanker::ComputeApproxDocFrequency(size_t aggregationGuid, float treshold)
{
  LOG("Not implemented: omputeApproxDocFrequency()!");

  /*
  _indexKwFrequency.reserve(_pViretKws->GetNetVectorSize());

  std::vector<size_t> indexKwFrequencyCount;
  indexKwFrequencyCount.resize(_pViretKws->GetNetVectorSize());


  // Iterate thorough all images
  for (auto&& [id, pImg] : _images)
  {
    auto it = pImg->_transformedImageScoringData.find(aggregationGuid);

    if (it == pImg->_transformedImageScoringData.end())
    {
      LOG_ERROR("Aggregation GUID"s + std::to_string(aggregationGuid) + " not found.");
    }

    {
      size_t i{ 0ULL };
      for (auto&& fl : it->second)
      {
        // If this keyword is truly present
        if (fl > treshold)
        {
          // Increment it's count
          ++indexKwFrequencyCount[i];
        }

        ++i;
      }
    }
  }

  // Find maximum
  std::pair<size_t, size_t> maxIndexCount{ 0ULL, 0ULL };
  {
    size_t i{ 0ULL };
    for (auto&& indexCount : indexKwFrequencyCount)
    {
      if (indexCount > maxIndexCount.second)
      {
        maxIndexCount.first = i;
        maxIndexCount.second = indexCount;
      }

      ++i;
    }
  }

  size_t i{ 0ULL };
  for (auto&& indexCount : indexKwFrequencyCount)
  {
    _indexKwFrequency.emplace_back( logf(( (float)maxIndexCount.second / indexCount)));
  }
  */
}

void ImageRanker::GenerateBestHypernymsForImages()
{
  /*
  auto cmp = [](const std::pair<size_t, float>& left, const std::pair<size_t, float>& right)
  {
    return left.second < right.second;
  };

  for (auto&& [imgId, pImg] : _images)
  {
    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> maxHeap(cmp);

    for (auto&& [wordnetId, pKw] : _pViretKws->_wordnetIdToKeywords)
    {

      // If has some hyponyms
      if (pKw->m_vectorIndex == SIZE_T_ERROR_VALUE)
      {
        float totalRank{ 0.0f };
        for (auto&& kwIndex : pKw->m_hyponymBinIndices)
        {
          totalRank += pImg->m_rawNetRanking[kwIndex];
        }

        maxHeap.push(std::pair(wordnetId, totalRank));
      }
    }

    while (!maxHeap.empty())
    {
      auto item = maxHeap.top();
      maxHeap.pop();

      pImg->m_hypernymsRankingSorted.emplace_back(std::move(item));
    }

  }
  */
}

void ImageRanker::InitializeGridTests()
{
  LOG("Not implemented: ImageRanker::InitializeGridTests()!");

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

std::pair<uint8_t, uint8_t> ImageRanker::GetGridTestProgress() const
{ 
  return GridTest::GetGridTestProgress(); 
}

// \todo Export to new class Exporter
void ImageRanker::PrintIntActionsCsv() const
{
  std::string query1{ "SELECT id, session_duration, end_status FROM `image-ranker-collector-data2`.interactive_searches;" };
  std::string query2{ "SELECT `interactive_search_id`, `index`, `action`, `score`, `operand` FROM `image-ranker-collector-data2`.interactive_searches_actions;" };
  auto result1{ _primaryDb.ResultQuery(query1) };
  auto result2{ _primaryDb.ResultQuery(query2) };

  auto actionIt{ result2.second.begin() };


  std::vector<std::vector<size_t>> sessProgresses;

  for (auto&& actionSess : result1.second)
  {
    std::vector<size_t> oneSess;

    size_t sessId{ (size_t)strToInt(actionSess[0]) };
    size_t sessDuration{ (size_t)strToInt(actionSess[1]) };
    size_t endStatus{ (size_t)strToInt(actionSess[2]) };

    bool isInitial{ true };
    std::vector<std::string> initialQuery;
    std::vector<std::string> fullQuery;

    size_t actionInitialCount{ 0_z };
    size_t actionFinalCount{ 0_z };
    std::string initialRank{ "" };
    std::string finalRank{ "" };

    for (; (result2.second.end() != actionIt && strToInt((*actionIt)[0]) == sessId); ++actionIt)
    {


      auto&& actionRow = (*actionIt);

      if (actionRow[2] == "2")
      {
        isInitial = false;


        // Push rank before start of interactive refining
        oneSess.push_back(strToInt(initialRank));
      }


      if (isInitial)
      {
        ++actionInitialCount;
        initialRank = actionRow[3];
        finalRank = actionRow[3];
      }
      else
      {
        ++actionFinalCount;
        ++actionInitialCount;
        finalRank = actionRow[3];

        // Push rank 
        oneSess.push_back(strToInt(finalRank));
      }

      if (actionRow[2] == "1" || actionRow[2] == "2")
      {
        if (isInitial)
        {
          initialQuery.push_back(actionRow[4]);
        }
        fullQuery.push_back(actionRow[4]);
      }
      else
      {
        for (auto it = initialQuery.begin(); it != initialQuery.end(); ++it)
        {
          if (*it == actionRow[4])
          {
            initialQuery.erase(it);
            break;
          }
        }

        for (auto it = fullQuery.begin(); it != fullQuery.end(); ++it)
        {
          if (*it == actionRow[4])
          {
            fullQuery.erase(it);
            break;
          }
        }
      }
    }




    if (initialQuery.empty() || fullQuery.empty())
    {
      continue;
    }

    std::cout << std::to_string(sessId) << "," << std::to_string(sessDuration) << "," << std::to_string(endStatus);

    {
      size_t initSize = initialQuery.size();
      size_t i{ 0_z };
      for (auto&& kwId : initialQuery)
      {
        std::cout << kwId;

        if (i < initSize - 1)
        {
          std::cout << "&";
        }
        ++i;
        std::cout << ",";
      }
    }
    std::cout << initialRank << ",";
    std::cout << std::to_string(actionInitialCount) << ",";

    {
      size_t initSize = fullQuery.size();
      size_t i{ 0_z };
      for (auto&& kwId : fullQuery)
      {
        std::cout << kwId;

        if (i < initSize - 1)
        {
          std::cout << "&";
        }
        ++i;
        std::cout << ",";
      }
    }
    if (!oneSess.empty())
    {
      sessProgresses.emplace_back(std::move(oneSess));
    }



    std::cout << finalRank << ",";
    std::cout << std::to_string(actionFinalCount) << std::endl;
  }


  std::set<size_t> m;

  for (auto&& vec : sessProgresses)
  {
    auto s{ vec.size() };
    m.insert(s);
  }

  std::vector<std::vector<size_t>> ddata;

  for (auto&& size : m)
  {
    std::vector<size_t> data;
    data.resize(size, 0_z);

    size_t i{ 0_z };


    for (auto&& vec : sessProgresses)
    {
      if (vec.size() == size)
      {
        size_t ii{ 0_z };
        for (auto&& d : vec)
        {
          data[ii] += d;
          ++ii;
        }

        ++i;
      }

    }

    // Divide
    for (auto&& d : data)
    {
      d = d / i;
    }

    ddata.push_back(data);
  }

}

#if TRECVID_MAPPING

std::tuple<float, std::vector<std::pair<size_t, size_t>>> ImageRanker::TrecvidGetRelevantShots(
  KwScoringDataId kwScDataId,
  const std::vector < std::string>& queriesEncodedPlaintext, size_t numResults,
  InputDataTransformId aggId, RankingModelId modelId,
  const RankingModelSettings& modelSettings, const InputDataTransformSettings& aggSettings,
  float elapsedTime,
  size_t imageId
)
{

#if DEBUG_SHOW_OUR_FRAME_IDS

  std::cout << "===============================" << std::endl;
  std::cout << "transformation ID = " << std::to_string((size_t)aggId) << std::endl;
  std::cout << "model ID = " << std::to_string((size_t)modelId) << std::endl;
  std::cout << "model settings: " << std::endl;
  for (auto&& q : modelSettings)
  {

    std::cout << q << std::endl;
  }

  std::cout << "transform settings:" << std::endl;
  for (auto&& q : aggSettings)
  {

    std::cout << q << std::endl;
  }
  std::cout << "elapsed time  = " << elapsedTime << std::endl;

#endif

  // Start timer
  auto start = std::chrono::steady_clock::now();

  std::vector<CnfFormula> formulae;

  for (auto&& queryString : queriesEncodedPlaintext)
  {
    // Decode query to logical CNF formula
    CnfFormula queryFormula{ _pViretKws->GetCanonicalQuery(EncodeAndQuery(queryString)) };

    formulae.push_back(queryFormula);
  }

  // Get desired aggregation
  auto pAggFn = GetAggregationById(aggId);
  // Setup model correctly
  pAggFn->SetTransformationSettings(aggSettings);

  // Get disired model
  auto pRankingModel = GetRankingModelById(modelId);
  // Setup model correctly
  pRankingModel->SetModelSettings(modelSettings);

  // Rank it
  auto [imgOrder, targetImgRank] {pRankingModel->GetRankedImages(formulae, kwScDataId, pAggFn, &_indexKwFrequency, _images, 40000, imageId)};



  std::vector<std::pair<size_t, size_t>> resultTrecvidShotIds;
  resultTrecvidShotIds.reserve(numResults);

#if DEBUG_SHOW_OUR_FRAME_IDS

  std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
  for (auto&& q : queriesEncodedPlaintext)
  {

    std::cout << "Q:" << q << std::endl;
  }
  std::cout << "--------------------" << std::endl;

#endif

  for (auto&& ourFrameId : imgOrder)
  {
    // If we have enough shots already
    if (resultTrecvidShotIds.size() >= numResults)
    {
      // Stop
      break;
    }

#if DEBUG_SHOW_OUR_FRAME_IDS

    std::cout << ourFrameId << std::endl;

#endif

    std::pair<size_t, size_t> trecvidVideoIdShotIdPair{ ConvertToTrecvidShotId(ourFrameId) };

    // If this shot is already picked
    if (trecvidVideoIdShotIdPair.first == SIZE_T_ERROR_VALUE || trecvidVideoIdShotIdPair.second == SIZE_T_ERROR_VALUE)
    {
      // Go on to next our frame
      continue;
    }

    // Check if it is dropped shot
    for (auto&&[dVideoId, dShotId] : _tvDroppedShots)
    {
      if (trecvidVideoIdShotIdPair.first == dVideoId && trecvidVideoIdShotIdPair.second == dShotId)
      {
        continue;
      }
    }

    // Add this TRECVID shot ID to resultset
    resultTrecvidShotIds.emplace_back(std::move(trecvidVideoIdShotIdPair));
  }

  // Stop timer
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);

  size_t calculationElapsedInMs{ static_cast<size_t>(duration.count()) };
  float totalElapsed{ elapsedTime + ((float)calculationElapsedInMs / 1000) };

  float totalElapsedRounded{ ((float)((int)(totalElapsed * 10))) / 10 };

  // Reset trecvid shot reference map
  ResetTrecvidShotMap();

#if 0

  std::set<std::pair<size_t, size_t>> set;
  for (auto&&[videoId, shotId] : resultTrecvidShotIds)
  {
    auto result{ set.insert(std::pair(videoId, shotId)) };

    if (result.second == false)
    {
      LOG_ERROR("Duplicate!!!");
    }
  }

#endif

  return std::tuple(totalElapsedRounded, std::move(resultTrecvidShotIds));
}

std::vector<std::vector<std::pair<std::pair<unsigned int, unsigned int>, bool>>>
ImageRanker::ParseTrecvidShotReferencesFromDirectory(const std::string& path) const
{
  std::vector<std::vector<std::pair<std::pair<unsigned int, unsigned int>, bool>>> resultMap;

  for (auto&& file : std::filesystem::directory_iterator(path))
  {
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, bool>> metaResult;

    // Open file for reading as binary from the end side
    std::ifstream ifs(file.path().string(), std::ios::ate);

    // If failed to open file
    if (!ifs)
    {
      LOG_ERROR("Error opening file: "s + file.path().string());
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


    size_t lineNr{ 0_z };
    std::string line;

    // Iterate until there is something to read from file
    while (std::getline(ifs, line))
    {
      ++lineNr;

      // Skip first line - there are only column headers
      if (lineNr == 1)
      {
        continue;
      }

      // WARNING:
      // TRECVID shot reference starts videos from 1, we do from 0
      // Index in this vector will match our indexing

      unsigned int frameFrom;
      unsigned int frameTo;
      float byteBin;
      std::stringstream lineStream(line);

      lineStream >> frameFrom;
      lineStream >> byteBin; // Throw this away
      lineStream >> frameTo;

      // Contains std::pair<std::pair<unsigned int, unsigned int>, bool>
      metaResult.emplace_back(std::pair(frameFrom, frameTo), false);
    }

    // Add this file reference to map
    resultMap.push_back(metaResult);
  }

  return resultMap;
}

std::vector<std::pair<size_t, size_t>> ImageRanker::ParseTrecvidDroppedShotsFile(const std::string& filepath) const
{
  std::vector<std::pair<size_t, size_t>> metaResult;

  // Open file for reading as binary from the end side
  std::ifstream ifs(filepath, std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file: "s + filepath);
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


  size_t lineNr{ 0_z };
  std::string line;

  // Iterate until there is something to read from file
  while (std::getline(ifs, line))
  {
    ++lineNr;

    // cut "shot" prefix
    line = line.substr(4);

    std::string videoIdStr{ line.substr(0, 5) };
    size_t videoId{ (size_t)strToInt(videoIdStr) };

    line = line.substr(6);
    size_t shotId{ (size_t)strToInt(line) };

    metaResult.emplace_back(videoId, shotId);
  }

  return metaResult;
}


void ImageRanker::ResetTrecvidShotMap()
{
  // Just reset all trues to falses
  for (auto&& submap : _trecvidShotReferenceMap)
  {
    for (auto&&[pair, isTaken] : submap)
    {
      isTaken = false;
    }
  }
}

std::pair<size_t, size_t> ImageRanker::ConvertToTrecvidShotId(size_t ourFrameId)
{
  auto ourFrameIdDowncasted{ static_cast<unsigned int>(ourFrameId) };

  // Get image pointer
  const Image* pImg{ GetImageDataById(ourFrameId) };
  auto ourFrameNumber{ pImg->m_frameNumber };

  // Get video ID, this is idx in trecvid map vector
  auto videoId{ static_cast<size_t>(pImg->m_videoId) };

  // Get correct submap for this video
  auto& videoMap{ _trecvidShotReferenceMap[videoId] };

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Return ID PLUS 1, because TRECVID vids start at 1 and our source file starts at 0
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  videoId = videoId + 1;

  size_t ourFrameNumberA;
  if (ourFrameNumber > 1)
  {
    ourFrameNumberA = ourFrameNumber - 1;
  }

  //
  // Binary search frame interval, that this frame belongs to
  //
  auto shotIntervalIt = std::lower_bound(videoMap.begin(), videoMap.end(), std::pair(std::pair(ourFrameNumber, ourFrameNumber), false),
    [](const std::pair<std::pair<size_t, size_t>, bool>& l, const std::pair<std::pair<size_t, size_t>, bool>& r)
    {
      auto lVal{ l.first };
      auto rVal{ r.first };

      return lVal.first < rVal.first && lVal.second < rVal.second;
    }
  );

  if (shotIntervalIt == videoMap.end())
  {
    std::cout << "videoId = " << videoId << std::endl;
    std::cout << "ourFrameNumber = " << ourFrameNumber << std::endl;
    std::cout << "shot ref intervals:" << std::endl;

    for (auto&&[pair, t] : videoMap)
    {
      std::cout << "[" << pair.first << ", " << pair.second << "]" << std::endl;
    }
    LOG("This frame not present in shot reference.");

    return std::pair(SIZE_T_ERROR_VALUE, SIZE_T_ERROR_VALUE);
  }

  // If this shot is already picked
  if (shotIntervalIt->second == true)
  {
    // Return "Fail value"
    return std::pair(SIZE_T_ERROR_VALUE, SIZE_T_ERROR_VALUE);
  }
  // Otherwise mark this shot as picked
  else
  {
    shotIntervalIt->second = true;
  }

  // Get idx of this iterator
  auto shotIdx{ shotIntervalIt - videoMap.begin() };
  assert(shotIdx >= 0);

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Return index PLUS 1, because TRECVID vids start at 1
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  size_t shotId{ static_cast<size_t>(shotIdx + 1) };

  return std::pair(videoId, shotId);
}
#endif

