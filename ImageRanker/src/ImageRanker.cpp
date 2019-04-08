#include "ImageRanker.h"


ImageRanker::ImageRanker(
  std::string_view imagesPath,
  std::string_view probabilityVectorFilepath,
  std::string_view deepFeaturesFilepath,
  std::string_view keywordClassesFilepath,
  std::string_view imagesListFilepath,
  size_t columnIndexFilename,
  size_t imageListFileLineLength,
  size_t numRows,
  size_t idOffset
):
  _imagesListFilepath(imagesListFilepath),
  _columnIndexFilename(columnIndexFilename),
  _imageListFileLineLength(imageListFileLineLength),
  _imagesPath(imagesPath),
  _numRows(numRows),
  _idOffset(idOffset),

  // Parse keywords from file
  _keywords(keywordClassesFilepath),

  // Parse images and probabilities from file
  //_images(),
  _primaryDb(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME),
  _secondaryDb(SECONDARY_DB_HOST, SECONDARY_DB_PORT, SECONDARY_DB_USERNAME, SECONDARY_DB_PASSWORD, SECONDARY_DB_DB_NAME)
{
  _images = ParseSoftmaxBinFile(probabilityVectorFilepath, _imagesListFilepath);
  


  // Connect to database
  auto result{ _primaryDb.EstablishConnection() };
  if (result != 0ULL)
  {
    LOG_ERROR("Connecting to primary DB failed.");
  }

#if PUSH_DATA_TO_DB

  PushDataToDatabase();

#endif
 
}

ImageRanker::ImageRanker(
  std::string_view imagesPath
) :
  _primaryDb(PRIMARY_DB_HOST, PRIMARY_DB_PORT, PRIMARY_DB_USERNAME, PRIMARY_DB_PASSWORD, PRIMARY_DB_DB_NAME),
  _secondaryDb(SECONDARY_DB_HOST, SECONDARY_DB_PORT, SECONDARY_DB_USERNAME, SECONDARY_DB_PASSWORD, SECONDARY_DB_DB_NAME)
{
  // Connect to database
  auto result{ _primaryDb.EstablishConnection() };
  if (result != 0ULL)
  {
    LOG_ERROR("Connecting to primary DB failed.");
  }

#if USE_SECONDARY_DB

  // Connect to secondary database
  auto result{ _secondaryDb.EstablishConnection() };
  if (result != 0ULL)
  {
    LOG_ERROR("Connecting to secondary DB failed.");
  }

#endif

  // Load from database
  LoadKeywordsFromDatabase(DATA_SOURCE_DB);
  LoadImagesFromDatabase(DATA_SOURCE_DB);
}

size_t ImageRanker::GetRandomImageId() const
{ 
  // Get random index
  return static_cast<size_t>(GetRandomInteger(0, (int)_numRows) * _idOffset);
}


ImageRanker::ImageReference ImageRanker::GetRandomImage() const
{
  size_t imageId{GetRandomImageId()};

  return ImageReference{ imageId, GetImageFilenameById(imageId) };
}

ImageRanker::KeywordReference ImageRanker::GetNearKeywords(const std::string& prefix)
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

std::vector<ImageRanker::GameSessionQueryResult> ImageRanker::SubmitUserQueriesWithResults(std::vector<ImageRanker::GameSessionInputQuery> inputQueries, QueryOrigin origin)
{
  /******************************
    Save query to database
  *******************************/
  // Input format:
  // <SessionID, ImageID, User query - "k_1&k_2& ... &k_n">

  // Resolve query origin
  size_t originNumber{ static_cast<size_t>(origin) };

  // Store it into database
  std::string sqlQuery{ "INSERT INTO `queries` (query, image_id, type) VALUES " };

  for (auto&& query : inputQueries)
  {
    // Get image ID
    size_t imageId = std::get<1>(query);
    std::string queryString = std::get<2>(query);

    sqlQuery += "('"s + queryString + "', "+ std::to_string(imageId) + ", "s + std::to_string(originNumber) + "),"s;
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
  std::vector<ImageRanker::GameSessionQueryResult> userResult;
  userResult.reserve(inputQueries.size());

  for (auto&& query : inputQueries)
  {
    // Get user keywords tokens
    std::vector<std::string> userKeywords{ TokenizeAndQuery(std::get<2>(query)) };

    // Get image ID
    size_t imageId = std::get<1>(query);

    // Get image filename
    std::string imageFilename{ GetImageFilenameById(imageId)};

    std::vector<std::pair<std::string, float>> netKeywordsProbs{};

    userResult.emplace_back(std::get<0>(query), std::move(imageFilename), std::move(userKeywords), GetHighestProbKeywords(imageId, 10ULL));
  }

  return userResult;
}

std::vector<std::pair<std::string, float>> ImageRanker::GetHighestProbKeywords(size_t imageId, size_t N) const
{
  // Find image in map
  auto imagePair = _images.find(imageId);

  // If no such image
  if (imagePair == _images.end())
  {
    LOG_ERROR("No image found!");
    return std::vector<std::pair<std::string, float>>();
  }

  // Construct new subvector
  std::vector<std::pair<std::string, float>> result;
  result.reserve(N);

  auto probabilites = imagePair->second.m_probabilityVector;

  // Get first N highest probabilites
  for (size_t i = 0ULL; i < N; ++i)
  {
    size_t index{ probabilites[i].first };
    float probability{ probabilites[i].second };

    // Get keyword string
    std::string keyword{ _keywords.GetKeywordByVectorIndex(index) }; ;

    // Place it into result vector
    result.emplace_back(std::pair(keyword, probability));
  }

  return result;
}

std::vector<std::string> ImageRanker::TokenizeAndQuery(std::string_view query) const
{
  // Create sstram from query
  std::stringstream querySs{query.data()};

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

std::vector<std::string> ImageRanker::ParseImageFilenamesTextFile(std::string_view filepath) const
{
  // Open file with list of files in images dir
  std::ifstream inFile(_imagesListFilepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    throw std::runtime_error(std::string("Error opening file :") + _imagesListFilepath);
  }

  std::vector<std::string> result;
  result.reserve(_numRows);

  std::string line;

  // While there are lines in file
  while (std::getline(inFile, line))
  {
    // Extract file name
    std::stringstream ss(line);

    std::string columnData;

    // Throw away correct number of columns
    for (size_t i = 0ULL; i < _columnIndexFilename; ++i)
    {
      ss >> columnData;
    }

    // Get file name
    ss >> columnData;

    result.emplace_back(columnData);
  }

  // Return result filepath 
  return result;
}

std::map<size_t, Image> ImageRanker::ParseSoftmaxBinFile(std::string_view filepath, std::string_view imageFilesFilepath) const
{
  std::vector<std::string> imageFilenames{ParseImageFilenamesTextFile(imageFilesFilepath)};

  // Create buffer from file
  Buffer buffer = LoadFileToBuffer(filepath);

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(buffer, 36ULL);

  // Byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Where rows data start
  size_t currOffset = 40ULL;

  // Declare result vector
  std::map<size_t, Image> images;


  // Size of buffer in bytes
  size_t bufferSize = buffer.size();

  // Iterate until buffer is empty
  while (currOffset + byteRowLengths <= bufferSize)
  {
    // Get picture ID of this row
    size_t id = ParseIntegerLE(buffer, currOffset);

    // Stride in bytes
    currOffset += sizeof(float);

    // Initialize vector of floats for this row
    std::vector<std::pair<size_t, float>> floats;
    std::vector<std::pair<size_t, uint8_t>> boolVector;

    // Reserve exact capacitys
    floats.reserve(numFloats);

    // Iterate through all floats in row
    for (size_t i = 0; i < numFloats; ++i)
    {
      float probability{ ParseFloatLE(buffer, currOffset) };

      // Push float value in
      floats.push_back(std::make_pair(i, probability));

    #if GENERATE_BOOL_IMAGE_VECTOR_NAIVELY

      if (probability >= IS_TRUE_TRESHOLD)
      {
        boolVector.push_back(std::pair(i, 1));
      }
      else 
      {
        boolVector.push_back(std::pair(i, 0));
      }
      
    #endif


      // Stride in bytes
      currOffset += sizeof(float);
    }

    // Get image filename 
    std::string filename{ imageFilenames[id / _idOffset] };

    // Sort probabilites
    std::sort(
      floats.begin(), floats.end(), 
      [](const std::pair<size_t, float>& a, const std::pair<size_t, float> & b) -> bool
      {
        return a.second > b.second;
      }
    );

    // Push final row
    images.emplace(std::make_pair(id, Image(id, std::move(filename), std::move(floats), std::move(boolVector))));
  }

  

  return images;
}


std::vector<ImageRanker::ImageReference> ImageRanker::GetRelevantImages(const std::string& query, RankingModel rankingModel) const
{
  switch (rankingModel) 
  {
  case ImageRanker::cBoolean:
    return GetRelevantImagesBooleanModel(query);
    break;

  case ImageRanker::cFuzzyLogic:
    return GetRelevantImagesFuzzyLogicModel(query);
    break;
  }

  return std::vector<ImageRanker::ImageReference>();
}



std::vector<ImageRanker::ImageReference> ImageRanker::GetRelevantImagesFuzzyLogicModel(const std::string& query) const
{
  return std::vector<ImageRanker::ImageReference>();
}

std::vector<ImageRanker::ImageReference> ImageRanker::GetRelevantImagesBooleanModel(const std::string& query) const
{
  CnfFormula fml = _keywords.GetCanonicalQuery(query);


  std::vector<ImageRanker::ImageReference> result;

  // Check every image if satisfies query formula
  for (auto&& idImgPair : _images) 
  {
    const Image& img{ idImgPair.second };
    bool imageSucc{true};

    // Itarate through clauses connected with AND
    for (auto&& clause : fml) 
    {
      bool clauseSucc{ false };

      // Iterate through predicates
      for (auto&& var : clause) 
      {
        // If this variable satisfies this clause
        if (img.m_booleanProbVector[var].second != 0) 
        {
          clauseSucc = true;
          break;
        }
      }

      // If this clause not satisfied
      if (!clauseSucc) 
      {
        imageSucc = false;
        break;
      }
    }

    // If image satisfies formula
    if (imageSucc) 
    {
      // Insert this file into result set
      result.emplace_back(std::pair(img.m_imageId, img.m_filename));
    }

  }

  return result;
}

std::vector< std::pair< size_t, std::unordered_map<size_t, float>>> ImageRanker::ParseSoftmaxBinFileFiltered(std::string_view filepath, float minProbabilty) const 
{
  // Create buffer from file
  Buffer buffer = LoadFileToBuffer(filepath);

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(buffer, 36ULL);

  // Byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Where rows data start
  size_t currOffset = 40ULL;

  // Result data structure
  std::vector< std::pair< size_t, std::unordered_map<size_t, float> > > rows;
  // Prepare enough space right away
  rows.reserve(NUM_ROWS);

  // Size of buffer in bytes
  size_t bufferSize = buffer.size();

  // Iterate until buffer is empty
  while (currOffset + byteRowLengths <= bufferSize)
  {
    // Get picture ID of this row
    size_t id = ParseIntegerLE(buffer, currOffset);

    // Stride in bytes
    currOffset += sizeof(float);

    // Initialize vector of floats for this row
    std::unordered_map<size_t, float> floats;
    // Reserve exact capacitys
    floats.reserve(numFloats);

    // Iterate through all floats in row
    for (size_t i = 0; i < numFloats; ++i)
    {
      // Get this probability
      float probability = ParseFloatLE(buffer, currOffset);

      // If this probability big enough
      if (probability >= minProbabilty)
      {
        // Push float value in
        floats.insert(std::make_pair(i, probability));
      }

      // Stride in bytes
      currOffset += sizeof(float);
    }

    // Push final row inside hash table
    rows.push_back(std::make_pair(id, floats));
  }

  return rows;
}

std::unordered_map<size_t, std::pair<size_t, std::string> > ImageRanker::ParseKeywordClassesTextFile(std::string_view filepath) const
{
  // Open file with list of files in images dir
  std::ifstream inFile(filepath.data(), std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    throw std::runtime_error(std::string("Error opening file :") + filepath.data());
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
    throw std::runtime_error(std::string("Error opening file :") + filepath.data());
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

  return imgPair->second.m_filename;
}


std::string ImageRanker::GetImageFilepathByIndex(size_t imgIndex, bool relativePaths) const
{

  // Open file with list of files in images dir
  std::ifstream inFile(_imagesListFilepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    throw std::runtime_error(std::string("Error opening file :") + _imagesListFilepath);
  }

  size_t desiredByteIndex = imgIndex * _imageListFileLineLength;

  //size_t desiredByteIndex = 0ULL;
  
  // Every line has 90 bytes
  // Jump to correct line
  inFile.seekg(desiredByteIndex);

  std::string line;
  std::getline(inFile, line);


  // Extract file name
  std::stringstream ss(line);

  std::string columnData;

  // Throw away correct number of columns
  for (size_t i = 0ULL; i < _columnIndexFilename; ++i)
  {
    ss >> columnData;
  }
    
  // Get file name
  ss >> columnData;

  std::string result;

  if (relativePaths)
  {
    result = std::string(columnData);
  }
  else
  {
    result = std::string(_imagesPath + columnData);
  }


  // Return result filepath 
  return result;
}

int ImageRanker::GetRandomInteger(int from, int to) const
{
  // Create random generator
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<int> randFromDistribution(from,to);

  return randFromDistribution(rng);
}


std::vector<std::byte> ImageRanker::LoadFileToBuffer(std::string_view filepath) const
{
  // Open file for reading as binary
  std::ifstream ifs(filepath.data(), std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    throw std::runtime_error(std::string("Error opening file :") + filepath.data());
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
    throw std::runtime_error(std::string("Error reading file :") + filepath.data());
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
  else if (type == Database::cSecondary)
  {
    pDb = &_secondaryDb;
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
  Database* pDb{ nullptr };

  if (type == Database::cPrimary)
  {
    pDb = &_primaryDb;
  }
  else if (type == Database::cSecondary)
  {
    pDb = &_secondaryDb;
  }
  else
  {
    LOG_ERROR("NOT IMPLEMENTED!");
    return false;
  }

  // Fetch data from db
  std::string query{ "SELECT * FROM `images`;" };
  auto result = pDb->ResultQuery(query);

  // Iterate through all images
  for (auto&& row : result.second)
  {
    std::stringstream imageIdSs{ row[0] };
    size_t imageId;
    imageIdSs >> imageId;
      
    std::string filename{ row[1] };

    std::vector<std::pair<size_t, float>> probabilityVector;

    std::string queryProVec{ "SELECT `probability` FROM `probability_vectors` WHERE `image_id` = " + std::to_string(imageId) + " ORDER BY `vector_index`;" };
    auto resultProbVec = pDb->ResultQuery(queryProVec);

    // Construct probability vector
    size_t i = 0ULL;
    for (auto&& prob : resultProbVec.second)
    {
      std::stringstream probSs{ prob.front() };

      float probability;
      probSs >> probability;

      size_t i = 0ULL;
      probabilityVector.push_back(std::pair(i, probability));
      
      ++i;
    }

    // Sort probabilities
    std::sort(
      probabilityVector.begin(), probabilityVector.end(), 
      [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b)
    {
      return a.second > b.second;
    }
    );

    _images.insert(std::make_pair(imageId, Image{ imageId, std::move(filename), std::move(probabilityVector) }));

  }
  return false;
}


int32_t ImageRanker::ParseIntegerLE(const Buffer& buffer, size_t startIndex) const
{
  // Get pointer to starting array byte field
  const std::byte* offBuffer = &(buffer[startIndex]);

  // Initialize value
  int32_t signedInteger = 0;

  // Construct final BE integer
  signedInteger = static_cast<uint32_t>(offBuffer[3]) << 24 | static_cast<uint32_t>(offBuffer[2]) << 16 | static_cast<uint32_t>(offBuffer[1]) << 8 | static_cast<uint32_t>(offBuffer[0]);

  // Return parsed integer
  return signedInteger;
}


float ImageRanker::ParseFloatLE(const Buffer& buffer, size_t startIndex) const
{
  // Get pointer to starting array byte field
  const std::byte* offBuffer = &(buffer[startIndex]);

  // Initialize temp value
  uint32_t byteFloat = 0;

  // Get correct unsigned value of float data
  byteFloat = static_cast<uint32_t>(offBuffer[3]) << 24 | static_cast<uint32_t>(offBuffer[2]) << 16 | static_cast<uint32_t>(offBuffer[1]) << 8 | static_cast<uint32_t>(offBuffer[0]);

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