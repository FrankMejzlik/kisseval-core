#include "ImageRanker.h"



bool KeywordsContainer::ParseKeywordClassesFile(std::string_view filepath)
{
  // Open file with list of files in images dir
  std::ifstream inFile(filepath.data(), std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    throw std::runtime_error(std::string("Error opening file :") + filepath.data());
  }

  std::string lineBuffer;

  // While there is something to read
  while (std::getline(inFile, lineBuffer))
  {

    // Extract file name
    std::stringstream lineBufferStream(lineBuffer);

    std::vector<std::string> tokens;
    std::string token;
    size_t i = 0ULL;

    while (std::getline(lineBufferStream, token, CSV_DELIMITER))
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

    
    // Get index that this description starts
    size_t descStartIndex = _allDescriptions.size();
    size_t descEndIndex = descStartIndex + tokens[5].size() - 1ULL;

    // Append description to all of them
    _allDescriptions.append(tokens[5]);
    _allDescriptions.push_back('\0');
    
    vectIndSs >> vectorIndex;
    wordnetIdSs >> wordnetId;

    

    // Create sstream from concatenated string of synonyms
    std::stringstream classnames(indexClassname);
    std::string finalWord;


    // Insert all synonyms as well
    while (std::getline(classnames, finalWord, SYNONYM_DELIMITER))
    {
      // Insert this record into table
      _keywords.emplace_back(std::make_unique<Keyword>(wordnetId, vectorIndex, std::move(finalWord), descStartIndex, tokens[3].size()));  

      // Insert into wordnetId -> Keyword
      _wordnetIdToKeywords.insert(std::make_pair(wordnetId, _keywords.back().get()));
    }
  }
  return true;
}


std::vector< std::tuple<size_t, std::string, std::string> > KeywordsContainer::GetNearKeywords(const std::string& prefix)
{
  KeywordsContainer::KeywordLessThanStringComparator comparator;
  size_t left = 0ULL;
  size_t right = _keywords.size() - 1ULL;

  size_t i = right / 2;

  while (true)
  {
    // Test if middle one is less than
    bool leftIsLess = comparator(_keywords[i]->m_word, prefix);

    if (leftIsLess)
    {
      left = i + 1;
    }
    else
    {
      right = i;
    }

    if (right - left < 1)
    {
      break;
    }

    i = (right + left) / 2;

  }
  
  std::vector< std::tuple<size_t, std::string, std::string> > resultWordnetIds;
  resultWordnetIds.reserve(NUM_SUGESTIONS);

  // Get desired number of results
  for (size_t j = 0ULL; j < NUM_SUGESTIONS; ++j)
  {
    resultWordnetIds.push_back(std::make_tuple(_keywords[left + j]->m_wordnetId, _keywords[left + j]->m_word, GetKeywordDescriptionByWordnetId(_keywords[left + j]->m_wordnetId)));
  }
  
  return resultWordnetIds;
}



ImageRanker::ImageRanker(
  std::string_view imagesPath,
  std::string_view probabilityVectorFilepath,
  std::string_view deepFeaturesFilepath,
  std::string_view keywordClassesFilepath
):
  _keywords(keywordClassesFilepath)
{}

size_t ImageRanker::GetRandomImageId() const
{ 
  // Get random index
  return static_cast<size_t>(GetRandomInteger(0, NUM_ROWS) * INDEX_OFFSET);
}




std::vector< std::pair< size_t, std::vector<float> > > ImageRanker::ParseSoftmaxBinFile(std::string_view filepath) const
{
  // Create buffer from file
  Buffer buffer = LoadFileToBuffer(filepath);

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(buffer, 36ULL);

  // Byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Where rows data start
  size_t currOffset = 40ULL;

  // Declare vector of pairs
  std::vector< std::pair< size_t, std::vector<float> > > rows;
  // Reserve space for it to aproximate number of wors
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
    std::vector<float> floats;
    // Reserve exact capacitys
    floats.reserve(numFloats);

    // Iterate through all floats in row
    for (size_t i = 0; i < numFloats; ++i)
    {
      // Push float value in
      floats.push_back(ParseFloatLE(buffer, currOffset));

      // Stride in bytes
      currOffset += sizeof(float);
    }

    // Push final row
    rows.push_back(std::make_pair(id, floats));
  }

  return rows;
}

std::vector< std::pair< size_t, std::unordered_map<size_t, float> > > ImageRanker::ParseSoftmaxBinFileFiltered(std::string_view filepath, float minProbabilty) const 
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

std::string ImageRanker::GetImageFilepathByIndex(size_t imgIndex, bool relativePaths) const
{
  constexpr const char fileFilepath[] = CONCATENATE_DEFINES(DATA_PATH, IMAGES_LIST_FILENAME);

  // Open file with list of files in images dir
  std::ifstream inFile(fileFilepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    throw std::runtime_error(std::string("Error opening file :") + fileFilepath);
  }

  size_t desiredByteIndex = imgIndex * 90ULL;

  // Every line has 90 bytes
  // Jump to correct line
  inFile.seekg(desiredByteIndex);

  std::string line;
  std::getline(inFile, line);


  // Extract file name
  std::stringstream ss(line);

  std::string columnData;
  // Throw away 4 columns
  ss >> columnData; ss >> columnData; ss >> columnData; ss >> columnData;
  // Get file name
  ss >> columnData;

  std::string result;

  if (relativePaths)
  {
    result = std::string(columnData);
  }
  else
  {
    result = std::string(IMAGES_PATH + columnData);
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