#include "FileParser.h"

#include "Image.hpp"
#include "KeywordsContainer.h"
#include "ImageRanker.h"

FileParser::FileParser(ImageRanker* pRanker):
  _pRanker(pRanker)
{
}


FileParser::~FileParser()
{
}


std::tuple<size_t, size_t, size_t> FileParser::ParseVideoFilename(const std::string& filename) const
{
  // Extract string representing video ID
  std::string videoIdString{ filename.substr(FILENAME_VIDEO_ID_FROM, FILENAME_VIDEO_ID_LEN) };

  // Extract string representing shot ID
  std::string shotIdString{ filename.substr(FILENAME_SHOT_ID_FROM, FILENAME_SHOT_ID_LEN) };

  // Extract string representing frame number
  std::string frameNumberString{ filename.substr(FILENAME_FRAME_NUMBER_FROM, FILENAME_FRAME_NUMBER_LEN) };


  return std::tuple(strToInt(videoIdString), strToInt(shotIdString), strToInt(frameNumberString));
}

size_t FileParser::GetVideoIdFromFrameFilename(const std::string& filename) const
{
  // Extract string representing video ID
  std::string videoIdString{ filename.substr(FILENAME_VIDEO_ID_FROM, FILENAME_VIDEO_ID_LEN) };

  // Return integral value of this string's representation
  return strToInt(videoIdString);
}

size_t FileParser::GetShotIdFromFrameFilename(const std::string& filename) const
{
  // Extract string representing video ID
  std::string videoIdString{ filename.substr(FILENAME_SHOT_ID_FROM, FILENAME_SHOT_ID_LEN) };

  // Return integral value of this string's representation
  return strToInt(videoIdString);
}


int32_t FileParser::ParseIntegerLE(const std::byte* pFirstByte) const
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


float FileParser::ParseFloatLE(const std::byte* pFirstByte) const
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

void FileParser::ProcessVideoShotsStack(std::stack<Image*>& videoFrames) const
{
  size_t i{ 0_z };

  // Loop until stack is empty
  while (!videoFrames.empty())
  {
    // Get top image from stack
    auto pImg{ videoFrames.top() };
    videoFrames.pop();

    // Asign this number to this image
    pImg->m_numSuccessorFrames = i;

    ++i;
  }
}

std::vector<ImageIdFilenameTuple> FileParser::GetImageFilenames(const std::string& _imageToIdMapFilepath) const
{
  // Open file with list of files in images dir
  std::ifstream inFile(_imageToIdMapFilepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR(std::string("Error opening file :") + _imageToIdMapFilepath);
  }

  std::vector<ImageIdFilenameTuple> result;

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

    result.emplace_back(imageId, std::move(filename));
  }

  return result;
}

std::vector<std::unique_ptr<Image>> FileParser::ParseImagesMetaData(
  const std::string& idToFilename, size_t imageIdStride
) const
{
  std::vector<ImageIdFilenameTuple> imageIdFilenameTuples{ GetImageFilenames(idToFilename) };

  // Create result variable
  std::vector<std::unique_ptr<Image>> resultImages;
  resultImages.reserve(imageIdFilenameTuples.size());

  //
  // Create prefiled Image instances
  //
  for (auto&&[imageId, filename] : imageIdFilenameTuples)
  {
    // Parse filename
    auto [videoId, shotId, frameNumber] = ParseVideoFilename(filename);

    // Create new Image instance
    resultImages.emplace_back(std::make_unique<Image>(imageId, resultImages.size(), filename, videoId, shotId, frameNumber));
  }


  //
  // Add number of successors to each frame
  //

  // Initialize video ID counter
  size_t prevVideoId{ SIZE_T_ERROR_VALUE };
  size_t prevShotId{ SIZE_T_ERROR_VALUE };
  std::stack<Image*> videoFrames;

  // Iterate over all images in ASC order by their IDs
  for (auto&& pImg : resultImages)
  {
    //
    // Determine how many successors from the same video it has
    //

    // Get ID of current video
    size_t currVideoId{ pImg->m_videoId };

#if USE_VIDEOS_AS_SHOTS

    // If this frame is from next video
    if (currVideoId != prevVideoId)

      // If this frame is from next video
      if (currVideoId != prevVideoId)
      {
        // Process and label all frames from this video
        ProcessVideoShotsStack(videoFrames);

        // Set new prev video ID
        prevVideoId = currVideoId;
      }

#else

    // Get ID of current shot
    size_t currShotId{ pImg->m_shotId };
    // If this frame is from next video
    if (currShotId != prevShotId || currVideoId != prevVideoId)
    {
      // Process and label all frames from this video
      ProcessVideoShotsStack(videoFrames);

      // Set new prev video ID
      prevShotId = currShotId;
      prevVideoId = currVideoId;
    }

#endif

    // Store this frame for future processing
    videoFrames.push(pImg.get());
  }
  
  return resultImages;
}

bool FileParser::ParseWordToVecFile(
  eKeywordsDataType kwType,
  std::vector<std::unique_ptr<Keyword>>& keywordsCont,
  const std::string& filepath
)
{
  if (filepath.empty())
    return true;

#if LOG_W2V_EXPANSION_KW_SETS

  std::cout << "=================================================" << std::endl;
  std::cout << "Parsing W2V file: " << filepath << std::endl;

#endif

  // Open file with list of files in images dir
  std::ifstream inFile(filepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR(std::string("Error opening file :") + filepath);
  }


  std::string lineBuffer;
  std::string lineBuffer2;

  size_t ii{ 0_z };

  // While there is something to read
  while (std::getline(inFile, lineBuffer))
  {
    ++ii;
    // Extract file name
    std::stringstream lineBufferStream(lineBuffer);


    std::string w;
    lineBufferStream >> w;

    if (ii == 5055)
    {
      ii += 300;
    }

    std::replace(w.begin(), w.end(), '_', ' ');

    // Find this image
    auto pKw = _pRanker->GetKeywordPtr(kwType, w);
    if (!pKw)
    {
      LOG_ERROR("Keyword not present in our dictionary.");
    }

    size_t expCount{ 0_z };

    // Parse inner lines until end of block
    while (std::getline(inFile, lineBuffer2) && lineBuffer2 != "---"s && lineBuffer2 != "--- N/A ---"s)
    {
      std::stringstream innerSs(lineBuffer2);

      std::string word;
      innerSs >> word;
      std::replace(word.begin(), word.end(), '_', ' ');

      float dist;
      innerSs >> dist;

      // Try if this word is in our dictionary
      auto pKwNew = _pRanker->GetKeywordPtr(kwType, word);
      if (!pKwNew)
        continue;


      // Add it as new possible expansion
      pKw->m_wordToVec.emplace_back(pKwNew, dist);
      ++expCount;
    }

  #if LOG_W2V_EXPANSION_KW_SETS

    std::cout << "---------------------------------------" << std::endl;
    std::cout << pKw->m_word << "< " << pKw->m_wordnetId << " > =>"<< std::endl;

    for (auto&& pWKw : pKw->m_wordToVec)
    {
      std::cout << "\t" << pWKw.first->m_word << "< " << pWKw.first->m_wordnetId << " > -> dist = "<< pWKw.second <<  std::endl;  
    }

  #endif
  }

  return true;
}

bool FileParser::LowMem_ParseRawScoringData_ViretFormat(
  std::vector<std::unique_ptr<Image>>& imagesCont,
  KwScoringDataId kwScDataId,
  const std::string& inputFilepath
  ) const
{
  // Open file for reading as binary from the end side
  std::ifstream ifs(inputFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file: "s + inputFilepath);
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
    LOG_ERROR("Error reading file: "s + inputFilepath);
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
    float varSum{ 0.0f };
    for (auto&& val : rawRankData)
    {
      float tmp{ val - mean };
      varSum += (tmp * tmp);
    }
    float variance = sqrtf((float)1 / (numFloats - 1) * varSum);

    Image* pImg{ imagesCont[_pRanker->MapIdToVectorIndex(id)].get() };

    // Push parsed data into the Image instance
    pImg->_rawImageScoringData.emplace(kwScDataId, std::move(rawRankData));

    // Push parsed data info into the Image instance
    pImg->_rawImageScoringDataInfo.emplace(kwScDataId, Image::ScoringDataInfo{ min, max, mean, variance });
  }

  return true;
}

bool FileParser::ParseRawScoringData_ViretFormat(
  std::vector<std::unique_ptr<Image>>& imagesCont,
  KwScoringDataId kwScDataId,
  const std::string& inputFilepath
) const
{
  // Open file for reading as binary from the end side
  std::ifstream ifs(inputFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file: "s + inputFilepath);
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
    LOG_ERROR("Error reading file: "s + inputFilepath);
  }

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(smallBuffer.data());

  // Calculate byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Where rows data start
  size_t currOffset = 40ULL;


  // Initialize video ID counter
  size_t prevVideoId{ SIZE_T_ERROR_VALUE };
  size_t prevShotId{ SIZE_T_ERROR_VALUE };
  std::stack<Image*> videoFrames;


  // Declare result vector
  std::map<size_t, std::unique_ptr<Image>> images;

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

    auto cmp = [](const std::pair<size_t, float>& left, const std::pair<size_t, float>& right)
    {
      return left.second < right.second;
    };

    // Initialize vector of floats for this row
    std::vector<float> rawRankData;

    // Reserve enough space in container
    std::vector<std::pair<size_t, float>> container;

    std::priority_queue<
      std::pair<size_t, float>,
      std::vector<std::pair<size_t, float>>,
      decltype(cmp)
    > maxHeap(cmp, std::move(container));


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

      maxHeap.push(std::pair(i, rankValue));

      // Stride in bytes
      currOffset += sizeof(float);
    }

    // Calculate mean value
    float mean{ sum / numFloats };

    // Calculate variance
    float varSum{ 0.0f };
    for (auto&& val : rawRankData)
    {
      float tmp{ val - mean };
      varSum += (tmp * tmp);
    }
    float variance = sqrtf((float)1 / (numFloats - 1) * varSum);

    Image* pImg{ imagesCont[_pRanker->MapIdToVectorIndex(id)].get() };
    std::vector<std::tuple<Keyword*, float>> topKeywords;
    topKeywords.reserve(NUM_TOP_KEYWORDS);

    for (size_t ii{ 0_z }; ii < NUM_TOP_KEYWORDS; ++ii)
    {
      if (maxHeap.size() <= 0)
        break;

      std::pair<size_t, float> pair{ maxHeap.top() };
      maxHeap.pop();

      auto pKw{ _pRanker->GetKeywordByVectorIndex(kwScDataId, pair.first) };

      topKeywords.emplace_back(pKw, pair.second);
    }

    // Push top keywpords
    pImg->_topKeywords.emplace(kwScDataId, std::move(topKeywords));

    // Push parsed data into the Image instance
    pImg->_rawImageScoringData.emplace(kwScDataId, std::move(rawRankData));

    // Push parsed data info into the Image instance
    pImg->_rawImageScoringDataInfo.emplace(kwScDataId, Image::ScoringDataInfo{ min, max, mean, variance });
  }

  return true;
}

bool FileParser::ParseSoftmaxBinFile_ViretFormat(
  std::vector<std::unique_ptr<Image>>& imagesCont,
  KwScoringDataId kwScDataId,
  const std::string& inputFilepath
) const
{
  // Open file for reading as binary from the end side
  std::ifstream ifs(inputFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file: "s + inputFilepath);
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
    LOG_ERROR("Error reading file: "s + inputFilepath);
  }

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(smallBuffer.data());

  // Calculate byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Where rows data start
  size_t currOffset = 40ULL;


  // Initialize video ID counter
  size_t prevVideoId{ SIZE_T_ERROR_VALUE };
  size_t prevShotId{ SIZE_T_ERROR_VALUE };
  std::stack<Image*> videoFrames;


  // Declare result vector
  std::map<size_t, std::unique_ptr<Image>> images;

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
    float varSum{ 0.0f };
    for (auto&& val : rawRankData)
    {
      float tmp{ val - mean };
      varSum += (tmp * tmp);
    }
    float variance = sqrtf((float)1 / (numFloats - 1) * varSum);

    // Get this image ptr
    Image* pImg{ imagesCont[_pRanker->MapIdToVectorIndex(id)].get() };

    // If no map exists for this kwScId create it
    if (pImg->_transformedImageScoringData.count(kwScDataId) == 0)
    {
      // Insert empty map there
      pImg->_transformedImageScoringData.emplace(kwScDataId, std::unordered_map<TransformFullId, std::vector<float>>{});
    }

    // Push parsed data into the Image instance
    // SUM based
    pImg->_transformedImageScoringData[kwScDataId].emplace(100, rawRankData);
    // MAX based
    pImg->_transformedImageScoringData[kwScDataId].emplace(110, std::move(rawRankData));
  }

  return true;
}

bool FileParser::ParseSoftmaxBinFile_GoogleAiVisionFormat(
  std::vector<std::unique_ptr<Image>>& imagesCont,
  KwScoringDataId kwScDataId,
  const std::string& inputFilepath
) const
{
  LOG_ERROR("Not implemented!"s);
  return false;
}

bool FileParser::ParseRawScoringData_GoogleAiVisionFormat(
  std::vector<std::unique_ptr<Image>>& imagesCont,
  KwScoringDataId kwScDataId,
  const std::string& inputFilepath
) const
{
  // Open file for reading as binary from the end side
  std::ifstream ifs(inputFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOG_ERROR("Error opening file: "s + inputFilepath);
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
  std::array<std::byte, sizeof(uint32_t)>  smallBuffer;

  // Discard first 36B of data
  //ifs.ignore(36ULL);

  

  // If something happened
  if (!ifs)
  {
    LOG_ERROR("Error reading file: "s + inputFilepath);
  }

  // Read number of items in each vector per image
  ifs.read((char*)smallBuffer.data(), sizeof(uint32_t));
  // Parse number of present floats in every row
  uint32_t numRecords = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

  // Read number of unique kws in annotation
  ifs.read((char*)smallBuffer.data(), sizeof(uint32_t));
  uint32_t numKeywords = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));


  float sum{ 0.0f };
  float min{ std::numeric_limits<float>::max() };
  float max{ -std::numeric_limits<float>::max() };

  for (size_t i{ 0_z }; i < numRecords; ++i)
  {
    std::vector<float> scoringData;
    scoringData.resize(numKeywords, GOOGLE_AI_NO_LABEL_SCORE);

    ifs.read((char*)smallBuffer.data(), sizeof(uint32_t));
    uint32_t imageId = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

    ifs.read((char*)smallBuffer.data(), sizeof(uint32_t));
    uint32_t numLabels = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

    auto cmp = [](const std::pair<size_t, float>& left, const std::pair<size_t, float>& right)
    {
      return left.second < right.second;
    };

    // Reserve enough space in container
    std::vector<std::pair<size_t, float>> container;

    std::priority_queue<
      std::pair<size_t, float>,
      std::vector<std::pair<size_t, float>>,
      decltype(cmp)
    > maxHeap(cmp, std::move(container));

    for (size_t iLabel{ 0_z }; iLabel < numLabels; ++iLabel)
    {
      ifs.read((char*)smallBuffer.data(), sizeof(uint32_t));
      uint32_t kwId = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

      ifs.read((char*)smallBuffer.data(), sizeof(uint32_t));
      float score = ParseFloatLE(smallBuffer.data());

      // Update this value
      scoringData[kwId] = score;

      // Update min
      if (score < min)
      {
        min = score;
      }
      // Update max
      if (score > max)
      {
        max = score;
      }

      // Add to sum
      sum += score;

      maxHeap.push(std::pair(kwId, score));
    }

    // Calculate mean value
    float mean{ sum / numLabels };

    // Calculate variance
    float varSum{ 0.0f };
    for (auto&& val : scoringData)
    {
      float tmp{ val - mean };
      varSum += (tmp * tmp);
    }
    float variance = sqrtf((float)1 / (numLabels - 1) * varSum);

    Image* pImg{ imagesCont[imageId].get() };

    std::vector<std::tuple<Keyword*, float>> topKeywords;
    topKeywords.reserve(NUM_TOP_KEYWORDS);

    for (size_t ii{ 0_z }; ii < NUM_TOP_KEYWORDS; ++ii)
    {
      if (maxHeap.size() <= 0)
        break;

      std::pair<size_t, float> pair{ maxHeap.top() };
      maxHeap.pop();

      auto pKw{ _pRanker->GetKeywordByVectorIndex(kwScDataId, pair.first) };

      topKeywords.emplace_back(pKw, pair.second);
    }

    // Push top keywpords
    pImg->_topKeywords.emplace(kwScDataId, std::move(topKeywords));

    // Push parsed data into the Image instance
    pImg->_rawImageScoringData.emplace(kwScDataId, std::move(scoringData));

    // Push parsed data info into the Image instance
    pImg->_rawImageScoringDataInfo.emplace(kwScDataId, Image::ScoringDataInfo{ min, max, mean, variance });
  }

  return true;
}

//
//bool FileParser::ParseSoftmaxBinFile_ViretFormat(
//  const std::string& inputFilepath,
//  const std::vector<std::string>& imageFilenames,
//  size_t imageIdStride
//) const
//{
//  if (_softmaxFilepath.empty())
//  {
//    return false;
//  }
//
//  // Open file for reading as binary from the end side
//  std::ifstream ifs(_softmaxFilepath, std::ios::binary | std::ios::ate);
//
//  // If failed to open file
//  if (!ifs)
//  {
//    LOG_ERROR("Error opening file: "s + _softmaxFilepath);
//    return false;
//  }
//
//  // Get end of file
//  auto end = ifs.tellg();
//
//  // Get iterator to begining
//  ifs.seekg(0, std::ios::beg);
//
//  // Compute size of file
//  auto size = std::size_t(end - ifs.tellg());
//
//  // If emtpy file
//  if (size == 0)
//  {
//    LOG_ERROR("Empty file opened!");
//    return false;
//  }
//
//
//  // Create 4B buffer
//  std::array<std::byte, sizeof(int32_t)>  smallBuffer;
//
//  // Discard first 36B of data
//  ifs.ignore(36ULL);
//
//  // Read number of items in each vector per image
//  ifs.read((char*)smallBuffer.data(), sizeof(int32_t));
//
//  // If something happened
//  if (!ifs)
//  {
//    LOG_ERROR("Error reading file: "s + _softmaxFilepath);
//    return false;
//  }
//
//  // Parse number of present floats in every row
//  int32_t numFloats = ParseIntegerLE(smallBuffer.data());
//
//  // Calculate byte length of each row
//  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);
//
//  // Where rows data start
//  size_t currOffset = 40ULL;
//
//
//
//  // Create line buffer
//  std::vector<std::byte>  lineBuffer;
//  lineBuffer.resize(byteRowLengths);
//
//  // Iterate until there is something to read from file
//  while (ifs.read((char*)lineBuffer.data(), byteRowLengths))
//  {
//    // Get picture ID of this row
//    size_t id = ParseIntegerLE(lineBuffer.data());
//
//    // Get this image
//    auto imageIt = _images.find(id);
//
//    // Stride in bytes
//    currOffset = sizeof(float);
//
//    std::vector<float> softmaxVector;
//    softmaxVector.reserve(numFloats);
//
//    // Iterate through all floats in row
//    for (size_t i = 0ULL; i < numFloats; ++i)
//    {
//      float rankValue{ ParseFloatLE(&lineBuffer[currOffset]) };
//
//      softmaxVector.emplace_back(rankValue);
//
//      // Stride in bytes
//      currOffset += sizeof(float);
//    }
//
//    // Store  vector of floats
//    auto&& [pair, result] {imageIt->second->m_aggVectors.insert(std::pair(static_cast<size_t>(InputDataTransformId::cSoftmax), std::move(softmaxVector)))};
//
//    // Recalculate all hypernyms in this vector
//    RecalculateHypernymsInVectorUsingSum(pair->second);
//  }
//
//  return true;
//}


std::tuple<
  std::string,
  std::map<size_t, Keyword*>,
  std::map<size_t, Keyword*>,
  std::vector<std::pair<size_t, Keyword*>>,
  std::vector<std::unique_ptr<Keyword>>> 
FileParser::ParseKeywordClassesFile_ViretFormat(const std::string& filepath) const
{
  // Open file with list of files in images dir
  std::ifstream inFile(filepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    LOG_ERROR(std::string("Error opening file :") + filepath);
  }

  // Declare return variables
  std::string _allDescriptions;
  std::map<size_t, Keyword*> _wordnetIdToKeywords;
  std::map<size_t, Keyword*> _vecIndexToKeyword;
  std::vector<std::pair<size_t, Keyword*>> _descIndexToKeyword;
  std::vector<std::unique_ptr<Keyword>> _keywords;

  std::string lineBuffer;

  // While there is something to read
  while (std::getline(inFile, lineBuffer))
  {

    // Extract file name
    std::stringstream lineBufferStream(lineBuffer);

    std::vector<std::string> tokens;
    std::string token;
    size_t i = 0ULL;

    while (std::getline(lineBufferStream, token, CSV_DELIMITER_001))
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

    // If pure hypernym
    if (tokens[0] == "H")
    {
      vectorIndex = SIZE_T_ERROR_VALUE;
    }
    else
    {
      vectIndSs >> vectorIndex;
    }

    wordnetIdSs >> wordnetId;


    // Get all hyponyms
    std::vector<size_t> hyponyms;

    std::stringstream hyponymsSs(tokens[3]);
    std::string stringHyponym;

    while (std::getline(hyponymsSs, stringHyponym, SYNONYM_DELIMITER_001))
    {
      std::stringstream hyponymIdSs(stringHyponym);
      size_t hyponymId;

      hyponymIdSs >> hyponymId;

      hyponyms.push_back(hyponymId);
    }

    // Get all hyperyms
    std::vector<size_t> hyperyms;

    std::stringstream hyperymsSs(tokens[4]);
    std::string stringHypernym;

    while (std::getline(hyperymsSs, stringHypernym, SYNONYM_DELIMITER_001))
    {
      std::stringstream hyperymIdSs(stringHypernym);
      size_t hyperymId;

      hyperymIdSs >> hyperymId;

      hyperyms.push_back(hyperymId);
    }


    // Create sstream from concatenated string of synonyms
    std::stringstream classnames(indexClassname);
    std::string finalWord;


    // Insert all synonyms as well
    while (std::getline(classnames, finalWord, SYNONYM_DELIMITER_001))
    {
      // Insert this record into table
      _keywords.emplace_back(std::make_unique<Keyword>(wordnetId, vectorIndex, std::move(finalWord), descStartIndex, tokens[3].size(), std::move(hyperyms), std::move(hyponyms)));

      // Insert into desc -> Keyword
      _descIndexToKeyword.push_back(std::pair(descStartIndex, _keywords.back().get()));

      // Insert into wordnetId -> Keyword
      _wordnetIdToKeywords.insert(std::make_pair(wordnetId, _keywords.back().get()));

      // Insert into vector index -> Keyword
      _vecIndexToKeyword.insert(std::make_pair(vectorIndex, _keywords.back().get()));
    }
  }
  
  return std::tuple{
    std::move(_allDescriptions),
    std::move(_vecIndexToKeyword),
    std::move(_wordnetIdToKeywords),
    std::move(_descIndexToKeyword),
    std::move(_keywords)
  };
}

std::tuple<
  std::string,
  std::map<size_t, Keyword*>,
  std::map<size_t, Keyword*>,
  std::vector<std::pair<size_t, Keyword*>>,
  std::vector<std::unique_ptr<Keyword>>>
  FileParser::ParseKeywordClassesFile_GoogleAiVisionFormat(const std::string& filepath) const
{
  return ParseKeywordClassesFile_ViretFormat(filepath);
}