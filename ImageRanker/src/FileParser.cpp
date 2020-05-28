
#include "FileParser.h"

#include "ImageRanker.h"
#include "KeywordsContainer.h"

using namespace image_ranker;

std::vector<std::vector<float>> FileParser::parse_float_matrix(const std::string& filepath, size_t row_dim,
                                                               size_t begin_offset)
{
  // Open file for reading as binary from the end side
  std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    std::string msg{ "Error opening file '" + filepath + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
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
    std::string msg{ "File '" + filepath + "' is empty." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

  // Calculate byte length of each row (dim_N * sizeof(float))
  size_t row_byte_len = row_dim * sizeof(float);

  // Create line buffer
  std::vector<char> line_byte_buffer(row_byte_len);

  // Start reading at this offset
  ifs.ignore(begin_offset);

  // Declare result structure
  std::vector<std::vector<float>> result_features;

  // Read binary "lines" until EOF
  while (ifs.read(line_byte_buffer.data(), row_byte_len))
  {
    // Initialize vector of floats for this row
    std::vector<float> features_vector;
    features_vector.reserve(row_dim);

    size_t curr_offset{ 0 };

    // Iterate through all floats in a row
    for (size_t i{ 0ULL }; i < row_dim; ++i)
    {
      // ParseFloatLE(&lineBuffer[currOffset])
      float feature_value = ParseFloatLE(&line_byte_buffer[curr_offset]);

      // Push float value in
      features_vector.emplace_back(feature_value);

      curr_offset += sizeof(float);
    }

    // Insert this row into the result
    result_features.emplace_back(std::move(features_vector));
  }

  return result_features;
}

std::vector<float> FileParser::parse_float_vector(const std::string& filepath, size_t dim, size_t begin_offset)
{
  // Open file for reading as binary from the end side
  std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    std::string msg{ "Error opening file '" + filepath + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
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
    std::string msg{ "File '" + filepath + "' is empty." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

  // Calculate byte length of each row (dim_N * sizeof(float))
  size_t row_byte_len = dim * sizeof(float);

  // Create line buffer
  std::vector<char> line_byte_buffer(row_byte_len);

  // Start reading at this offset
  ifs.ignore(begin_offset);

  // Initialize vector of floats for this row
  std::vector<float> features_vector;
  features_vector.reserve(dim);

  // Read binary "lines" until EOF
  while (ifs.read((char*)line_byte_buffer.data(), row_byte_len))  // NOLINT
  {
    size_t curr_offset = 0;

    // Iterate through all floats in a row
    for (size_t i{ 0ULL }; i < dim; ++i)
    {
      float feature_value = ParseFloatLE(&line_byte_buffer[curr_offset]);

      // Push float value in
      features_vector.emplace_back(feature_value);

      curr_offset += sizeof(float);
    }

    // Read just one line
    break;
  }

  return features_vector;
}

std::map<std::string, size_t> FileParser::parse_w2vv_word_to_idx_file(const std::string& filepath)
{
  std::map<std::string, size_t> result_map;

  // Open file with list of files in  images dir
  std::ifstream ifs(filepath.data(), std::ios::in);

  // If failed to open file
  if (!ifs)
  {
    std::string msg{ "Error opening file '" + filepath + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

  std::string line_text_buffer;

  // While there is something to read
  while (std::getline(ifs, line_text_buffer))
  {
    if (line_text_buffer.empty())
    {
      continue;
    }

    std::stringstream line_buffer_ss(line_text_buffer);

    std::vector<std::string> tokens;

    // Tokenize line with ':' separator
    {
      std::string token;
      size_t i = 0;
      while (std::getline(line_buffer_ss, token, ':'))
      {
        tokens.push_back(token);

        ++i;
      }
    }

    std::string word(tokens[0]);

    std::stringstream idx_ss(tokens[1]);
    size_t idx;
    idx_ss >> idx;

    // Insert this record into table
    result_map.emplace(word, idx);
  }

  return result_map;
};

std::tuple<VideoId, ShotId, FrameNumber> FileParser::parse_video_filename_string(const std::string& filename,
                                                                                 const FrameFilenameOffsets& offsets)
{
  // Extract string representing video ID
  std::string videoIdString = filename.substr(offsets.v_ID_off, offsets.v_ID_len);

  // Extract string representing shot ID
  std::string shotIdString = filename.substr(offsets.s_ID_off, offsets.s_ID_len);

  // Extract string representing frame number
  std::string frameNumberString = filename.substr(offsets.fn_ID_off, offsets.fn_ID_len);

  return std::tuple(strTo<VideoId>(videoIdString), strTo<ShotId>(shotIdString), strTo<FrameNumber>(frameNumberString));
}

void FileParser::process_shot_stack(std::stack<SelFrame*>& videoFrames)
{
  size_t i{ 0_z };

  // Loop until stack is empty
  while (!videoFrames.empty())
  {
    // Get top image from stack
    auto pImg{ videoFrames.top() };
    videoFrames.pop();

    // Asign this number to this image
    pImg->m_num_successors = i;

    ++i;
  }
}

std::vector<ImageIdFilenameTuple> FileParser::get_image_filenames(const std::string& frame_to_ID_map_fpth)
{
  // Open file with list of files in images dir
  std::ifstream ifs(frame_to_ID_map_fpth, std::ios::in);

  // If failed to open file
  if (!ifs)
  {
    std::string msg{ "Error opening file '" + frame_to_ID_map_fpth + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

  std::vector<ImageIdFilenameTuple> result;

  std::string line;

  // While there are lines in file
  while (std::getline(ifs, line))
  {
    // Extract file name
    std::stringstream ss(line);

    // FILE FORMAT: filename   imageId
    FrameId frame_ID;
    std::string filename;

    ss >> filename;
    ss >> frame_ID;

    result.emplace_back(frame_ID, std::move(filename));
  }

  return result;
}

std::vector<SelFrame> FileParser::parse_image_metadata(const std::string& idToFilename,
                                                       const FrameFilenameOffsets& offsets,
                                                       [[maybe_unused]] size_t imageIdStride)
{
  std::vector<ImageIdFilenameTuple> imageIdFilenameTuples = get_image_filenames(idToFilename);

  // Create result variable
  std::vector<SelFrame> resultImages;
  resultImages.reserve(imageIdFilenameTuples.size());

  //
  // Create prefiled Image instances
  //
  for (auto&& [imageId, filename] : imageIdFilenameTuples)
  {
    // Parse filename
    auto [videoId, shotId, frameNumber] = parse_video_filename_string(filename, offsets);

    // Create new Image instance
    resultImages.emplace_back(SelFrame(FrameId(resultImages.size()), imageId, filename, videoId, shotId, frameNumber));
  }

  //
  // Add number of successors to each frame
  //

  // Initialize video ID counter
  size_t prevVideoId{ ERR_VAL<size_t>() };
  size_t prevShotId{ ERR_VAL<size_t>() };
  std::stack<SelFrame*> videoFrames;

  // Iterate over all images in ASC order by their IDs
  for (auto&& frame : resultImages)
  {
    //
    // Determine how many successors from the same video it has
    //

    // Get ID of current video
    size_t currVideoId = frame.m_video_ID;

    // Get ID of current shot
    size_t currShotId{ frame.m_shot_ID };

    // If this frame is from next video
    if (currShotId != prevShotId || currVideoId != prevVideoId)
    {
      // Process and label all frames from this video
      process_shot_stack(videoFrames);

      // Set new prev video ID
      prevShotId = currShotId;
      prevVideoId = currVideoId;
    }

    // Store this frame for future processing
    videoFrames.push(&frame);
  }

  return resultImages;
}

ViretKeywordClassesParsedData FileParser::parse_VIRET_format_keyword_classes_file(const std::string& filepath,
                                                                                  bool first_col_is_ID)
{
  // Open file with list of files in images dir
  std::ifstream inFile(filepath, std::ios::in);

  // If failed to open file
  if (!inFile)
  {
    std::string msg{ "Error opening file '" + filepath + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

  constexpr size_t idx_index{ 0 };
  constexpr size_t idx_wordnet_ID{ 1 };
  constexpr size_t idx_classname{ 2 };
  constexpr size_t idx_hyponyms{ 3 };
  constexpr size_t idx_hypernyms{ 4 };
  constexpr size_t idx_description{ 5 };

  // Declare return variables
  std::string _allDescriptions;
  std::map<size_t, Keyword*> _wordnetIdToKeywords;
  std::map<size_t, Keyword*> _vecIndexToKeyword;
  std::vector<std::pair<size_t, Keyword*>> _descIndexToKeyword;
  std::vector<std::unique_ptr<Keyword>> _keywords;
  std::map<KeywordId, Keyword*> ID_to_keyword;
  std::map<KeywordId, std::set<Keyword*>> ID_to_allkeywords;

  std::string lineBuffer;
  size_t iii{ 0 };
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
    std::stringstream vectIndSs(tokens[idx_index]);
    std::stringstream wordnetIdSs(tokens[idx_wordnet_ID]);

    size_t vectorIndex;
    size_t wordnetId;
    std::string indexClassname = tokens[idx_classname];

    // Get index that this description starts
    size_t descStartIndex = _allDescriptions.size();

    // Append description to all of them
    _allDescriptions.append(tokens[idx_description]);
    _allDescriptions.push_back('\0');

    // If pure hypernym
    if (tokens[idx_index] == "H")
    {
      vectorIndex = ERR_VAL<size_t>();
    }
    else
    {
      vectIndSs >> vectorIndex;
    }

    wordnetIdSs >> wordnetId;

    // Get all hyponyms
    std::vector<size_t> hyponyms;

    std::stringstream hyponymsSs(tokens[idx_hyponyms]);
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

    std::stringstream hyperymsSs(tokens[idx_hypernyms]);
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

    // Decide what is considered ID
    FrameId frame_ID{ FrameId(iii) };
    if (first_col_is_ID)
    {
      frame_ID = FrameId(vectorIndex);
    }

    std::set<Keyword*> syn_kws;
    // Insert all synonyms as well
    while (std::getline(classnames, finalWord, SYNONYM_DELIMITER_001))
    {
      std::string description{ *(_allDescriptions.begin() + descStartIndex) };

      // Insert this record into table
      _keywords.emplace_back(std::make_unique<Keyword>(FrameId(frame_ID), wordnetId, vectorIndex, std::move(finalWord),
                                                       descStartIndex, tokens[idx_hyponyms].size(), std::move(hyperyms),
                                                       std::move(hyponyms), std::move(description)));

      // Insert into desc -> Keyword
      _descIndexToKeyword.push_back(std::pair(descStartIndex, _keywords.back().get()));

      // Insert into wordnetId -> Keyword
      _wordnetIdToKeywords.insert(std::make_pair(wordnetId, _keywords.back().get()));

      // Insert into vector index -> Keyword
      _vecIndexToKeyword.insert(std::make_pair(vectorIndex, _keywords.back().get()));

      syn_kws.emplace(_keywords.back().get());
    }

    ID_to_allkeywords.insert(std::make_pair(frame_ID, std::move(syn_kws)));
    ID_to_keyword.insert(std::make_pair(frame_ID, _keywords.back().get()));
    ++iii;
  }

  return ViretKeywordClassesParsedData{
    std::move(_allDescriptions),    std::move(_vecIndexToKeyword), std::move(_wordnetIdToKeywords),
    std::move(_descIndexToKeyword), std::move(_keywords),          std::move(ID_to_keyword),
    std::move(ID_to_allkeywords)
  };
}

std::pair<Matrix<float>, DataParseStats> FileParser::parse_VIRET_format_frame_vector_file(
    const std::string& inputFilepath, size_t num_frames)
{
  DataParseStats stats{};

  // \todo Make dynamic
  std::vector<std::vector<float>> result{ num_frames };

  // Open file for reading as binary from the end side
  std::ifstream ifs(inputFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    std::string msg{ "Error opening file '" + inputFilepath + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
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
    std::string msg{ "File '" + inputFilepath + "' is empty." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

  // Create 4B buffer
  std::vector<char> buff_4B(sizeof(int32_t));

  // Discard first 36B of data
  ifs.ignore(VIRET_FORMAT_NET_DATA_HEADER_SIZE);

  // Read number of items in each vector per image
  ifs.read(buff_4B.data(), sizeof(int32_t));

  // If something happened
  if (!ifs)
  {
    std::string msg{ "Error reading file '" + inputFilepath + "'." };
    LOGE(msg);
    PROD_THROW("Error loading the program.");
  }

  // Parse number of present floats in every row
  int32_t numFloats = ParseIntegerLE(buff_4B);

  // Calculate byte length of each row
  size_t byteRowLengths = numFloats * sizeof(float) + sizeof(int32_t);

  // Create line buffer
  std::vector<char> lineBuffer(byteRowLengths);

  // Iterate until there is something to read from file
  while (ifs.read(lineBuffer.data(), byteRowLengths))
  {
    // Get picture ID of this row
    size_t id = ParseIntegerLE(lineBuffer);

    // Stride in bytes
    size_t currOffset = sizeof(float);

    // Initialize vector of floats for this row
    std::vector<float> rawRankData;

    // Reserve exact capacitys
    rawRankData.reserve(numFloats);

    float sum{ 0.0F };
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
    float varSum{ 0.0F };
    for (auto&& val : rawRankData)
    {
      float tmp{ val - mean };
      varSum += (tmp * tmp);
    }

    result[id] = std::move(rawRankData);
  }

  return std::pair(result, stats);
}

std::pair<Matrix<float>, DataParseStats> FileParser::parse_GoogleVision_format_frame_vector_file(
    const std::string& inputFilepath, size_t num_frames)
{
  DataParseStats stats{};

  // \todo Make dynamic
  std::vector<std::vector<float>> result(num_frames);

  // Open file for reading as binary from the end side
  std::ifstream ifs(inputFilepath, std::ios::binary | std::ios::ate);

  // If failed to open file
  if (!ifs)
  {
    LOGE("Error opening file: "s + inputFilepath);
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
    LOGE("Empty file opened!");
  }

  // Create 4B buffer
  std::vector<char> smallBuffer(sizeof(uint32_t));

  // If something happened
  if (!ifs)
  {
    LOGE("Error reading file: "s + inputFilepath);
  }

  // Read number of items in each vector per image
  ifs.read(smallBuffer.data(), sizeof(uint32_t));
  // Parse number of present floats in every row
  uint32_t numRecords = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

  // Read number of unique kws in annotation
  ifs.read(smallBuffer.data(), sizeof(uint32_t));
  uint32_t numKeywords = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

  float sum{ 0.0F };
  float min{ std::numeric_limits<float>::max() };
  float max{ -std::numeric_limits<float>::max() };

  std::vector<size_t> num_label_cnter;
  size_t num_labels_sum{ 0_z };

  std::vector<std::pair<size_t, size_t>> labels;

  for (size_t i{ 0_z }; i < numRecords; ++i)
  {
    std::vector<float> scoringData;
    scoringData.resize(numKeywords, ZERO_WEIGHT);

    ifs.read(smallBuffer.data(), sizeof(uint32_t));
    uint32_t ID = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

    ifs.read(smallBuffer.data(), sizeof(uint32_t));
    uint32_t numLabels = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

    auto cmp = [](const std::pair<size_t, float>& left, const std::pair<size_t, float>& right) {
      return left.second < right.second;
    };

    // Reserve enough space in container
    std::vector<std::pair<size_t, float>> container;

    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> maxHeap(
        cmp, std::move(container));

    labels.emplace_back(ID, numLabels);

    num_label_cnter.emplace_back(numLabels);
    num_labels_sum += numLabels;

    for (size_t iLabel{ 0_z }; iLabel < numLabels; ++iLabel)
    {
      ifs.read(smallBuffer.data(), sizeof(uint32_t));
      uint32_t kwId = static_cast<uint32_t>(ParseIntegerLE(smallBuffer.data()));

      ifs.read(smallBuffer.data(), sizeof(uint32_t));
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
    float varSum{ 0.0F };
    for (auto&& val : scoringData)
    {
      float tmp{ val - mean };
      varSum += (tmp * tmp);
    }

    result[ID] = std::move(scoringData);
  }

  // Get mean of how many labels was used
  std::sort(num_label_cnter.begin(), num_label_cnter.end());

  std::sort(labels.begin(), labels.end(),
            [](const std::pair<size_t, size_t>& l, const std::pair<size_t, size_t>& r) { return l.first < r.first; });

  auto median_num_classes{ float(num_label_cnter[num_label_cnter.size() / 2]) };
  float avg_num_classes{ float(num_labels_sum) / num_label_cnter.size() };

  stats.median_num_labels_asigned = median_num_classes;
  stats.avg_num_labels_asigned = avg_num_classes;

  return std::pair(result, stats);
}
