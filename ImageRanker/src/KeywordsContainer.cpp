

#include <algorithm>

#include "KeywordsContainer.h"


KeywordsContainer::KeywordsContainer(std::string_view keywordClassesFilepath)
{
  // Parse data
  ParseKeywordClassesFile(keywordClassesFilepath);

  // Sort keywords
  std::sort(_keywords.begin(), _keywords.end(), keywordLessThan);

  // \todo Replace
  _descIndexToKeyword.reserve(20000);


  _approxDescLen = _allDescriptions.size() / _descIndexToKeyword.size();
}



KeywordsContainer::KeywordsContainer(std::vector< std::vector<std::string>>&& data)
{
  // Parse data
  ParseKeywordDbDataStructure(std::move(data));

  // Sort keywords
  std::sort(_keywords.begin(), _keywords.end(), keywordLessThan);
}



std::string KeywordsContainer::GetKeywordByVectorIndex(size_t index) const
{
  auto result = _vecIndexToKeyword.find(index);

  if (result == _vecIndexToKeyword.end())
  {
    return std::string();
  }

  return result->second->m_word;
}

bool KeywordsContainer::ParseKeywordDbDataStructure(std::vector< std::vector<std::string>>&& data)
{
  /*
  ROW DATA format:
    0 => Wordnet ID
    1 => Vector index
    2 => Word
    3 => Description
    4 => Hypernyms
    5 => Hyponyms
  */

  std::string lineBuffer;

  // While there is something to read
  for (auto&& row : data)
  {
    std::stringstream wordnetIdSs(row[0]);
    size_t wordnetId;
    wordnetIdSs >> wordnetId;

    std::stringstream vectIndSs(row[1]);
    std::string word = row[2];
    size_t vectorIndex;

    if (row[1].empty())
    {
      vectorIndex = 0ULL;
    }
    else
    {
      vectIndSs >> vectorIndex;
    }

    // Get index that this description starts
    size_t descStartIndex = _allDescriptions.size();
    size_t descEndIndex = descStartIndex + row[3].size() - 1ULL;

    std::stringstream hypernymsSs(row[4]);
    std::vector<size_t> hypernyms;

    // Extract hypernyms
    std::string hypernymIdString;
    while (std::getline(hypernymsSs, hypernymIdString, ';'))
    {
      size_t hypernymId;
      std::stringstream hypernymIdStringSs{ hypernymIdString };
      hypernymIdStringSs >> hypernymId;

      hypernyms.push_back(hypernymId);
    }

    std::stringstream hyponymsSs(row[5]);
    std::vector<size_t> hyponyms;

    // Extract hyponyms
    std::string hyponymIdString;
    while (std::getline(hyponymsSs, hyponymIdString, ';'))
    {
      size_t hyponymId;
      std::stringstream hyponymIdStringSs{ hyponymIdString };
      hyponymIdStringSs >> hyponymId;

      hyponyms.push_back(hyponymId);
    }

    // Append description to all of them
    _allDescriptions.append(row[3]);
    _allDescriptions.push_back('\0');


    // Insert this record into table
    _keywords.emplace_back(std::make_unique<Keyword>(wordnetId, vectorIndex, std::move(word), descStartIndex, row[3].size(), std::move(hypernyms), std::move(hyponyms)));

    // Insert into wordnetId -> Keyword
    _wordnetIdToKeywords.insert(std::make_pair(wordnetId, _keywords.back().get()));

  }
  return true;
}

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
    
    while (std::getline(hyponymsSs, stringHyponym, SYNONYM_DELIMITER))
    {
      std::stringstream hyponymIdSs(stringHyponym);
      size_t hyponymId;

      hyponymIdSs >> hyponymId;

      hyponyms.push_back(hyponymId);
    }

    // Get all hyperyms
    std::vector<size_t> hyperyms;

    std::stringstream hyperymsSs(tokens[3]);
    std::string stringHypernym;

    while (std::getline(hyperymsSs, stringHypernym, SYNONYM_DELIMITER))
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
    while (std::getline(classnames, finalWord, SYNONYM_DELIMITER))
    {
    #if PUSH_DATA_TO_DB

      // Insert this word representation into all words
      _words.insert(finalWord);

      // Create WordnetID -> word pairs
      _keywordToWord.push_back(std::make_pair(wordnetId, finalWord));

    #endif


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
  return true;
}


std::vector<size_t> KeywordsContainer::FindAllNeedles(std::string_view hey, std::string_view needle)
{
  // Step 0. Should not be empty heying
  if (hey.size() == 0 || needle.size() == 0)
  {
    return std::vector<size_t>();
  }

  std::vector<size_t> resultIndices;

  // Step 1. Compute failure function
  std::vector<int> failure(needle.size(), -1 );

  for(int r = 1, l = -1; r < needle.size(); ++r) {

    while( l != -1 && needle[l+1] != needle[r])
      l = failure[l];

    // assert( l == -1 || needle[l+1] == needle[r]);
    if( needle[l+1] == needle[r])
      failure[r] = ++l;
  }

  // Step 2. Search needle
  int tail = -1;
  for(int i=0; i<hey.size(); i++) {

    while( tail != -1 && hey[i] != needle[tail+1])
      tail = failure[tail];

    if( hey[i] == needle[tail+1])
      tail++;

    if (tail == needle.size() - 1)
    {
      resultIndices.push_back(i - tail);
      tail = -1;
    }

    // Gather maximum of needles
    if (resultIndices.size() >= NUM_SUGESTIONS)
    {
      return resultIndices;
    }
  }

  return resultIndices;
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

  std::vector< std::tuple<size_t, std::string, std::string> > postResultWordnetIds;


  // Get desired number of results
  for (size_t j = 0ULL; j < NUM_SUGESTIONS; ++j)
  {
    Keyword* pKeyword{_keywords[left + j].get()};

    // Check if prefix is equal to searched word

    // Force lowercase
    std::locale loc;
    std::string lowerWord;
    std::string lowerPrefix;

    for (auto elem : pKeyword->m_word)
    {
      lowerWord.push_back(std::tolower(elem, loc));
    }

    for (auto elem : prefix)
    {
      lowerPrefix.push_back(std::tolower(elem, loc));
    }

    auto res = std::mismatch(lowerPrefix.begin(), lowerPrefix.end(), lowerWord.begin());

    if (res.first == lowerPrefix.end())
    {
      resultWordnetIds.push_back(std::make_tuple(pKeyword->m_wordnetId, pKeyword->m_word, GetKeywordDescriptionByWordnetId(pKeyword->m_wordnetId)));
    }
    else
    {
      postResultWordnetIds.push_back(std::make_tuple(pKeyword->m_wordnetId, pKeyword->m_word, GetKeywordDescriptionByWordnetId(pKeyword->m_wordnetId)));
    }
  }

  // If we need to add up desc search results
  if (resultWordnetIds.size() < NUM_SUGESTIONS && prefix.size() >= MIN_DESC_SEARCH_LENGTH)
  {
    std::vector<size_t> needleIndices = FindAllNeedles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      Keyword* pKeyword = MapDescIndexToKeyword(index);
      
      resultWordnetIds.push_back(std::make_tuple(pKeyword->m_wordnetId, pKeyword->m_word, GetKeywordDescriptionByWordnetId(pKeyword->m_wordnetId)));
    }
  }

  size_t j = 0ULL;
  while (resultWordnetIds.size() < NUM_SUGESTIONS)
  {
    resultWordnetIds.push_back(postResultWordnetIds[j]);

    ++j;
  }

  return resultWordnetIds;
}


Keyword* KeywordsContainer::MapDescIndexToKeyword(size_t descIndex) const
{

  size_t left = 0ULL;
  size_t right = _descIndexToKeyword.size();

  size_t i = (right + left) / 2;

  while (true)
  {
    // Test if middle one is less than
    bool pivotLess = descIndex < _descIndexToKeyword[i].first;

    if (pivotLess)
    {
      right = i - 1;
    }
    else
    {
      left = i;
    }

    if (right - left <= 1)
    {
      break;
    }

    i = (right + left) / 2;

  }

  return _descIndexToKeyword[left].second;
}

std::string KeywordsContainer::GetKeywordByWordnetId(size_t wordnetId)
{
  auto resultIt = _wordnetIdToKeywords.find(wordnetId);

  if (resultIt == _wordnetIdToKeywords.end())
  {
    std::string("NOT FOUND");
  }

  return resultIt->second->m_word;
}

#if PUSH_DATA_TO_DB

bool KeywordsContainer::PushKeywordsToDatabase(Database& db)
{
  /*===========================
    Push into `words`table
    ===========================*/
  {
    // Start query
    std::string query{ "INSERT IGNORE INTO words (`word`) VALUES " };

    // Words first
    for (auto&& word : _words)
    {
      query.append("('");
      query.append(db.EscapeString(word));
      query.append("'),");
    }

    // Delete last comma
    query.pop_back();
    // Add semicolon
    query.append(";");

    // Send query
    db.NoResultQuery(query);
  }

  /*===========================
    Push into `keywords`table
    ===========================*/
  {
    // Start query
    std::string query{ "INSERT IGNORE INTO keywords (`wordnet_id`, `vector_index`, `description`) VALUES" };

    // Keywords then
    for (auto&& pKeyword : _keywords)
    {
      std::string desctiption{ db.EscapeString(GetKeywordDescriptionByWordnetId(pKeyword->m_wordnetId)) };

      query.append("( ");
      query.append(std::to_string(pKeyword->m_wordnetId));
      query.append(", ");


      if (pKeyword->m_vectorIndex == SIZE_T_ERROR_VALUE)
      {
        query.append("NULL");
      }
      else
      {
        query.append(std::to_string(pKeyword->m_vectorIndex));
      }

      if (pKeyword->m_vectorIndex == 0ULL)
      {
        std::cout << "aa" << std::endl;
      }
      
      query.append(", '");
      query.append(desctiption);
      query.append("'),");
    }

    // Delete last comma
    query.pop_back();
    // Add semicolon
    query.append(";");

    // Send query
    db.NoResultQuery(query);
  }

  /*===========================
    Push into `keywords_words`table
    ===========================*/
  {
    // Start query
    std::string query{ "INSERT IGNORE INTO `keyword_word` (`keyword_id`, `word_id`) VALUES" };

    // Keywords then
    for (auto&& pKeyword : _keywordToWord)
    {
      std::string word{ pKeyword.second };

      auto aa = db.ResultQuery("SELECT `id` FROM `words` WHERE `word` LIKE '" + db.EscapeString(word) + "'");

      if (aa.second.empty())
      {
        continue;
      }

      std::stringstream sstream(aa.second.front().front());

      size_t word_id;
      sstream >> word_id;

      query.append("( ");
      query.append(std::to_string(pKeyword.first));
      query.append(", ");
      query.append(std::to_string(word_id));
      query.append("),");
    }

    // Delete last comma
    query.pop_back();
    // Add semicolon
    query.append(";");

    // Send query
    db.NoResultQuery(query);

  }
  //! \todo Populate tables - keywords_hyponyms & keywords_hypenyms

  return false;
}

#endif


std::string KeywordsContainer::GetKeywordDescriptionByWordnetId(size_t wordnetId)
{

  auto resultIt = _wordnetIdToKeywords.find(wordnetId);

  if (resultIt == _wordnetIdToKeywords.end())
  {
    std::string("NOT FOUND");
  }

  size_t startDescIndex = resultIt->second->m_descStartIndex;

  char* pDesc = (_allDescriptions.data()) + startDescIndex;


  return std::string(pDesc);
}
