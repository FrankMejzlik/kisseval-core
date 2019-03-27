
#include "KeywordsContainer.h"


KeywordsContainer::KeywordsContainer(std::string_view keywordClassesFilepath)
{
  // Parse data
  ParseKeywordClassesFile(keywordClassesFilepath);

  // Sort keywords
  std::sort(_keywords.begin(), _keywords.end(), keywordLessThan);
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
      vectorIndex = 0ULL;
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
    #if !USE_DATA_FROM_DATABASE
      // Insert this word representation into all words
      _words.insert(finalWord);
    #endif

      // Insert this record into table
      _keywords.emplace_back(std::make_unique<Keyword>(wordnetId, vectorIndex, std::move(finalWord), descStartIndex, tokens[3].size(), std::move(hyperyms), std::move(hyponyms)));

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
      query.append(std::to_string(pKeyword->m_vectorIndex));
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

  //! \todo Populate tables - keywords_hyponyms & keywords_hypenyms

  return false;
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
