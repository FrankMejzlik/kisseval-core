

#include "KeywordsContainer.h"

#include <algorithm>
#include <cctype>

#include "ImageRanker.h"

KeywordsContainer::KeywordsContainer(ImageRanker* pRanker, eKeywordsDataType type, const std::string& keywordClassesFilepath) :
  _pRanker(pRanker),
  _pDataType(type),
  _keywordsFilepath(keywordClassesFilepath)
{}

bool KeywordsContainer::Initialize()
{
  std::tuple<
    std::string,
    std::map<size_t, Keyword*>,
    std::map<size_t, Keyword*>,
    std::vector<std::pair<size_t, Keyword*>>,
    std::vector<std::unique_ptr<Keyword>>> res;

  switch (_pDataType)
  {
  case eKeywordsDataType::cViret1:

    // Parse data
    res = _pRanker->GetFileParser()->ParseKeywordClassesFile_ViretFormat(_keywordsFilepath);

    break;

  case eKeywordsDataType::cGoogleAI:
    // Parse data
    res = _pRanker->GetFileParser()->ParseKeywordClassesFile_GoogleAiVisionFormat(_keywordsFilepath);

    break;

  }

  // Store results
  _allDescriptions = std::move(std::get<0>(res));
  _vecIndexToKeyword = std::move(std::get<1>(res));
  _wordnetIdToKeywords = std::move(std::get<2>(res));
  _descIndexToKeyword = std::move(std::get<3>(res));
  _keywords = std::move(std::get<4>(res));

  // Sort keywords
  std::sort(_keywords.begin(), _keywords.end(), keywordLessThan);

  // Iterate over all unique keywords 
  for (auto&&[wordnetId, pKw] : _wordnetIdToKeywords)
  {
#if LOG_DEBUG_HYPERNYMS_EXPANSION
    LOG_NO_ENDL(std::to_string(wordnetId) + "; "s + pKw->m_word + ";");
#endif

    GetVectorKeywordsIndicesSet(pKw->m_hyponymBinIndices, wordnetId);

#if LOG_DEBUG_HYPERNYMS_EXPANSION
    for (auto&& index : pKw->m_hyponymBinIndices)
    {
      auto kw = GetKeywordConstPtrByVectorIndex(index);

      LOG_NO_ENDL(std::to_string(kw->m_vectorIndex) + ",");
    }
    LOG("");
#endif

  }

  return true;
}


KeywordData KeywordsContainer::GetKeywordByVectorIndex(size_t index) const
{
  auto result = _vecIndexToKeyword.find(index);

  if (result == _vecIndexToKeyword.end())
  {
    return KeywordData();
  }

  return KeywordData(result->second->m_wordnetId, result->second->m_word, GetKeywordDescriptionByWordnetId(result->second->m_wordnetId));
}

Keyword* KeywordsContainer::GetKeywordPtrByVectorIndex(size_t index) const
{
  auto result = _vecIndexToKeyword.find(index);

  if (result == _vecIndexToKeyword.end())
  {
    return nullptr;
  }

  return result->second;
}

const Keyword* KeywordsContainer::GetKeywordConstPtrByVectorIndex(size_t index) const
{
  auto result = _vecIndexToKeyword.find(index);

  if (result == _vecIndexToKeyword.end())
  {
    return nullptr;
  }

  return result->second;
}


std::vector<size_t> KeywordsContainer::GetVectorKeywords(size_t wordnetId) const
{
  // Find root keyword
  auto wordnetIdKeywordPair = _wordnetIdToKeywords.find(wordnetId);

  if (wordnetIdKeywordPair == _wordnetIdToKeywords.end()) 
  {
    LOG_ERROR("Keyword not found!");

    return std::vector<size_t>();
  }

  Keyword* pRootKeyword = wordnetIdKeywordPair->second;

  // It not vector keyword
  if (pRootKeyword->m_vectorIndex == SIZE_T_ERROR_VALUE)
  {
    std::vector<size_t> result;

    // Recursively get hyponyms
    for (auto&& hypo : pRootKeyword->m_hyponyms) 
    {
      auto recur = GetVectorKeywords(hypo);
      result.reserve(result.size() + recur.size());
      result.insert(result.end(),  recur.begin(), recur.end());
    }
    
    return result;
  }

  // If vector word, return self
  return std::vector<size_t>{ pRootKeyword->m_wordnetId };
}

std::vector<size_t> KeywordsContainer::GetVectorKeywordsIndices(size_t wordnetId) const
{
  // Find root keyword
  auto wordnetIdKeywordPair = _wordnetIdToKeywords.find(wordnetId);

  if (wordnetIdKeywordPair == _wordnetIdToKeywords.end())
  {
    LOG_ERROR("Keyword not found!");

    return std::vector<size_t>();
  }

  Keyword* pRootKeyword = wordnetIdKeywordPair->second;

  // It not vector keyword
  if (pRootKeyword->m_vectorIndex == SIZE_T_ERROR_VALUE)
  {
    std::vector<size_t> result;

    // Recursively get hyponyms
    for (auto&& hypo : pRootKeyword->m_hyponyms)
    {
      auto recur = GetVectorKeywordsIndices(hypo);
      result.reserve(result.size() + recur.size());
      result.insert(result.end(), recur.begin(), recur.end());
    }

    return result;
  }
 

  // If vector word, return self
  return std::vector<size_t>{ pRootKeyword->m_vectorIndex };
}

const Keyword* KeywordsContainer::GetKeywordConstPtrByWordnetId(size_t wordnetId) const
{
  // Find root keyword
  auto pairIt = _wordnetIdToKeywords.find(wordnetId);

  // If this key was not found
  if (pairIt  == _wordnetIdToKeywords.end())
  {
    LOG_ERROR("Keyword with ID "s + std::to_string(wordnetId) + " not found!");
    return nullptr;
  }

  return pairIt->second;
}


Keyword* KeywordsContainer::GetKeywordPtrByWordnetId(size_t wordnetId) const
{
  // Find root keyword
  auto pairIt = _wordnetIdToKeywords.find(wordnetId);

  // If this key was not found
  if (pairIt == _wordnetIdToKeywords.end())
  {
    LOG_ERROR("Keyword with ID "s + std::to_string(wordnetId) + " not found!");
    return nullptr;
  }

  return pairIt->second;
}

void KeywordsContainer::GetVectorKeywordsIndicesSet(std::unordered_set<size_t>& destIndicesSetRef, size_t wordnetId) const
{
  // Get this Keyword
  Keyword* pRootKw = GetKeywordPtrByWordnetId(wordnetId);

  // If this hypernyms has spot in data vector
  if (!pRootKw->IsInBinVector())
  {
    // Add it to set as well
    destIndicesSetRef.emplace(pRootKw->m_vectorIndex);
  }

   // If is hypernym
  if (pRootKw->IsHypernym())
  {
    // Recursively get hyponyms into provided set
    for (auto&& hypo : pRootKw->m_hyponyms)
    {
      GetVectorKeywordsIndicesSet(destIndicesSetRef, hypo);
    }
  }
}

void KeywordsContainer::GetVectorKeywordsIndicesSetShallow(
  std::unordered_set<size_t>& destIndicesSetRef, size_t wordnetId,
  bool skipConstructedHypernyms
) const
{
  // Get this Keyword
  Keyword* pRootKw = GetKeywordPtrByWordnetId(wordnetId);

  // If this hypernyms has spot in data vector
  if (!pRootKw->IsInBinVector())
  {
    // Add it to set as well
    destIndicesSetRef.emplace(pRootKw->m_vectorIndex);
  }
  else
  {
    // If we want to include subsets of constructed hypernyms
    if (!skipConstructedHypernyms)
    {
      // Recursively get hyponyms into provided set
      for (auto&& hypo : pRootKw->m_hyponyms)
      {
        GetVectorKeywordsIndicesSetShallow(destIndicesSetRef, hypo);
      }
    }
  }
}


CnfFormula KeywordsContainer::GetCanonicalQuery(const std::string& query, bool skipConstructedHypernyms) const
{
  // \todo implement properly
  //skipConstructedHypernyms = IGNORE_CONSTRUCTED_HYPERNYMS;

  //EG: &-8252602+-8256735+-3206282+-4296562+

  std::stringstream idSs;
  size_t wordnetId;
  bool nextIdNegate{false};

  std::vector<Clause> resultFormula;

  // Parse query
  // \todo write complete parser
  for (auto&& c : query)
  {
    // Ignore 
    if (c == '&') continue;


    if (c == '~')
    {
      nextIdNegate = true;
    }
    else if (std::isdigit(c))
    {
      idSs << c;
    }
    // If ss not empty
    else if (idSs.rdbuf()->in_avail() > 0)
    {
      idSs >> wordnetId;
      idSs.str("");
      idSs.clear();

      std::unordered_set<size_t> vecIds;
      GetVectorKeywordsIndicesSetShallow(vecIds, wordnetId, skipConstructedHypernyms);

      // If empty set returned
      if (vecIds.empty())
      {
        break;
      }

      // If this Clause should be negated
      if (nextIdNegate) 
      {
        // Add their negations as ANDs
        for (auto&& id : vecIds)
        {
          Clause tmp{ std::pair(true, id) };        

          resultFormula.push_back(tmp);
        }
        
      }
      else 
      {

        Clause tmp;
        // Add their
        for (auto&& id : vecIds)
        {
          tmp.emplace_back(std::pair(false, id));
        }

        resultFormula.emplace_back(tmp);
      }

      
      nextIdNegate = false;
    }
    else if (c == '-' || c == '+')
    {

    }
    else 
    {
      LOG_ERROR("Parsing query failed");
    }

  }

  return resultFormula;
}

std::vector<size_t> KeywordsContainer::GetCanonicalQueryNoRecur(const std::string& query) const
{
  //EG: &-8252602+-8256735+-3206282+-4296562+

  std::stringstream idSs;
  size_t wordnetId;


  std::vector<size_t> resultFormula;

  // Parse query
  // \todo write complete parser
  for (auto&& c : query)
  {
    // Ignore 
    if (c == '&') continue;

    if (std::isdigit(c))
    {
      idSs << c;
    }
    // If ss not empty
    else if (idSs.rdbuf()->in_avail() > 0)
    {
      idSs >> wordnetId;
      idSs.str("");
      idSs.clear();

      resultFormula.push_back(wordnetId);
    }
    else if (c == '-' || c == '+')
    {

    }
    else
    {
      LOG_ERROR("Parsing query failed");
    }

  }

  return resultFormula;
}




std::vector<size_t> KeywordsContainer::FindAllNeedles(std::string_view hey, std::string_view needle) const
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

std::vector<const Keyword*> KeywordsContainer::GetNearKeywordsConstPtrs(const std::string& prefix) const
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

  std::vector<const Keyword*> resultKeywords;
  resultKeywords.reserve(NUM_SUGESTIONS);
  std::vector<const Keyword*> postResultKeywords;



  // Get desired number of results
  for (size_t j = 0ULL; j < NUM_SUGESTIONS; ++j)
  {
    size_t idx{left + j};

    if (idx >= _keywords.size())
    {
      break;
    }

    Keyword* pKeyword{ _keywords[idx].get() };

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
      resultKeywords.emplace_back(pKeyword);
    }
    else
    {
      postResultKeywords.emplace_back(pKeyword);
    }
  }

  // If we need to add up desc search results
  if (resultKeywords.size() < NUM_SUGESTIONS && prefix.size() >= MIN_DESC_SEARCH_LENGTH)
  {
    std::vector<size_t> needleIndices = FindAllNeedles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      Keyword* pKeyword = MapDescIndexToKeyword(index);

      resultKeywords.emplace_back(pKeyword);
    }
  }

  size_t j = 0ULL;
  while (resultKeywords.size() < NUM_SUGESTIONS)
  {
    if (j >= postResultKeywords.size())
    {
      break;
    }

    resultKeywords.push_back(postResultKeywords[j]);

    ++j;
  }

  return resultKeywords;
}


std::vector<Keyword*> KeywordsContainer::GetNearKeywordsPtrs(const std::string& prefix, size_t numResults)
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

  std::vector<Keyword*> resultKeywords;
  resultKeywords.reserve(numResults);
  std::vector<Keyword*> postResultKeywords;



  // Get desired number of results
  for (size_t j = 0ULL; j < numResults; ++j)
  {
    size_t idx{ left + j };

    if (idx >= _keywords.size())
    {
      break;
    }

    Keyword* pKeyword{ _keywords[idx].get() };

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
      resultKeywords.emplace_back(pKeyword);
    }
    else
    {
      postResultKeywords.emplace_back(pKeyword);
    }
  }

  // If we need to add up desc search results
  if (resultKeywords.size() < NUM_SUGESTIONS && prefix.size() >= MIN_DESC_SEARCH_LENGTH)
  {
    std::vector<size_t> needleIndices = FindAllNeedles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      Keyword* pKeyword = MapDescIndexToKeyword(index);

      resultKeywords.emplace_back(pKeyword);
    }
  }

  size_t j = 0ULL;
  while (resultKeywords.size() < NUM_SUGESTIONS)
  {
    if (j >= postResultKeywords.size())
    {
      break;
    }

    resultKeywords.push_back(postResultKeywords[j]);

    ++j;
  }

  return resultKeywords;
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
      if (descIndex < _descIndexToKeyword[right].first)
      {
        return _descIndexToKeyword[left].second;
      }
      else 
      {
        return _descIndexToKeyword[right].second;
      }
      break;
    }

    i = (right + left) / 2;

  }

  
}

std::string KeywordsContainer::GetKeywordByWordnetId(size_t wordnetId) const
{
  auto resultIt = _wordnetIdToKeywords.find(wordnetId);

  if (resultIt == _wordnetIdToKeywords.end())
  {
    std::string("NOT FOUND");
  }

  return resultIt->second->m_word;
}

Keyword* KeywordsContainer::GetWholeKeywordByWordnetId(size_t wordnetId) const
{
  auto resultIt = _wordnetIdToKeywords.find(wordnetId);

  if (resultIt == _wordnetIdToKeywords.end())
  {
    std::string("NOT FOUND");
  }

  return resultIt->second;
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


std::string KeywordsContainer::GetKeywordDescriptionByWordnetId(size_t wordnetId) const
{

  auto resultIt = _wordnetIdToKeywords.find(wordnetId);

  if (resultIt == _wordnetIdToKeywords.end())
  {
    std::string("NOT FOUND");
  }

  size_t startDescIndex = resultIt->second->m_descStartIndex;

  const char* pDesc = (_allDescriptions.data()) + startDescIndex;


  return std::string(pDesc);
}
