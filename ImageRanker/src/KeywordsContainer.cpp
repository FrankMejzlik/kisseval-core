

#include "KeywordsContainer.h"

#include <algorithm>
#include <cctype>

#include "ImageRanker.h"

using namespace image_ranker;

KeywordsContainer::KeywordsContainer(const ViretDataPackRef::VocabData& vocab_data_refs)
    : _ID(vocab_data_refs.ID),
      _description(vocab_data_refs.description),
      _kw_classes_fpth(vocab_data_refs.keyword_synsets_fpth)
{
  parse_keywords(_kw_classes_fpth);
}

KeywordsContainer::KeywordsContainer(const GoogleDataPackRef::VocabData& vocab_data_refs)
    : _ID(vocab_data_refs.ID),
      _description(vocab_data_refs.description),
      _kw_classes_fpth(vocab_data_refs.keyword_synsets_fpth)
{
  parse_keywords(_kw_classes_fpth);
}

KeywordsContainer::KeywordsContainer(const W2vvDataPackRef::VocabData& vocab_data_refs)
    : _ID(vocab_data_refs.ID),
      _description(vocab_data_refs.description),
      _kw_classes_fpth(vocab_data_refs.keyword_synsets_fpth)
{
  parse_keywords(_kw_classes_fpth);
}

void KeywordsContainer::parse_keywords(const std::string& filepath)
{
  ViretKeywordClassesParsedData res = FileParser::parse_VIRET_format_keyword_classes_file(filepath);

  // Store results
  _allDescriptions = std::move(res.all_descriptions);
  _vecIndexToKeyword = std::move(res.vec_idx_to_keyword);
  _wordnetIdToKeywords = std::move(res.wordnet_ID_to_keyword);
  _descIndexToKeyword = std::move(res.desc_index_to_keyword);
  _keywords = std::move(res.keywords);
  _ID_to_keyword = std::move(res.ID_to_keyword);
  _ID_to_allkeywords = std::move(res.ID_to_all_keywords);

  // Sort keywords
  std::sort(_keywords.begin(), _keywords.end(), keywordLessThan);

  for (auto&& kw : _keywords)
  {
    std::unordered_set<size_t> accum_indices;
    // Add self
    accum_indices.emplace(kw->m_vectorIndex);

    for (auto&& hypo_wordnet_ID : kw->m_hyponyms)
    {
      GetVectorKeywordsIndicesSet(accum_indices, hypo_wordnet_ID);
    }

    kw->m_hyponymBinIndices = std::move(accum_indices);
  }
}

void KeywordsContainer::GetVectorKeywordsIndicesSet(std::unordered_set<size_t>& destIndicesSetRef,
                                                    size_t wordnetId) const
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

void KeywordsContainer::GetVectorKeywordsIndicesSetShallow(std::unordered_set<size_t>& destIndicesSetRef,
                                                           size_t wordnetId, bool skipConstructedHypernyms) const
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
    LOGE("Keyword not found!");

    return std::vector<size_t>();
  }

  Keyword* pRootKeyword = wordnetIdKeywordPair->second;

  // It not vector keyword
  if (pRootKeyword->m_vectorIndex == ERR_VAL<size_t>())
  {
    std::vector<size_t> result;

    // Recursively get hyponyms
    for (auto&& hypo : pRootKeyword->m_hyponyms)
    {
      auto recur = GetVectorKeywords(hypo);
      result.reserve(result.size() + recur.size());
      result.insert(result.end(), recur.begin(), recur.end());
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
    LOGE("Keyword not found!");

    return std::vector<size_t>();
  }

  Keyword* pRootKeyword = wordnetIdKeywordPair->second;

  // It not vector keyword
  if (pRootKeyword->m_vectorIndex == ERR_VAL<size_t>())
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
  if (pairIt == _wordnetIdToKeywords.end())
  {
    LOGE("Keyword with ID "s + std::to_string(wordnetId) + " not found!");
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
    LOGE("Keyword with ID "s + std::to_string(wordnetId) + " not found!");
    PROD_THROW("Data error.");
  }

  return pairIt->second;
}

CnfFormula KeywordsContainer::GetCanonicalQuery(const std::string& query, bool skipConstructedHypernyms) const
{
  // skipConstructedHypernyms = IGNORE_CONSTRUCTED_HYPERNYMS;
  // EG: &-8252602+-8256735+-3206282+-4296562+

  std::stringstream idSs;
  size_t wordnetId;
  bool nextIdNegate{ false };

  std::vector<Clause> resultFormula;

  // Parse query
  // \todo write complete parser
  for (auto&& c : query)
  {
    // Ignore
    if (c == '&')
    {
      continue;
    }

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
          Clause tmp{ { id, true } };

          resultFormula.push_back(tmp);
        }
      }
      else
      {
        Clause tmp;
        // Add their
        for (auto&& id : vecIds)
        {
          tmp.emplace_back(Literal<KeywordId>{ id, false });
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
      LOGE("Parsing query failed");
    }
  }

  return resultFormula;
}

std::vector<size_t> KeywordsContainer::GetCanonicalQueryNoRecur(const std::string& query) const
{
  // EG: &-8252602+-8256735+-3206282+-4296562+

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
      LOGE("Parsing query failed");
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
  std::vector<int> failure(needle.size(), -1);

  for (int r = 1, l = -1; r < needle.size(); ++r)
  {
    while (l != -1 && needle[l + 1] != needle[r]) l = failure[l];

    // assert( l == -1 || needle[l+1] == needle[r]);
    if (needle[l + 1] == needle[r]) failure[r] = ++l;
  }

  // Step 2. Search needle
  int tail = -1;
  for (int i = 0; i < hey.size(); i++)
  {
    while (tail != -1 && hey[i] != needle[tail + 1]) tail = failure[tail];

    if (hey[i] == needle[tail + 1]) tail++;

    if (tail == needle.size() - 1)
    {
      resultIndices.push_back(i - tail);
      tail = -1;
    }

    // Gather maximum of needles
    if (resultIndices.size() >= DEF_NUMBER_OF_TOP_KWS)
    {
      return resultIndices;
    }
  }

  return resultIndices;
}

std::vector<std::tuple<size_t, std::string, std::string>> KeywordsContainer::GetNearKeywords(const std::string& prefix)
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

  std::vector<std::tuple<size_t, std::string, std::string>> resultWordnetIds;
  resultWordnetIds.reserve(DEF_NUM_AUTOCOMPLETE_RESULTS);

  std::vector<std::tuple<size_t, std::string, std::string>> postResultWordnetIds;

  // Get desired number of results
  for (size_t j = 0ULL; j < DEF_NUM_AUTOCOMPLETE_RESULTS; ++j)
  {
    Keyword* pKeyword{ _keywords[left + j].get() };

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
      resultWordnetIds.push_back(std::make_tuple(pKeyword->m_wordnetId, pKeyword->m_word,
                                                 GetKeywordDescriptionByWordnetId(pKeyword->m_wordnetId)));
    }
    else
    {
      postResultWordnetIds.push_back(std::make_tuple(pKeyword->m_wordnetId, pKeyword->m_word,
                                                     GetKeywordDescriptionByWordnetId(pKeyword->m_wordnetId)));
    }
  }

  // If we need to add up desc search results
  if (resultWordnetIds.size() < DEF_NUM_AUTOCOMPLETE_RESULTS && prefix.size() >= MIN_DESC_SEARCH_LENGTH)
  {
    std::vector<size_t> needleIndices = FindAllNeedles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      Keyword* pKeyword = MapDescIndexToKeyword(index);

      resultWordnetIds.push_back(std::make_tuple(pKeyword->m_wordnetId, pKeyword->m_word,
                                                 GetKeywordDescriptionByWordnetId(pKeyword->m_wordnetId)));
    }
  }

  size_t j = 0ULL;
  while (resultWordnetIds.size() < DEF_NUM_AUTOCOMPLETE_RESULTS)
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
  resultKeywords.reserve(DEF_NUM_AUTOCOMPLETE_RESULTS);
  std::vector<const Keyword*> postResultKeywords;

  // Get desired number of results
  for (size_t j = 0ULL; j < DEF_NUM_AUTOCOMPLETE_RESULTS; ++j)
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
  if (resultKeywords.size() < DEF_NUM_AUTOCOMPLETE_RESULTS && prefix.size() >= MIN_DESC_SEARCH_LENGTH)
  {
    std::vector<size_t> needleIndices = FindAllNeedles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      Keyword* pKeyword = MapDescIndexToKeyword(index);

      resultKeywords.emplace_back(pKeyword);
    }
  }

  size_t j = 0ULL;
  while (resultKeywords.size() < DEF_NUM_AUTOCOMPLETE_RESULTS)
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

std::vector<const Keyword*> KeywordsContainer::GetNearKeywordsPtrs(const std::string& prefix, size_t numResults) const
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
  resultKeywords.reserve(numResults);
  std::vector<const Keyword*> postResultKeywords;

  // Get desired number of results
  for (size_t j = 0ULL; j < numResults; ++j)
  {
    size_t idx{ left + j };

    if (idx >= _keywords.size())
    {
      break;
    }

    const Keyword* pKeyword{ _keywords[idx].get() };

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
  if (resultKeywords.size() < numResults && prefix.size() >= MIN_DESC_SEARCH_LENGTH)
  {
    std::vector<size_t> needleIndices = FindAllNeedles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      const Keyword* pKeyword = MapDescIndexToKeyword(index);

      resultKeywords.emplace_back(pKeyword);
    }
  }

  size_t j = 0ULL;
  while (resultKeywords.size() < numResults)
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
  if (resultKeywords.size() < numResults && prefix.size() >= MIN_DESC_SEARCH_LENGTH)
  {
    std::vector<size_t> needleIndices = FindAllNeedles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      Keyword* pKeyword = MapDescIndexToKeyword(index);

      resultKeywords.emplace_back(pKeyword);
    }
  }

  size_t j = 0ULL;
  while (resultKeywords.size() < numResults)
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
  size_t right = _descIndexToKeyword.size() - 1;

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
