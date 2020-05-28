

#include "KeywordsContainer.h"

#include <algorithm>
#include <cctype>

#include "ImageRanker.h"

using namespace image_ranker;

KeywordsContainer::KeywordsContainer(const ViretDataPackRef::VocabData& vocab_data_refs)
    : _vocabulary_ID(vocab_data_refs.ID),
      _vocabulary_desc(vocab_data_refs.description),
      _kw_classes_fpth(vocab_data_refs.keyword_synsets_fpth)
{
  parse_keywords(_kw_classes_fpth);
}

KeywordsContainer::KeywordsContainer(const GoogleDataPackRef::VocabData& vocab_data_refs)
    : _vocabulary_ID(vocab_data_refs.ID),
      _vocabulary_desc(vocab_data_refs.description),
      _kw_classes_fpth(vocab_data_refs.keyword_synsets_fpth)
{
  parse_keywords(_kw_classes_fpth);
}

KeywordsContainer::KeywordsContainer(const W2vvDataPackRef::VocabData& vocab_data_refs)
    : _vocabulary_ID(vocab_data_refs.ID),
      _vocabulary_desc(vocab_data_refs.description),
      _kw_classes_fpth(vocab_data_refs.keyword_synsets_fpth)
{
  parse_keywords(_kw_classes_fpth);
}

void KeywordsContainer::parse_keywords(const std::string& filepath)
{
  ViretKeywordClassesParsedData res = FileParser::parse_VIRET_format_keyword_classes_file(filepath);

  // Store results
  _allDescriptions = std::move(res.all_descriptions);
  class_idx_to_keyword = std::move(res.vec_idx_to_keyword);
  _wordnet_ID_to_keyword = std::move(res.wordnet_ID_to_keyword);
  _desc_indext_to_keyword = std::move(res.desc_index_to_keyword);
  _keywords = std::move(res.keywords);
  _ID_to_keyword = std::move(res.ID_to_keyword);
  _ID_to_allkeywords = std::move(res.ID_to_all_keywords);

  // Sort keywords
  std::sort(_keywords.begin(), _keywords.end(), kw_less_hierarch_cmp);

  for (auto&& kw : _keywords)
  {
    std::unordered_set<size_t> accum_indices;
    // Add self
    accum_indices.emplace(kw->classification_index);

    for (auto&& hypo_wordnet_ID : kw->hyponyms)
    {
      get_keyword_hyponyms_indices_set(accum_indices, hypo_wordnet_ID);
    }

    kw->hyponym_class_inidces = std::move(accum_indices);
  }
}

void KeywordsContainer::get_keyword_hyponyms_indices_set(std::unordered_set<size_t>& dest_set, size_t wordnet_ID) const
{
  // Get this Keyword
  const Keyword* p_root_kw = get_keyword_by_wordnet_ID(wordnet_ID);

  // If this hypernyms has spot in data vector
  if (!p_root_kw->is_classified())
  {
    // Add it to set as well
    dest_set.emplace(p_root_kw->classification_index);
  }

  // If is hypernym
  if (p_root_kw->is_hypernym())
  {
    // Recursively get hyponyms into provided set
    for (auto&& hypo : p_root_kw->hyponyms)
    {
      get_keyword_hyponyms_indices_set(dest_set, hypo);
    }
  }
}

void KeywordsContainer::get_keyword_hyponyms_indices_set_nearest(std::unordered_set<size_t>& dest_set,
                                                                 size_t wordnet_ID, bool skip_pure_hypers) const
{
  // Get this Keyword
  const Keyword* p_root_kw = get_keyword_by_wordnet_ID(wordnet_ID);

  // If this hypernyms has spot in data vector
  if (!p_root_kw->is_classified())
  {
    // Add it to set as well
    dest_set.emplace(p_root_kw->classification_index);
  }
  else
  {
    // If we want to include subsets of constructed hypernyms
    if (!skip_pure_hypers)
    {
      // Recursively get hyponyms into provided set
      for (auto&& hypo : p_root_kw->hyponyms)
      {
        get_keyword_hyponyms_indices_set_nearest(dest_set, hypo);
      }
    }
  }
}

Keyword* KeywordsContainer::get_keyword_by_class_index(size_t index)
{
  auto result = class_idx_to_keyword.find(index);

  if (result == class_idx_to_keyword.end())
  {
    return nullptr;
  }

  return result->second;
}

const Keyword* KeywordsContainer::get_keyword_ptr_by_class_index(size_t index) const
{
  auto result = class_idx_to_keyword.find(index);

  if (result == class_idx_to_keyword.end())
  {
    return nullptr;
  }

  return result->second;
}

std::vector<size_t> KeywordsContainer::get_classified_hyponyms_IDs(size_t wordnet_ID) const
{
  // Find root keyword
  auto wordnetIdKeywordPair = _wordnet_ID_to_keyword.find(wordnet_ID);

  if (wordnetIdKeywordPair == _wordnet_ID_to_keyword.end())
  {
    LOGE("Keyword not found!");

    return std::vector<size_t>();
  }

  Keyword* pRootKeyword = wordnetIdKeywordPair->second;

  // It not vector keyword
  if (pRootKeyword->classification_index == ERR_VAL<size_t>())
  {
    std::vector<size_t> result;

    // Recursively get hyponyms
    for (auto&& hypo : pRootKeyword->hyponyms)
    {
      auto recur = get_classified_hyponyms_IDs(hypo);
      result.reserve(result.size() + recur.size());
      result.insert(result.end(), recur.begin(), recur.end());
    }

    return result;
  }

  // If vector word, return self
  return std::vector<size_t>{ pRootKeyword->wordnet_ID };
}

const Keyword* KeywordsContainer::get_keyword_by_wordnet_ID(size_t wordnetId) const
{
  // Find root keyword
  auto pairIt = _wordnet_ID_to_keyword.find(wordnetId);

  // If this key was not found
  if (pairIt == _wordnet_ID_to_keyword.end())
  {
    LOGE("Keyword with ID "s + std::to_string(wordnetId) + " not found!");
    PROD_THROW("Data error.");
  }

  return pairIt->second;
}

Keyword* KeywordsContainer::get_keyword_by_wordnet_ID(size_t wordnetId)
{
  // Find root keyword
  auto pairIt = _wordnet_ID_to_keyword.find(wordnetId);

  // If this key was not found
  if (pairIt == _wordnet_ID_to_keyword.end())
  {
    LOGE("Keyword with ID "s + std::to_string(wordnetId) + " not found!");
    PROD_THROW("Data error.");
  }

  return pairIt->second;
}

Keyword* KeywordsContainer::get_keyword_by_word(const std::string& word)
{
  auto item = std::lower_bound(_keywords.begin(), _keywords.end(), word,
                               [](const std::unique_ptr<Keyword>& pWord, const std::string& str) {
                                 // Compare strings
                                 auto result = pWord->word.compare(str);

                                 return result <= -1;
                               });
  if (item == _keywords.cend() || item->get()->word != word)
  {
    return nullptr;
  }

  return item->get();
}

std::vector<size_t> KeywordsContainer::find_all_needles(std::string_view hey, std::string_view needle) const
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

const Keyword* KeywordsContainer::get_keyword_by_word(const std::string& word) const
{
  auto p_kw{ get_near_keywords(word, 1)[0] };

  if (p_kw == nullptr)
  {
    return nullptr;
  }

  // Force lowercase
  std::locale loc;
  std::string lower1;
  std::string lower2;

  // Convert to lowercase
  for (auto elem : p_kw->word)
  {
    lower1.push_back(std::tolower(elem, loc));
  }

  // Convert to lowercase
  for (auto elem : word)
  {
    lower2.push_back(std::tolower(elem, loc));
  }

  // The nearest kw must be the exact match
  if (lower1 != lower2)
  {
    return nullptr;
  }

  return p_kw;
}

std::string KeywordsContainer::CNF_index_formula_to_string(const CnfFormula& fml) const
{
  std::string result;

  for (auto&& clause : fml)
  {
    result += "( ";
    for (auto&& [idx, is_negated] : clause)
    {
      result += get_keyword_ptr_by_class_index(idx)->word + "|";
    }
    result += " )&";
  }

  return result;
}

std::vector<const Keyword*> KeywordsContainer::get_near_keywords(const std::string& prefix, size_t numResults) const
{
  size_t left = 0ULL;
  size_t right = _keywords.size() - 1ULL;

  size_t i = right / 2;

  while (true)
  {
    // Test if middle one is less than
    bool leftIsLess = kw_string_less_cmp(_keywords[i]->word, prefix);

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

    for (auto elem : pKeyword->word)
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
    std::vector<size_t> needleIndices = find_all_needles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      const Keyword* pKeyword = desc_index_to_keyword(index);

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

std::vector<Keyword*> KeywordsContainer::get_near_keywords(const std::string& prefix, size_t numResults)
{
  size_t left = 0ULL;
  size_t right = _keywords.size() - 1ULL;

  size_t i = right / 2;

  while (true)
  {
    // Test if middle one is less than
    bool leftIsLess = kw_string_less_cmp(_keywords[i]->word, prefix);

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

    for (auto elem : pKeyword->word)
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
    std::vector<size_t> needleIndices = find_all_needles(_allDescriptions, prefix);

    for (auto&& index : needleIndices)
    {
      Keyword* pKeyword = desc_index_to_keyword(index);

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

Keyword* KeywordsContainer::desc_index_to_keyword(size_t descIndex) const
{
  size_t left = 0ULL;
  size_t right = _desc_indext_to_keyword.size() - 1;

  size_t i = (right + left) / 2;

  while (true)
  {
    // Test if middle one is less than
    bool pivotLess = descIndex < _desc_indext_to_keyword[i].first;

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
      if (descIndex < _desc_indext_to_keyword[right].first)
      {
        return _desc_indext_to_keyword[left].second;
      }
      else
      {
        return _desc_indext_to_keyword[right].second;
      }
      break;
    }

    i = (right + left) / 2;
  }
}

std::string KeywordsContainer::GetKeywordDescriptionByWordnetId(size_t wordnetId) const
{
  auto resultIt = _wordnet_ID_to_keyword.find(wordnetId);

  if (resultIt == _wordnet_ID_to_keyword.end())
  {
    std::string("NOT FOUND");
  }

  size_t startDescIndex = resultIt->second->description_begin_idx;

  const char* pDesc = (_allDescriptions.data()) + startDescIndex;

  return std::string(pDesc);
}
