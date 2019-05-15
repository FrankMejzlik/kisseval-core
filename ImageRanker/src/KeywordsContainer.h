
#pragma once

#include <string>
using namespace std::literals::string_literals;

#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <locale>

#include <set>

#include "config.h"

#include "utility.h"
#include "Database.h"

using Clause = std::vector<std::pair<bool, size_t>>;
using CnfFormula = std::vector<Clause>;

struct Keyword
{
  Keyword() = default;
  Keyword(
    size_t wordnetId, 
    size_t vectorIndex, 
    std::string&& word,
    size_t descStartIndex,
    size_t descEndIndex,
    std::vector<size_t>&& hypernyms,
    std::vector<size_t>&& hyponyms
  ):
    m_wordnetId(wordnetId),
    m_vectorIndex(vectorIndex),
    m_word(std::move(word)),
    m_descStartIndex(descStartIndex),
    m_descEndIndex(descEndIndex),
    m_hypernyms(hypernyms),
    m_hyponyms(hyponyms)

  {}

  size_t m_wordnetId;
  size_t m_vectorIndex;
  size_t m_descStartIndex;
  size_t m_descEndIndex;
  std::string m_word;
  std::vector<size_t> m_hypernyms;
  std::vector<size_t> m_hyponyms;
  std::set<size_t> m_hypernymIndices;
};

class KeywordsContainer
{
public:
  KeywordsContainer() = default;
  KeywordsContainer(const std::string& keywordClassesFilepath);

  //! Constructor from database data
  KeywordsContainer(std::vector< std::vector<std::string>>&& data);

  std::vector<std::tuple<size_t, std::string, std::string>> GetNearKeywords(const std::string& prefix);

  Keyword* MapDescIndexToKeyword(size_t descIndex) const;

#if PUSH_DATA_TO_DB
  bool PushKeywordsToDatabase(Database& db);
#endif

  std::string GetKeywordByWordnetId(size_t wordnetId)const ;
  KeywordData GetKeywordByVectorIndex(size_t index) const;

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId) const;
  
  CnfFormula GetCanonicalQuery(const std::string& query) const;



  Keyword* GetKeywordByWord(const std::string& keyword) const
  {
    auto item = std::lower_bound(_keywords.begin(), _keywords.end(), keyword,
      [](const std::unique_ptr<Keyword>& pWord, const std::string& str) 
      {
        // Compare strings
        auto result = pWord->m_word.compare(str);

        return result <= -1;
      }
    );
    if (item == _keywords.cend()) 
    {
      LOG_ERROR("This keyword not found.");
    }

    return item->get();
  }

  /*!
   * Returns vector of keywords that are present in ranking vector of images.
   *
   * \return
   */
  std::vector<size_t> GetVectorKeywords(size_t wordnetId) const;

  std::vector<size_t> GetVectorKeywordsIndices(size_t wordnetId) const;
  std::set<size_t> GetVectorKeywordsIndicesSet(size_t wordnetId) const;


  std::string StringifyCnfFormula(const CnfFormula& formula)
  {
    std::string result;

    for (auto&& clause : formula)
    {
      result += "(";

      for (auto&& vecIndex : clause)
      {
        result += std::get<1>(GetKeywordByVectorIndex(vecIndex.second)) + " | "s;
      }
      result.pop_back();
      result.pop_back();

      result += ") & ";
    }
    result.pop_back();
    result.pop_back();

    return result;
  }

  size_t GetNetVectorSize() const { return _vecIndexToKeyword.size(); }

private:
  bool ParseKeywordClassesFile(const std::string& filepath);
  bool ParseKeywordDbDataStructure(std::vector< std::vector<std::string>>&& data);

  /*!
  * Functor for comparing our string=>wordnetId structure
  * 
  */
  struct KeywordLessThanComparator {
    bool operator()(const std::unique_ptr<Keyword>& a, const std::unique_ptr<Keyword>& b) const
    {   
      // Compare strings
      auto result = a->m_word.compare(b->m_word);

      return result <= -1;
    }   
  } keywordLessThan;


  struct KeywordLessThanStringComparator {
    bool operator()(const std::string& a, const std::string& b) const
    {   
      // Compare strings

      std::string aa{a};
      std::string bb{ b };
      std::transform(aa.begin(), aa.end(), aa.begin(), ::tolower);
      std::transform(bb.begin(), bb.end(), bb.begin(), ::tolower);

      auto result = aa.compare(bb);

      return result <= -1;
    }   
  };

  public:
  std::vector<size_t> FindAllNeedles(std::string_view hey, std::string_view needle);
  //! Keywords
  std::vector< std::unique_ptr<Keyword> > _keywords;

  //! Maps wordnetID to Keyword
  std::map<size_t, Keyword*> _wordnetIdToKeywords;
private:
 

  //! One huge string of all descriptions for fast keyword search
  std::string _allDescriptions;

  //! Maps index from probability vector to Keyword
  std::map<size_t, Keyword*> _vecIndexToKeyword;

  std::vector<std::pair<size_t, Keyword*>> _descIndexToKeyword;

  size_t _approxDescLen;

  

#if PUSH_DATA_TO_DB
  
  std::vector<std::pair<size_t, std::string>> _keywordToWord;
  std::set<std::string> _words;
#endif

};
