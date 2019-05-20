
#pragma once

#include <string>
using namespace std::literals::string_literals;

#include <algorithm>
#include <cctype>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <locale>

#include <unordered_set>

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

  bool IsHypernym() const { return !m_hyponyms.empty(); };
  bool IsInBinVector() const { return m_vectorIndex == SIZE_T_ERROR_VALUE; };
  bool IsLeafKeyword() const { return (IsHypernym() && IsInBinVector()); }


  size_t m_wordnetId;
  size_t m_vectorIndex;
  size_t m_descStartIndex;
  size_t m_descEndIndex;
  std::string m_word;
  std::vector<size_t> m_hypernyms;
  std::vector<size_t> m_hyponyms;

  //! Set of indices that are hyponyms of this keyword
  std::unordered_set<size_t> m_hyponymBinIndices;

  std::vector<std::string> m_exampleImageFilenames;
};

class KeywordsContainer
{
public:
  KeywordsContainer() = default;
  KeywordsContainer(const std::string& keywordClassesFilepath);

  //! Constructor from database data
  KeywordsContainer(std::vector< std::vector<std::string>>&& data);

  [[deprecated]] 
  std::vector<std::tuple<size_t, std::string, std::string>> GetNearKeywords(const std::string& prefix);

  std::vector<Keyword*> GetNearKeywordsPtrs(const std::string& prefix);

  const Keyword* GetKeywordConstPtrByWordnetId(size_t wordnetId) const;

  Keyword* GetKeywordPtrByWordnetId(size_t wordnetId) const;

  Keyword* MapDescIndexToKeyword(size_t descIndex) const;

#if PUSH_DATA_TO_DB
  bool PushKeywordsToDatabase(Database& db);
#endif
  [[deprecated]]
  Keyword* GetWholeKeywordByWordnetId(size_t wordnetId) const;

  [[deprecated]]
  std::string GetKeywordByWordnetId(size_t wordnetId)const ;

  [[deprecated]]
  KeywordData GetKeywordByVectorIndex(size_t index) const;


  Keyword* GetKeywordPtrByVectorIndex(size_t index) const;
  const Keyword* GetKeywordConstPtrByVectorIndex(size_t index) const;

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
  void GetVectorKeywordsIndicesSet(std::unordered_set<size_t>& destSetRef, size_t wordnetId) const;
  void GetVectorKeywordsIndicesSetShallow(std::unordered_set<size_t>& destIndicesSetRef, size_t wordnetId) const;


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
  
  std::vector<size_t> GetCanonicalQueryNoRecur(const std::string& query) const;

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
      // Force lowercase
      std::locale loc;
      std::string left;
      std::string right;

      for (auto elem : a->m_word)
      {
        left.push_back(std::tolower(elem, loc));
      }

      for (auto elem : b->m_word)
      {
        right.push_back(std::tolower(elem, loc));
      }

      auto result = left.compare(right);

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
  

#if PUSH_DATA_TO_DB
  
  std::vector<std::pair<size_t, std::string>> _keywordToWord;
  std::set<std::string> _words;
#endif

};
