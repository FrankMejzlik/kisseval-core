
#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
#include <locale>
#include <set>

#include "config.h"
#include "Database.h"

struct Keyword
{
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
};

class KeywordsContainer
{
public:
  KeywordsContainer() = default;
  KeywordsContainer(std::string_view keywordClassesFilepath);

  //! Constructor from database data
  KeywordsContainer(std::vector< std::vector<std::string>>&& data);

  std::vector<std::tuple<size_t, std::string, std::string>> GetNearKeywords(const std::string& prefix);

  Keyword* MapDescIndexToKeyword(size_t descIndex) const;

#if PUSH_DATA_TO_DB
  bool PushKeywordsToDatabase(Database& db);
#endif

  std::string GetKeywordByWordnetId(size_t wordnetId);
  std::string GetKeywordByVectorIndex(size_t index) const;

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId);

private:
  bool ParseKeywordClassesFile(std::string_view filepath);
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

  std::vector<size_t> FindAllNeedles(std::string_view hey, std::string_view needle);

private:
  std::vector< std::unique_ptr<Keyword> > _keywords;

  //! Maps wordnetID to Keyword
  std::map<size_t, Keyword*> _wordnetIdToKeywords;

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
