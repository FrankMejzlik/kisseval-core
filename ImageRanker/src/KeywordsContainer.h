
#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <map>
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

  std::vector<std::tuple<size_t, std::string, std::string>> GetNearKeywords(const std::string& prefix);

  Keyword* MapDescIndexToKeyword() const;

  bool PushKeywordsToDatabase(Database& db);
  

  std::string GetKeywordByWordnetId(size_t wordnetId);

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId);

private:
  bool ParseKeywordClassesFile(std::string_view filepath);

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
      auto result = a.compare(b);

      return result <= -1;
    }   
  };

private:
  std::vector< std::unique_ptr<Keyword> > _keywords;

  //! Maps wordnetID to Keyword
  std::map<size_t, Keyword*> _wordnetIdToKeywords;

  //! One huge string of all descriptions for fast keyword search
  std::string _allDescriptions;

  //! Maps index from probability vector to Keyword
  std::map<size_t, Keyword*> _vecIndexToKeyword;

#if !USE_DATA_FROM_DATABASE
  std::set<std::string> _words;
#endif

};
