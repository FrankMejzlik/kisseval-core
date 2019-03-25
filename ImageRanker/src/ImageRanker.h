#pragma once

#include <iostream>
#include <assert.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <array>
#include <string>
#include <chrono>
#include <unordered_map>
#include <sstream>
#include <random>
#include <algorithm>
#include <windows.h>
#include <map>
#include <locale>

#include "config.h"

struct Keyword
{
  Keyword(
    size_t wordnetId, 
    size_t vectorIndex, 
    const std::string& word,
    size_t descStartIndex,
    size_t descEndIndex
  ):
    m_wordnetId(wordnetId),
    m_vectorIndex(vectorIndex),
    m_word(word),
    m_pHypernym(nullptr),
    m_descStartIndex(descStartIndex),
    m_descEndIndex(descEndIndex)
  {}

  Keyword(
    size_t wordnetId, 
    size_t vectorIndex, 
    std::string&& word,
    size_t descStartIndex,
    size_t descEndIndex
  ):
    m_wordnetId(wordnetId),
    m_word(std::move(word)),
    m_pHypernym(nullptr),
    m_descStartIndex(descStartIndex),
    m_descEndIndex(descEndIndex)
  {}


  size_t m_wordnetId;
  size_t m_vectorIndex;
  size_t m_descStartIndex;
  size_t m_descEndIndex;
  std::string m_word;
  Keyword* m_pHypernym;
  std::vector<Keyword*> m_pHyponyms;
};

struct Image
{
  size_t _wordnetId;
  std::string _filename;
  std::vector< std::pair<Keyword, float> >_probabilityVector;
};

class KeywordsContainer
{
public:
  KeywordsContainer(std::string_view keywordClassesFilepath)
  {
    // Parse data
    ParseKeywordClassesFile(keywordClassesFilepath);

    // Sort keywords
    std::sort(_keywords.begin(), _keywords.end(), KeywordLessThan);
  }

  std::vector<std::tuple<size_t, std::string, std::string>> GetNearKeywords(const std::string& prefix);

  Keyword* MapDescIndexToKeyword() const;


  std::string GetKeywordByWordnetId(size_t wordnetId)
  {
    auto resultIt = _wordnetIdToKeywords.find(wordnetId);

    if (resultIt == _wordnetIdToKeywords.end())
    {
      std::string("NOT FOUND");
    }
    
    return resultIt->second->m_word;
  }

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId)
  {

    auto resultIt = _wordnetIdToKeywords.find(wordnetId);

    if (resultIt == _wordnetIdToKeywords.end())
    {
      std::string("NOT FOUND");
    }

    size_t startDescIndex = resultIt->second->m_descStartIndex;
    //size_t endDescIndex = resultIt->second->m_descEndIndex;

    char* pDesc = (_allDescriptions.data()) + startDescIndex;


    return std::string(pDesc);
  }

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
  } KeywordLessThan;


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

};

class ImageRanker
{
  // Structures
public:
  using Buffer = std::vector<std::byte>;


  // Methods
public:
  ImageRanker() = delete;
  ImageRanker(
    std::string_view imagesPath,
    std::string_view probabilityVectorFilepath,
    std::string_view deepFeaturesFilepath,
    std::string_view keywordClassesFilepath
  );

  size_t GetRandomImageId() const;
  

  std::vector< std::tuple<size_t, std::string, std::string> > GetNearKeywords(const std::string& prefix)
  {
    // Force lowercase
    std::locale loc;
    std::string lower;

    for (auto elem : prefix)
    {
      lower.push_back(std::tolower(elem,loc));
    }

    return _keywords.GetNearKeywords(lower);
  }


  std::string GetKeywordByWordnetId(size_t wordnetId)
  {
    return _keywords.GetKeywordByWordnetId(wordnetId);
  }

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId)
  {
    return _keywords.GetKeywordDescriptionByWordnetId(wordnetId);
  }

  std::string GetImageFilepathByIndex(size_t imgIndex, bool relativePaths = false) const;

  

  int GetRandomInteger(int from, int to) const;
  private:

    

    std::vector< std::pair< size_t, std::vector<float> > > ParseSoftmaxBinFile(std::string_view filepath) const;

    std::vector< std::pair< size_t, std::unordered_map<size_t, float> > > ParseSoftmaxBinFileFiltered(std::string_view filepath, float minProbabilty) const;

    std::unordered_map<size_t, std::pair<size_t, std::string> > ParseKeywordClassesTextFile(std::string_view filepath) const;

    std::unordered_map<size_t, std::pair<size_t, std::string> > ParseHypernymKeywordClassesTextFile(std::string_view filepath) const;

    /*!
    * Loads bytes from specified file into buffer
    *
    * \param filepath  Path to file to load.
    * \return New vector byte buffer.
    */
    std::vector<std::byte> LoadFileToBuffer(std::string_view filepath) const;

    /*!
    * Parses Little Endian integer from provided buffer starting at specified index.
    *
    * \param buffer  Reference to source buffer.
    * \param startIndex  Index where integer starts.
    * \return  Correct integer representation.
    */
    int32_t ParseIntegerLE(const Buffer& buffer, size_t startIndex) const;
    /*!
    * Parses Little Endian float from provided buffer starting at specified index.
    *
    * \param buffer  Reference to source buffer.
    * \param startIndex  Index where float starts.
    * \return  Correct float representation.
    */
    float ParseFloatLE(const Buffer& buffer, size_t startIndex) const;


public:

  KeywordsContainer _keywords;

};