
#pragma once

#include <string>
using namespace std::literals::string_literals;

#include <algorithm>
#include <cctype>
#include <fstream>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include <unordered_set>

#include "common.h"

#include "Database.h"
#include "utility.h"

namespace image_ranker
{
class ImageRanker;

class Keyword
{
 public:
  Keyword() = default;
  Keyword(FrameId ID, size_t wordnetId, size_t vectorIndex, std::string&& word, size_t descStartIndex, size_t descEndIndex,
          std::vector<size_t>&& hypernyms, std::vector<size_t>&& hyponyms, std::string&& description)
      : ID(ID),
        m_wordnetId(wordnetId),
        m_vectorIndex(vectorIndex),
        m_word(std::move(word)),
        m_descStartIndex(descStartIndex),
        m_descEndIndex(descEndIndex),
        m_hypernyms(hypernyms),
        m_hyponyms(hyponyms),
        description(std::move(description))

  {
  }

  std::string description;
  FrameId ID;
  // ==================================

  bool IsHypernym() const { return !m_hyponyms.empty(); };
  bool IsInBinVector() const { return m_vectorIndex == ERR_VAL<size_t>(); };
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
  size_t lastExampleFramesHash = ERR_VAL<size_t>();

  std::vector<Keyword*> m_expanded1Concat;
  std::vector<Keyword*> m_expanded1Substrings;
  std::vector<Keyword*> m_expanded2Concat;
  std::vector<Keyword*> m_expanded2Substrings;

  std::set<std::pair<Keyword*, float>> m_wordToVec;
};

class KeywordsContainer
{
 public:
  KeywordsContainer() = delete;

  // \todo Remove code redundancy here!
  KeywordsContainer(const ViretDataPackRef::VocabData& vocab_data_refs);
  KeywordsContainer(const GoogleDataPackRef::VocabData& vocab_data_refs);
  KeywordsContainer(const W2vvDataPackRef::VocabData& vocab_data_refs);

  const std::string& get_ID() const { return _ID; }
  const std::string& get_description() const { return _description; }

  const Keyword& operator[](KeywordId keyword_ID) const
  {
    return *_ID_to_keyword.at(keyword_ID);
  }
  Keyword& operator[](KeywordId keyword_ID)
  {
    return *_ID_to_keyword.at(keyword_ID);
  }

 private:
  std::string _ID;
  std::string _description;
  std::string _kw_classes_fpth;

  std::map<KeywordId, Keyword*> _ID_to_keyword;

 public:
  void SubstringExpansionPrecompute()
  {
    SubstringExpansionPrecompute1();
    SubstringExpansionPrecompute2();
  }

  const Keyword* GetKeywordPtr(const std::string& wordString) const
  {
    auto pKw{GetNearKeywordsPtrs(wordString, 1)[0]};

    if (!pKw)
    {
      return nullptr;
    }

    // Force lowercase
    std::locale loc;
    std::string lower1;
    std::string lower2;

    // Convert to lowercase
    for (auto elem : pKw->m_word)
    {
      lower1.push_back(std::tolower(elem, loc));
    }

    // Convert to lowercase
    for (auto elem : wordString)
    {
      lower2.push_back(std::tolower(elem, loc));
    }

    // The nearest kw must be the exact match
    if (lower1 != lower2)
    {
      return nullptr;
    }

    return pKw;
  }

  std::string cnfFormulaToString(const CnfFormula& fml)
  {
    std::string result;

    for (auto&& clause : fml)
    {
      result += "( ";
      for (auto&& [isNegated, id] : clause)
      {
        result += GetKeywordConstPtrByVectorIndex(id)->m_word + "|";
      }
      result += " )&";
    }

    return result;
  }

  void SubstringExpansionPrecompute1()
  {
    // For every keyword
    for (auto&& pKwLeft : _keywords)
    {
      // Find all words that contain this word
      for (auto&& pKwRight : _keywords)
      {
        // Do not match against itself
        if (pKwLeft == pKwRight)
        {
          continue;
        }

        // If pKwLeft is subString of pKwRight
        if (pKwRight->m_word.find(pKwLeft->m_word) != std::string::npos)
        {
          pKwLeft->m_expanded1Concat.emplace_back(pKwRight.get());
        }
      }

      // Find all words that are contained in this word
      for (auto&& pKwRight : _keywords)
      {
        // Do not match against itself
        if (pKwLeft == pKwRight)
        {
          continue;
        }

        // If pKwLeft is subString of pKwRight
        if (pKwLeft->m_word.find(pKwRight->m_word) != std::string::npos)
        {
          pKwLeft->m_expanded1Substrings.emplace_back(pKwRight.get());
        }
      }
    }
  }

  void SubstringExpansionPrecompute2()
  {
    // For every keyword
    for (auto&& pKwLeft : _keywords)
    {
      // Find all words that contain this word
      for (auto&& pKwRight : _keywords)
      {
        // Do not match against itself
        if (pKwLeft == pKwRight)
        {
          continue;
        }

        auto subwords{SplitString(pKwRight->m_word, ' ')};

        for (auto&& w : subwords)
        {
          // If pKwLeft is subString of pKwRight
          if (pKwLeft->m_word == w)
          {
            pKwLeft->m_expanded2Concat.emplace_back(pKwRight.get());
          }
        }
      }

      auto subwordsL{SplitString(pKwLeft->m_word, ' ')};
      // Find all words that are contained in this word
      for (auto&& pKwRight : _keywords)
      {
        // Do not match against itself
        if (pKwLeft == pKwRight)
        {
          continue;
        }

        for (auto&& w : subwordsL)
        {
          // If pKwLeft is subString of pKwRight
          if (w == pKwRight->m_word)
          {
            pKwLeft->m_expanded2Substrings.emplace_back(pKwRight.get());
          }
        }
      }
    }
  }

  [[deprecated]] std::vector<std::tuple<size_t, std::string, std::string>> GetNearKeywords(const std::string& prefix);

  std::vector<const Keyword*> GetNearKeywordsConstPtrs(const std::string& prefix) const;
  std::vector<const Keyword*> GetNearKeywordsPtrs(const std::string& prefix, size_t numResults) const;
  std::vector<Keyword*> GetNearKeywordsPtrs(const std::string& prefix, size_t numResults);

  const Keyword* GetKeywordConstPtrByWordnetId(size_t wordnetId) const;

  Keyword* GetKeywordPtrByWordnetId(size_t wordnetId) const;

  Keyword* MapDescIndexToKeyword(size_t descIndex) const;

  [[deprecated]] Keyword* GetWholeKeywordByWordnetId(size_t wordnetId) const;

  [[deprecated]] std::string GetKeywordByWordnetId(size_t wordnetId) const;

  [[deprecated]] KeywordData GetKeywordByVectorIndex(size_t index) const;

  Keyword* GetKeywordPtrByVectorIndex(size_t index) const;
  const Keyword* GetKeywordConstPtrByVectorIndex(size_t index) const;

  std::string GetKeywordDescriptionByWordnetId(size_t wordnetId) const;

  CnfFormula GetCanonicalQuery(const std::string& query, bool skipConstructedHypernyms = false) const;

  Keyword* GetKeywordByWord(const std::string& keyword) const
  {
    auto item = std::lower_bound(_keywords.begin(), _keywords.end(), keyword,
                                 [](const std::unique_ptr<Keyword>& pWord, const std::string& str) {
                                   // Compare strings
                                   auto result = pWord->m_word.compare(str);

                                   return result <= -1;
                                 });
    if (item == _keywords.cend() || item->get()->m_word != keyword)
    {
      //LOGE("This keyword not found.");
      return nullptr;
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
  void GetVectorKeywordsIndicesSetShallow(std::unordered_set<size_t>& destIndicesSetRef, size_t wordnetId,
                                          bool skipConstructedHypernyms = false) const;

  std::string StringifyCnfFormula(const CnfFormula& formula)
  {
    std::string result;

    for (auto&& clause : formula)
    {
      result += "(";

      for (auto&& vecIndex : clause)
      {
        result += std::get<1>(GetKeywordByVectorIndex(vecIndex.atom)) + " | "s;
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
  /*!
   * Functor for comparing our string=>wordnetId structure
   *
   */
  struct KeywordLessThanComparator
  {
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

  struct KeywordLessThanStringComparator
  {
    bool operator()(const std::string& a, const std::string& b) const
    {
      // Compare strings

      std::string aa{a};
      std::string bb{b};
      std::transform(aa.begin(), aa.end(), aa.begin(), ::tolower);
      std::transform(bb.begin(), bb.end(), bb.begin(), ::tolower);

      auto result = aa.compare(bb);

      return result <= -1;
    }
  };

 public:
  std::vector<size_t> FindAllNeedles(std::string_view hey, std::string_view needle) const;
  //! Keywords
  std::vector<std::unique_ptr<Keyword>> _keywords;

  //! Maps wordnetID to Keyword
  std::map<size_t, Keyword*> _wordnetIdToKeywords;

 private:
  //! One huge string of all descriptions for fast keyword search
  std::string _allDescriptions;

  //! Maps index from probability vector to Keyword
  std::map<size_t, Keyword*> _vecIndexToKeyword;

  std::vector<std::pair<size_t, Keyword*>> _descIndexToKeyword;
};
}  // namespace image_ranker