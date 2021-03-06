
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
#include <unordered_set>
#include <vector>

#include "common.h"
#include "utility.h"

namespace image_ranker
{
class ImageRanker;

/**
 * Represents one keyword (class).
 *
 * \see image_ranker::KeywordsContainer
 */
struct [[nodiscard]] Keyword
{
  /*
   * Methods
   */
  Keyword(FrameId ID, size_t wordnet_ID, size_t classification_index, std::string && word, size_t description_begin_idx,
          size_t description_end_idx, std::vector<size_t> && hypernyms, std::vector<size_t> && hyponyms,
          std::string && description)
      : ID(ID),
        wordnet_ID(wordnet_ID),
        classification_index(classification_index),
        description_begin_idx(description_begin_idx),
        description_end_idx(description_end_idx),
        hypernyms(hypernyms),
        hyponyms(hyponyms),
        word(std::move(word)),
        description(std::move(description))
  {
  }

  [[nodiscard]] bool is_hypernym() const { return !hyponyms.empty(); };
  [[nodiscard]] bool is_classified() const { return classification_index == ERR_VAL<size_t>(); };
  [[nodiscard]] bool is_leaf_keyword() const { return (is_hypernym() && is_classified()); }

  /*
   * Member variables
   */
  FrameId ID{ ERR_VAL<FrameId>() };
  size_t wordnet_ID{ ERR_VAL<size_t>() };
  size_t classification_index{ ERR_VAL<size_t>() };
  size_t description_begin_idx{ ERR_VAL<size_t>() };
  size_t description_end_idx{ ERR_VAL<size_t>() };
  std::vector<size_t> hypernyms{};
  std::vector<size_t> hyponyms{};
  std::string word{};
  std::string description{};

  /** Set of indices that are hyponyms of this keyword. */
  std::unordered_set<size_t> hyponym_class_inidces{};

  /** Example images' filenames for last cached request. */
  std::vector<std::string> example_frames_filenames{};

  /** Unique hash determining the last caching call for frame examples. */
  size_t last_examples_hash = ERR_VAL<size_t>();
};

/**
 * Container representing one vocabulary.
 *
 * \see image_ranker::Keyword
 */
class [[nodiscard]] KeywordsContainer
{
  /*
   * Subtypes
   */
 private:
  /**
   * Hierarchical less than coparator.
   */
  struct [[nodiscard]] KeywordLessThanComparator
  {
    [[nodiscard]] bool operator()(const std::unique_ptr<Keyword>& a, const std::unique_ptr<Keyword>& b) const
    {
      // Force lowercase
      std::locale loc;
      std::string left;
      std::string right;

      for (auto elem : a->word)
      {
        left.push_back(std::tolower(elem, loc));
      }

      for (auto elem : b->word)
      {
        right.push_back(std::tolower(elem, loc));
      }

      auto result = left.compare(right);

      return result <= -1;
    }
  }
  kw_less_hierarch_cmp;

  /**
   * Alphabetical comparator.
   */
  struct [[nodiscard]] KeywordLessThanStringComparator
  {
    [[nodiscard]] bool operator()(const std::string& a, const std::string& b) const
    {
      // Compare strings

      std::string aa{ a };
      std::string bb{ b };
      std::transform(aa.begin(), aa.end(), aa.begin(),
                     [](char c) { return static_cast<char>(tolower(static_cast<int>(c))); });
      std::transform(bb.begin(), bb.end(), bb.begin(),
                     [](char c) { return static_cast<char>(tolower(static_cast<int>(c))); });

      auto result = aa.compare(bb);

      return result <= -1;
    }
  }
  kw_string_less_cmp;

 public:
  /** Not default-constructible */
  KeywordsContainer() = delete;

  /** Main VIRET ctor. */
  KeywordsContainer(const ViretDataPackRef::VocabData& vocab_data_refs);

  /** Main Google ctor. */
  KeywordsContainer(const GoogleDataPackRef::VocabData& vocab_data_refs);

  /** Main W2VV ctor. */
  KeywordsContainer(const W2vvDataPackRef::VocabData& vocab_data_refs);

  /** Parsees given file into this instance. */
  void parse_keywords(const std::string& filepath);

  /**
   * Gets string ID of this vocabulary.
   */
  [[nodiscard]] const std::string& get_ID() const { return _vocabulary_ID; }

  /**
   * Gets description of this vocabulary.
   */
  [[nodiscard]] const std::string& get_description() const { return _vocabulary_desc; }

  /**
   * Accesses keyword with the given ID.
   */
  [[nodiscard]] const Keyword& operator[](KeywordId keyword_ID) const { return *_ID_to_keyword.at(keyword_ID); }

  /**
   * Accesses keyword with the given ID.
   */
  [[nodiscard]] Keyword& operator[](KeywordId keyword_ID) { return *_ID_to_keyword.at(keyword_ID); }

  /**
   * Returns set of pointers to all keywords with this ID (synonyms).
   */
  [[nodiscard]] std::set<Keyword*>& get_all_keywords_ptrs(KeywordId keyword_ID)
  {
    return _ID_to_allkeywords.at(keyword_ID);
  }

  /**
   * Returns set of pointers to all keywords with this ID (synonyms).
   */
  [[nodiscard]] const Keyword* get_keyword_by_word(const std::string& word) const;
  [[nodiscard]] Keyword* get_keyword_by_word(const std::string& word);

  [[nodiscard]] const Keyword* get_keyword_by_wordnet_ID(size_t wordnetId) const;
  [[nodiscard]] Keyword* get_keyword_by_wordnet_ID(size_t wordnetId);

  [[nodiscard]] const Keyword* get_keyword_ptr_by_class_index(size_t index) const;
  [[nodiscard]] Keyword* get_keyword_by_class_index(size_t index);

  [[nodiscard]] std::vector<const Keyword*> get_near_keywords(const std::string& prefix, size_t numResults) const;
  [[nodiscard]] std::vector<Keyword*> get_near_keywords(const std::string& prefix, size_t numResults);

  [[nodiscard]] Keyword* desc_index_to_keyword(size_t descIndex) const;
  [[nodiscard]] std::string GetKeywordDescriptionByWordnetId(size_t wordnetId) const;

  /*!
   * Returns vector of keywords that are present in ranking vector of images.
   *
   * \return
   */
  [[nodiscard]] std::vector<size_t> get_classified_hyponyms_IDs(size_t wordnet_ID) const;

  /**
   * Converts CNF formula (containing keyword indices) into the readable string.
   *
   * \param fml   CNF formula with indices.
   * \return String representing the formula.
   */
  [[nodiscard]] std::string CNF_index_formula_to_string(const CnfFormula& fml) const;

  void get_keyword_hyponyms_indices_set(std::unordered_set<size_t> & dest_set, size_t wordnet_ID) const;
  void get_keyword_hyponyms_indices_set_nearest(std::unordered_set<size_t> & dest_set, size_t wordnet_ID,
                                                bool skip_pure_hypers = false) const;

  [[nodiscard]] size_t GetNetVectorSize() const { return class_idx_to_keyword.size(); }

  /*
   * Member variables
   */
 public:
  /** Supported keywords. */
  std::vector<std::unique_ptr<Keyword>> _keywords;

  /** Maps wordnetID to Keyword. */
  std::map<size_t, Keyword*> _wordnet_ID_to_keyword;

 private:
  std::string _vocabulary_ID;
  std::string _vocabulary_desc;
  std::string _kw_classes_fpth;

  std::map<KeywordId, Keyword*> _ID_to_keyword;
  std::map<KeywordId, std::set<Keyword*>> _ID_to_allkeywords;

  //! One huge string of all descriptions for fast keyword search
  std::string _allDescriptions;

  //! Maps index from probability vector to Keyword
  std::map<size_t, Keyword*> class_idx_to_keyword;

  std::vector<std::pair<size_t, Keyword*>> _desc_indext_to_keyword;
};
}  // namespace image_ranker