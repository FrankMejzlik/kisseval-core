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
#include <windows.h>

#include "config.h"

class ImagesContainer;

class ImageRanker
{
  // Structures
public:
  using Buffer = std::vector<std::byte>;


  // Methods
public:
  ImageRanker() = delete;
  ImageRanker(
    std::string_view probabilityVectorFilepath,
    std::string_view deepFeaturesFilepath,
    std::string_view keywordClassesFilepath
  );
  

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
};