// HelloWorld.cpp : Defines the entry point for the application.
//

#include <fstream>
#include <iostream>

#include "sciter-x.h"
#include "sciter-x-window.hpp"

#include "Database.h"
#include "ImageRanker.h"


class frame: public sciter::window {
public:
  frame() :
    window(SW_TITLEBAR | SW_RESIZEABLE | SW_CONTROLS | SW_MAIN | SW_ENABLE_DEBUG),
    _currentImageIndex(0ULL),
    _isInitialized(false),
    _imageRanker(
      IMAGES_PATH,
      DATA_PATH SOFTMAX_BIN_FILENAME,
      DATA_PATH DEEP_FEATURES_FILENAME,
      DATA_PATH KEYWORD_CLASSES_FILENAME
    )
  {
    //size_t result = db.NoResultQuery("INSERT INTO images (id, filename) VALUES (13, 'fuga'), (14, 'fugaa');");


  #if 0

    // Test GetNearKeywords()
    for (size_t i = 0ULL; i < 100LL; ++i)
    {
      std::string prefix = "do";


      auto sugg = _imageRanker.GetNearKeywords(prefix);

      for (auto&& tuple : sugg)
      {
        size_t id = std::get<0>(tuple);
        std::string word = std::get<1>(tuple);
        std::string desc = std::get<2>(tuple);
      }

    }
  #endif

  #if GENERATE_COLLECTOR_INPUT_DATA

    //// Generate ImageId => ImgFilename data for collector
    //GenerateImgToSrcDataForCollector(_softmaxData);
    //
    //// Generate VectorIndex => Description data for collector
    //GenerateVecIndexToDescriptionDataForCollector(_indexToClassnames, _hypernysmData);



  #endif

  }

  // map of native functions exposed to script:
  BEGIN_FUNCTION_MAP
    FUNCTION_0("GetRandomImage", GetRandomImage);
    FUNCTION_0("GetNextImage", GetNextImage);
    FUNCTION_0("GetPrevImage", GetPrevImage);
  END_FUNCTION_MAP


  bool GenerateImgToSrcDataForCollector(std::vector< std::pair< size_t, std::unordered_map<size_t, float> > > inputMap)
  {
    // Open file for writing
    std::ofstream outFile(COLLECTOR_INPUT_OUTPUT_DATA_PATH "images.in", std::ios::out);

    // Start out JSON file
    outFile << "{" << std::endl;
    outFile << "\t \"idSrcPairs\": [" << std::endl;

    size_t counter = inputMap.size();
    size_t i = 0ULL;

    for (auto picPair : inputMap)
    {
      size_t fileStructureIndex = picPair.first / INDEX_OFFSET;

      // Get filepath
      std::string filepath = _imageRanker.GetImageFilepathByIndex(fileStructureIndex, true);
      
      outFile << "\t\t{" << std::endl;
      
      outFile << "\t\t\t \"id\": \"" << std::to_string(picPair.first) <<"\"," << std::endl;
      outFile << "\t\t\t \"src\": \"" << filepath << "\"" << std::endl;

      outFile << "\t\t }";

      // If not last, put comma there
      if (i < counter - 1)
      {
        outFile << ",";
      }

      outFile << std::endl;

      ++i;
    }



    // End JSON
    outFile << "\t ]" << std::endl;
    outFile << "}" << std::endl;

    // Close file
    outFile.close();

    return true;
  }


  bool GenerateVecIndexToDescriptionDataForCollector(
    std::unordered_map<size_t, std::pair<size_t, std::string> > inputMap, 
    std::unordered_map<size_t, std::pair<size_t, std::string> > inputMapHyper
  )
  {
    // Open file for writing
    std::ofstream outFile(COLLECTOR_INPUT_OUTPUT_DATA_PATH "vecIndexToDescription.in", std::ios::out);
    

    // Start out JSON file
    outFile << "{" << std::endl;
    outFile << "\t \"vecIndexToDescription\": [" << std::endl;

    size_t counter = inputMap.size();
    size_t i = 0ULL;

    for (auto picPair : inputMap)
    {
      size_t vectorIndex = picPair.first;
      std::string_view description = picPair.second.second;
      size_t wordnetId = picPair.second.first;

      outFile << "\t\t {" << std::endl;

      outFile << "\t\t\t \"vecIndex\": \"" << std::to_string(vectorIndex) <<"\"," << std::endl;
      outFile << "\t\t\t \"description\": \"" << description.data() << "\"," << std::endl;
      outFile << "\t\t\t \"wordnetId\": \"" << std::to_string(wordnetId) << "\"" << std::endl;

      outFile << "\t\t },";

      //// If not last, put comma there
      //if (i < counter - 1)
      //{
      //  outFile << ",";
      //}

      outFile << std::endl;

      ++i;
    }


    counter = inputMapHyper.size();
    i = 0ULL;
    for (auto picPair : inputMapHyper)
    {
      size_t vectorIndex = picPair.first;
      std::string_view description = picPair.second.second;
      size_t wordnetId = picPair.second.first;

      outFile << "\t\t {" << std::endl;

      outFile << "\t\t\t \"vecIndex\": \"" << std::to_string(0) <<"\"," << std::endl;
      outFile << "\t\t\t \"description\": \"" << description.data() << "\"," << std::endl;
      outFile << "\t\t\t \"wordnetId\": \"" << std::to_string(wordnetId) << "\"" << std::endl;

      outFile << "\t\t }";

      // If not last, put comma there
      if (i < counter - 1)
      {
        outFile << ",";
      }

      outFile << std::endl;

      ++i;
    }



    // End JSON
    outFile << "\t ]" << std::endl;
    outFile << "}" << std::endl;

    // Close file
    outFile.close();

    return true;
  }


  sciter::value GetRandomImage() const
  { 
    // Get random index
    const size_t index = static_cast<size_t>(_imageRanker.GetRandomInteger(0, NUM_ROWS) * INDEX_OFFSET);

    // Get index that correcponds to index in this dataset file structure
    size_t fileStructureIndex = index / INDEX_OFFSET;

    // Get data 
    auto&& imageProbabilitesIt = _softmaxData[fileStructureIndex];

    // Get map of probabilities
    std::unordered_map<size_t, float> probabilities = imageProbabilitesIt.second;

    
    // Get name of file
    std::string filepath = _imageRanker.GetImageFilepathByIndex(fileStructureIndex);


    // Construct JSON data file for TIScript
    sciter::value retval;

    // Insert data values
    retval.make_map();
    retval.set_item("id", std::to_string(index));
    retval.set_item("filepath", filepath);

    sciter::value probabilityLevel;
    probabilityLevel.make_map();

    // Iterate through all probabilities
    for (auto&& probPair : probabilities)
    {
      // Get vector index
      size_t vectorIndex = probPair.first;

      // Get probability of this index
      float probability = probPair.second;

      // Get classname table record for this index
      auto&& classnameData = _indexToClassnames.find(vectorIndex)->second;

      // Get  classname for this index;
      std::string className = classnameData.second;
      
      // Wordnet id
      size_t wordnetId = classnameData.first;

      // Construct JSON probability data level 
      sciter::value probDataLevel;
      {
        probDataLevel.make_map();

        probDataLevel.set_item("vectorIndex", std::to_string(vectorIndex));
        probDataLevel.set_item("wordnetId", std::to_string(wordnetId));
        probDataLevel.set_item("probabilityValue", std::to_string(probability));
        probDataLevel.set_item("className", className);
      }

      probabilityLevel.set_item(std::to_string(vectorIndex), probDataLevel);
    }

    // Insert this level to JSON
    retval.set_item("pro", probabilityLevel);

    return retval;
  }
  
  sciter::value GetImage(size_t index)
  {
    // Get data 
    auto&& imageProbabilites = _softmaxData[_currentImageIndex];

    // Get index that correcponds to index in this dataset file structure
    size_t fileStructureIndex = _currentImageIndex / INDEX_OFFSET;

    // Get data 
    auto&& imageProbabilitesIt = _softmaxData[fileStructureIndex];

    // Get map of probabilities
    std::unordered_map<size_t, float> probabilities = imageProbabilitesIt.second;

    // Get name of file
    std::string filepath = _imageRanker.GetImageFilepathByIndex(fileStructureIndex);

    // Construct JSON data file for TIScript
    sciter::value retval;

    // Insert data values
    retval.make_map();
    retval.set_item("id", std::to_string(index));
    retval.set_item("filepath", filepath);

    sciter::value probabilityLevel;
    probabilityLevel.make_map();

    // Iterate through all probabilities
    for (auto&& probPair : probabilities)
    {
      // Get vector index
      size_t vectorIndex = probPair.first;

      // Get probability of this index
      float probability = probPair.second;

      // Get classname table record for this index
      auto&& classnameData = _indexToClassnames.find(vectorIndex)->second;

      // Get  classname for this index;
      std::string className = classnameData.second;

      // Wordnet id
      size_t wordnetId = classnameData.first;

      // Construct JSON probability data level 
      sciter::value probDataLevel;
      {
        probDataLevel.make_map();

        probDataLevel.set_item("vectorIndex", std::to_string(vectorIndex));
        probDataLevel.set_item("wordnetId", std::to_string(wordnetId));
        probDataLevel.set_item("probabilityValue", std::to_string(probability));
        probDataLevel.set_item("className", className);
      }

      probabilityLevel.set_item(std::to_string(vectorIndex), probDataLevel);
    }
    // Insert this level to JSON
    retval.set_item("pro", probabilityLevel);

    return retval;
  }

  sciter::value GetNextImage()
  {
    if (!_isInitialized)
    {
      _currentImageIndex = 0ULL;
      _isInitialized = true;
    }

    if (_currentImageIndex <= (NUM_ROWS * INDEX_OFFSET) - 50ULL)
    {
      _currentImageIndex += 50ULL;
    }

    return GetImage(_currentImageIndex);
  }

  sciter::value GetPrevImage()
  {
    if (!_isInitialized)
    {
      _currentImageIndex = 0ULL;
      _isInitialized = true;
    }

    if (_currentImageIndex >= 50)
    {
      _currentImageIndex -= 50ULL;
    }


    // Get next
    return GetImage(_currentImageIndex);
  }
  


private:
  bool _isInitialized;
  size_t _currentImageIndex;


  std::vector< std::pair< size_t, std::unordered_map<size_t, float> > > _softmaxData;
  std::vector< std::pair< size_t, std::vector<float> > > _deepFeaturesData;
  std::unordered_map<size_t, std::pair<size_t, std::string> > _indexToClassnames;
  std::unordered_map<size_t, std::pair<size_t, std::string> >_hypernysmData;

  ImageRanker _imageRanker;
};

#include "resources.cpp" // resources packaged into binary blob.

int uimain(std::function<int()> run ) {

  // enable features you may need in your scripts:
  SciterSetOption(NULL, SCITER_SET_SCRIPT_RUNTIME_FEATURES,
    ALLOW_FILE_IO |
    ALLOW_SOCKET_IO | // NOTE: the must for communication with Inspector
    ALLOW_EVAL |
    ALLOW_SYSINFO);

  sciter::archive::instance().open(aux::elements_of(resources)); // bind resources[] (defined in "resources.cpp") with the archive

  aux::asset_ptr<frame> pwin = new frame();

  // note: this:://app URL is dedicated to the sciter::archive content associated with the application
  pwin->load( WSTR("this://app/main.htm") );
  //or use this to load UI from  
  //  pwin->load( WSTR("file:///home/andrew/Desktop/Project/res/main.htm") );

  pwin->expand();

  return run();
}