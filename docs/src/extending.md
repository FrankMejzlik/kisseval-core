@page extending Extending library

@section add_data_pack Using custom data packs
To use custom data packs in already supported format, you need to prepare data files in specific formats (as described in \ref Data). Then just add new record into `data_config.json` file that you pass in as parameter when instantiating \ref image_ranker::ImageRanker. You specify paths to those files in this config and state it as active. Ranker will then load it during instantiation and you are good to go.

If you need to know more about how data packs are loaded, please see source files. More specificaly, focus on their constructors where data is passed in.

Points of interest are:
- \ref image_ranker::ViretDataPack
- \ref image_ranker::GoogleVisionDataPack
- \ref image_ranker::W2vvDataPack


\section add_data_pack Add new type of data packs
You need to add container into \ref `image_ranker::ImageRanker` class that will store them. To do this, please use any of the existing data packs as a reference and do it similarly. Comments in code will help you with that.

Points of interest are:
- \ref image_ranker::ViretDataPack
- \ref image_ranker::GoogleVisionDataPack
- \ref image_ranker::W2vvDataPack

\section add_models Add new models
Please use existing models as a reference for creating the new one. All of them inherit from specific base class that enforces implementation of some API that is necessary for given model. Also bear in mind that models for specific data packs are bit different.

Points of interest are:
- \ref image_ranker::BooleanModel
- \ref image_ranker::VectorSpaceModel
- \ref image_ranker::MultSumMaxModel
- \ref image_ranker::PlainBowModel