
#include "GoogleVisionDataPack.h"

#include <thread>

#include "./datasets/BaseImageset.h"
#include "./ranking_models/ranking_models.h"
#include "./transformations/transformations.h"

using namespace image_ranker;

GoogleVisionDataPack::GoogleVisionDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID,
                                           const std::string& model_options, const std::string& description,
                                           const GoogleDataPackRef::VocabData& vocab_data_refs,
                                           std::vector<std::vector<float>>&& presoft)
    : BaseDataPack(p_is, ID, target_imageset_ID, model_options, description),
      _presoftmax_data_raw(std::move(presoft)),
      _keywords(vocab_data_refs)
{
  // Instantiate all wanted transforms

  std::thread t2([this]() {
    _transforms.emplace(enum_label(eTransformationIds::LINEAR_01).first,
                        std::make_unique<TransformationLinear01GoogleVision>(_keywords, _presoftmax_data_raw));
  });
  std::thread t3([this]() {
    _transforms.emplace(enum_label(eTransformationIds::NO_TRANSFORM).first,
                        std::make_unique<NoTransformGoogleVision>(_keywords, _presoftmax_data_raw));
  });

  // Instantiate all wanted models
  _models.emplace(enum_label(eModelIds::BOOLEAN).first, std::make_unique<BooleanModel>());
  _models.emplace(enum_label(eModelIds::VECTOR_SPACE).first, std::make_unique<VectorSpaceModel>());
  _models.emplace(enum_label(eModelIds::MULT_SUM_MAX).first, std::make_unique<MultSumMaxModel>());

  t2.join();
  t3.join();
}

const std::string& GoogleVisionDataPack::get_vocab_ID() const { return _keywords.get_ID(); }

const std::string& GoogleVisionDataPack::get_vocab_description() const { return _keywords.get_description(); }

std::string GoogleVisionDataPack::humanize_and_query(const std::string& and_query) const
{
  LOGW("Not implemented!");

  return "I am just dummy query!"s;
}

std::vector<Keyword*> GoogleVisionDataPack::top_frame_keywords(FrameId frame_ID, PackModelCommands model_commands, size_t count) const
{
  LOGW("Not implemented!");

  return std::vector({
      _keywords.GetKeywordPtrByVectorIndex(0),
      _keywords.GetKeywordPtrByVectorIndex(1),
      _keywords.GetKeywordPtrByVectorIndex(2),
  });
}

RankingResult GoogleVisionDataPack::rank_frames(const std::vector<CnfFormula>& user_queries,
                                                PackModelCommands model_commands, size_t result_size,
                                                FrameId target_image_ID) const
{
  // Expand query to vector indices
  std::vector<CnfFormula> idx_queries;
  idx_queries.reserve(user_queries.size());
  for (auto&& q : user_queries)
  {
    idx_queries.emplace_back(keyword_IDs_to_vector_indices(q));
  }

  std::vector<std::string> tokens = split(model_commands, ';');

  std::vector<ModelKeyValOption> opt_key_vals;

  std::string model_ID;
  std::string transform_ID;

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');

    // Model ID && Transform ID
    if (key_val[0] == enum_label(eModelOptsKeys::MODEL_ID).first)
    {
      model_ID = key_val[1];
    }
    else if (key_val[0] == enum_label(eModelOptsKeys::TRANSFORM_ID).first)
    {
      transform_ID = key_val[1];
    }
    // Options for model itself
    else
    {
      opt_key_vals.emplace_back(key_val[0], key_val[1]);
    }
  }

  // Choose desired model
  auto iter_m = _models.find(model_ID);
  if (iter_m == _models.end())
  {
    LOGE("Uknown model_ID: '" + model_ID + "'.");
    return RankingResult{};
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Uknown transform_ID: '" + transform_ID + "'.");
    return RankingResult{};
  }
  const auto& transform = *(iter_t->second);

  // Run this model
  return ranking_model.rank_frames(transform, _keywords, idx_queries, result_size, opt_key_vals, target_image_ID);
}

ModelTestResult GoogleVisionDataPack::test_model(const std::vector<UserTestQuery>& test_queries,
                                                 PackModelCommands model_commands, size_t num_points) const
{
  // Expand query to vector indices
  std::vector<UserTestQuery> idx_test_queries;
  idx_test_queries.reserve(test_queries.size());

  for (auto&& [test_query, target_ID] : test_queries)
  {
    std::vector<CnfFormula> idx_query;
    idx_query.reserve(test_query.size());
    for (auto&& q : test_query)
    {
      idx_query.emplace_back(keyword_IDs_to_vector_indices(q));
    }
    idx_test_queries.emplace_back(std::move(idx_query), target_ID);
  }

  // Parse model & transformation
  std::vector<std::string> tokens = split(model_commands, ';');

  std::vector<ModelKeyValOption> opt_key_vals;

  std::string model_ID;
  std::string transform_ID;
  std::string sim_user_ID;

  for (auto&& tok : tokens)
  {
    auto key_val = split(tok, '=');

    // Model ID && Transform ID
    if (key_val[0] == enum_label(eModelOptsKeys::MODEL_ID).first)
    {
      model_ID = key_val[1];
    }
    else if (key_val[0] == enum_label(eModelOptsKeys::TRANSFORM_ID).first)
    {
      transform_ID = key_val[1];
    }
    else if (key_val[0] == enum_label(eModelOptsKeys::SIM_USER_ID).first)
    {
      sim_user_ID = key_val[1];
    }
    // Options for model itself
    else
    {
      opt_key_vals.emplace_back(key_val[0], key_val[1]);
    }
  }

  // Choose desired model
  auto iter_m = _models.find(model_ID);
  if (iter_m == _models.end())
  {
    if (!_models.empty())
    {
      iter_m = _models.begin();
      LOGW("Uknown model_ID: '" + model_ID + "'. Falling back to: " + iter_m->first);
    }
    else
    {
      LOGE("Uknown models!");
      return ModelTestResult{};
    }
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    if (!_transforms.empty())
    {
      iter_t = _transforms.begin();
      LOGW("Uknown transform_ID: '" + transform_ID + "'. Falling back to: " + iter_t->first);
    }
    else
    {
      LOGE("No transformations!");
      return ModelTestResult{};
    }
  }
  const auto& transform = *(iter_t->second);

  return ranking_model.test_model(transform, _keywords, idx_test_queries, opt_key_vals, num_points);
}

AutocompleteInputResult GoogleVisionDataPack::get_autocomplete_results(const std::string& query_prefix,
                                                                       size_t result_size,
                                                                       bool with_example_image, PackModelCommands model_commands) const
{
  auto kws = _keywords.GetNearKeywordsPtrs(query_prefix, result_size);

  std::vector<const Keyword*> res_kws;

  auto hasher{ std::hash<std::string>{} };

  size_t curr_opts_hash{ hasher(model_commands) };

  for (auto&& p_kw : kws)
  {
    if (curr_opts_hash != p_kw->lastExampleFramesHash)
    {
      CnfFormula fml{ Clause{ Literal<KeywordId>{ p_kw->ID } } };

      std::vector<CnfFormula> v{ fml  };

      // Rank frames with query "this_kw"
      auto ranked_frames{ rank_frames(v, model_commands, NUM_EXAMPLE_FRAMES) };

      p_kw->m_exampleImageFilenames.clear();
      for (auto&& f_ID : ranked_frames.m_frames)
      {
        std::string filename{ get_imageset_ptr()->operator[](f_ID).m_filename };
        p_kw->m_exampleImageFilenames.emplace_back(std::move(filename));
      }

      // Update hash
      p_kw->lastExampleFramesHash = curr_opts_hash;
    }

    res_kws.emplace_back(p_kw);
  }

  AutocompleteInputResult res{ res_kws };

  return res;
}

DataPackInfo GoogleVisionDataPack::get_info() const
{
  return DataPackInfo{get_ID(),           get_description(),          get_model_options(), target_imageset_ID(),
                      _keywords.get_ID(), _keywords.get_description()};
}

CnfFormula GoogleVisionDataPack::keyword_IDs_to_vector_indices(CnfFormula ID_query) const
{
  // Google vocabulary has no hypernyms
  return ID_query;
}
