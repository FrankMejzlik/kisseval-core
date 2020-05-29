
#include "GoogleVisionDataPack.h"

#include <thread>

#include "./datasets/BaseImageset.h"
#include "./ranking_models/ranking_models.h"
#include "./transformations/transformations.h"

using namespace image_ranker;

GoogleVisionDataPack::GoogleVisionDataPack(const BaseImageset* p_is, const StringId& ID,
                                           const StringId& target_imageset_ID, const std::string& model_options,
                                           const std::string& description, const DataPackStats& stats,
                                           const GoogleDataPackRef::VocabData& vocab_data_refs,
                                           std::vector<std::vector<float>>&& presoft)
    : BaseDataPack(p_is, ID, target_imageset_ID, model_options, description, stats),
      _keywords(vocab_data_refs),
      _presoftmax_data_raw(std::move(presoft))
{
  // Instantiate all wanted transforms

  std::thread t2([this]() {
    _transforms.emplace(enum_label(eTransformationIds::LINEAR_01).first,
                        std::make_unique<TransformationLinear01GoogleVision>(_keywords, _presoftmax_data_raw));
  });
  
  std::thread t3([this]() {
    _transforms.emplace(enum_label(eTransformationIds::SOFTMAX).first,
                        std::make_unique<TransformationSoftmaxGoogleVision>(_keywords, _presoftmax_data_raw));
  });

  // Instantiate all wanted models
  _models.emplace(enum_label(eModelIds::BOOLEAN).first, std::make_unique<BooleanModel>());
  _models.emplace(enum_label(eModelIds::VECTOR_SPACE).first, std::make_unique<VectorSpaceModel>());
  _models.emplace(enum_label(eModelIds::MULT_SUM_MAX).first, std::make_unique<MultSumMaxModel>());

  t2.join();
  t3.join();
}

void GoogleVisionDataPack::cache_up_example_images(const std::vector<const Keyword*>& kws,
                                                   const std::string& model_commands) const
{
  auto hasher{ std::hash<std::string>{} };

  size_t curr_opts_hash{ hasher(model_commands) };

  for (auto&& cp_kw : kws)
  {
    const auto& all_kws_with_ID{ _keywords.get_all_keywords_ptrs(cp_kw->ID) };

    // Check if images are already cached
    Keyword& kw{ **all_kws_with_ID.begin() };
    if (curr_opts_hash == kw.last_examples_hash)
    {
      continue;
    }

    // Get example images
    CnfFormula fml{ Clause{ Literal<KeywordId>{ kw.ID } } };
    std::vector<CnfFormula> v{ fml };

    // Rank frames with query "this_kw"
    auto ranked_frames{ rank_frames(v, model_commands, NUM_EXAMPLE_FRAMES) };

    // Store them in all keyword synonyms
    for (auto&& p_kw : all_kws_with_ID)
    {
      p_kw->example_frames_filenames.clear();
      for (auto&& f_ID : ranked_frames.m_frames)
      {
        std::string filename{ get_imageset_ptr()->operator[](f_ID).m_filename };
        p_kw->example_frames_filenames.emplace_back(std::move(filename));
      }

      // Update hash
      p_kw->last_examples_hash = curr_opts_hash;
    }
  }
}

std::vector<const Keyword*> GoogleVisionDataPack::get_frame_top_classes(
    FrameId frame_ID, const std::vector<ModelKeyValOption>& opt_key_vals, [[maybe_unused]] bool accumulated) const
{
  // Not yet used.
  //bool max_based{ false };
  std::string transform_ID{ "linear_01" };

  for (auto&& [key, val] : opt_key_vals)
  {
    if (key == "transform")
    {
      transform_ID = val;
    }
    else if (key == "model_operations")
    {
      // If max-based model
      // Not yet used.
      /*if (val == "mult-max" || val == "sum-max" || val == "max-max")
      {
        max_based = true;
      }*/
    }
  }

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Tranform not found!");
  }
  const auto& transform = *(iter_t->second);

  const DataInfo* p_data_info{ nullptr };

  p_data_info = &transform.data_sum_info();

  const auto& kw_IDs{ p_data_info->top_classes[frame_ID] };

  std::vector<const Keyword*> res;

  std::transform(kw_IDs.begin(), kw_IDs.end(), std::back_inserter(res),
                 [this](const size_t& kw_ID) { return &_keywords[kw_ID]; });

  return res;
};

const std::string& GoogleVisionDataPack::get_vocab_ID() const { return _keywords.get_ID(); }

const std::string& GoogleVisionDataPack::get_vocab_description() const { return _keywords.get_description(); }

RankingResult GoogleVisionDataPack::rank_frames(const std::vector<CnfFormula>& user_queries,
                                                const std::string& model_commands, size_t result_size,
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
    PROD_THROW("Data error.")
  }
  const auto& ranking_model = *(iter_m->second);

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Uknown transform_ID: '" + transform_ID + "'.");
    PROD_THROW("Data error.")
  }
  const auto& transform = *(iter_t->second);

  // Run this model
  return ranking_model.rank_frames(transform, _keywords, idx_queries, result_size, opt_key_vals, target_image_ID);
}

ModelTestResult GoogleVisionDataPack::test_model(const std::vector<UserTestQuery>& test_queries,
                                                 const std::string& model_commands, size_t num_points) const
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
      PROD_THROW("Data error.")
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
      PROD_THROW("Data error.")
    }
  }
  const auto& transform = *(iter_t->second);

  return ranking_model.test_model(transform, _keywords, idx_test_queries, opt_key_vals, num_points);
}

AutocompleteInputResult GoogleVisionDataPack::get_autocomplete_results(const std::string& query_prefix,
                                                                       size_t result_size,
                                                                       [[maybe_unused]] bool with_example_image,
                                                                       const std::string& model_commands) const
{
  auto kws = _keywords.get_near_keywords(query_prefix, result_size);

  std::vector<const Keyword*> res_kws;

  auto hasher{ std::hash<std::string>{} };

  size_t curr_opts_hash{ hasher(model_commands) };

  for (auto&& p_kw : kws)
  {
    if (curr_opts_hash != p_kw->last_examples_hash)
    {
      CnfFormula fml{ Clause{ Literal<KeywordId>{ p_kw->ID } } };

      std::vector<CnfFormula> v{ fml };

      // Rank frames with query "this_kw"
      auto ranked_frames{ rank_frames(v, model_commands, NUM_EXAMPLE_FRAMES) };

      p_kw->example_frames_filenames.clear();
      for (auto&& f_ID : ranked_frames.m_frames)
      {
        std::string filename{ get_imageset_ptr()->operator[](f_ID).m_filename };
        p_kw->example_frames_filenames.emplace_back(std::move(filename));
      }

      // Update hash
      p_kw->last_examples_hash = curr_opts_hash;
    }

    res_kws.emplace_back(p_kw);
  }

  AutocompleteInputResult res{ res_kws };

  return res;
}

DataPackInfo GoogleVisionDataPack::get_info() const
{
  return DataPackInfo{ get_ID(),
                       get_description(),
                       get_model_options(),
                       target_imageset_ID(),
                       _presoftmax_data_raw.size(),
                       _keywords.get_ID(),
                       _keywords.get_description() };
}

CnfFormula GoogleVisionDataPack::keyword_IDs_to_vector_indices(CnfFormula ID_query)
{
  // Google vocabulary has no hypernyms
  return ID_query;
}
