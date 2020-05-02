
#include "ViretDataPack.h"

#include <thread>

#include "./datasets/BaseImageset.h"
#include "./ranking_models/ranking_models.h"
#include "./transformations/transformations.h"

using namespace image_ranker;

ViretDataPack::ViretDataPack(const BaseImageset* p_is, const StringId& ID, const StringId& target_imageset_ID,
                             const std::string& model_options, const std::string& description,
                             const ViretDataPackRef::VocabData& vocab_data_refs,
                             std::vector<std::vector<float>>&& presoft, std::vector<std::vector<float>>&& softmax_data,
                             std::vector<std::vector<float>>&& feas_data)
    : BaseDataPack(p_is, ID, target_imageset_ID, model_options, description),
      _feas_data_raw(std::move(feas_data)),
      _presoftmax_data_raw(std::move(presoft)),
      _softmax_data_raw(std::move(softmax_data)),
      _keywords(vocab_data_refs)
{
  // Instantiate all wanted transforms
  std::thread t1([this]() {
    _transforms.emplace(enum_label(eTransformationIds::SOFTMAX).first,
                        std::make_unique<TransformationSoftmax>(_keywords, _softmax_data_raw));
  });
  std::thread t2([this]() {
    _transforms.emplace(enum_label(eTransformationIds::LINEAR_01).first,
                        std::make_unique<TransformationLinear01>(_keywords, _presoftmax_data_raw));
  });
  std::thread t3([this]() {
    _transforms.emplace(enum_label(eTransformationIds::NO_TRANSFORM).first,
                        std::make_unique<NoTransform>(_keywords, _presoftmax_data_raw));
  });

  // Instantiate all wanted models
  _models.emplace(enum_label(eModelIds::BOOLEAN).first, std::make_unique<BooleanModel>());
  _models.emplace(enum_label(eModelIds::VECTOR_SPACE).first, std::make_unique<VectorSpaceModel>());
  _models.emplace(enum_label(eModelIds::MULT_SUM_MAX).first, std::make_unique<MultSumMaxModel>());

  // Simulated user models
  _sim_users.emplace(enum_label(eSimUserIds::NO_SIM).first, std::make_unique<SimUserNoSim>());
  _sim_users.emplace(enum_label(eSimUserIds::USER_X_TO_P).first, std::make_unique<SimUserXToP>());

  t1.join();
  t2.join();
  t3.join();
}

const std::string& ViretDataPack::get_vocab_ID() const { return _keywords.get_ID(); }

const std::string& ViretDataPack::get_vocab_description() const { return _keywords.get_description(); }

[[nodiscard]] std::string ViretDataPack::humanize_and_query(const std::string& and_query) const
{
  LOGW("Not implemented!");

  return "I am just dummy query!"s;
}

[[nodiscard]] std::vector<Keyword*> ViretDataPack::top_frame_keywords(FrameId frame_ID,
                                                                      PackModelCommands model_commands,
                                                                      size_t count) const
{
  LOGW("Not implemented!");

  return std::vector({
      _keywords.GetKeywordPtrByVectorIndex(0),
      _keywords.GetKeywordPtrByVectorIndex(1),
      _keywords.GetKeywordPtrByVectorIndex(2),
  });
}

RankingResult ViretDataPack::rank_frames(const std::vector<CnfFormula>& user_queries, PackModelCommands model_commands,
                                         size_t result_size, FrameId target_image_ID) const
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

ModelTestResult ViretDataPack::test_model(const std::vector<UserTestQuery>& test_queries,
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

  // Choose desired simulated user
  auto iter_su = _sim_users.find(sim_user_ID);
  if (iter_su == _sim_users.end())
  {
    if (!_sim_users.empty())
    {
      iter_su = _sim_users.begin();
      LOGW("Uknown sim_user_ID: '" + sim_user_ID + "'. Falling back to: " + iter_su->first);
    }
    else
    {
      LOGE("No sim users!");
      return ModelTestResult{};
    }
  }
  const auto& sim_user = *(iter_su->second);

  // Process sim user
  // Use linear transform data without hypernyms accumulated
  const auto& sim_user_tr{ *(_transforms.find(enum_label(eTransformationIds::LINEAR_01).first)->second) };
  idx_test_queries =
      sim_user.process_sim_user(get_imageset_ptr(), sim_user_tr, _keywords, idx_test_queries, opt_key_vals);

  return ranking_model.test_model(transform, _keywords, idx_test_queries, opt_key_vals, num_points);
}

AutocompleteInputResult ViretDataPack::get_autocomplete_results(const std::string& query_prefix, size_t result_size,
                                                                bool with_example_images,
                                                                const std::string& model_commands) const
{
  auto kws = const_cast<const KeywordsContainer&>(_keywords).GetNearKeywordsPtrs(query_prefix, result_size);

  // Cache it up if needed
  cache_up_example_images(kws, model_commands);

  AutocompleteInputResult res{ kws };

  return res;
}

std::vector<const Keyword*> ViretDataPack::get_frame_top_classes(FrameId frame_ID,
                                                                 std::vector<ModelKeyValOption> opt_key_vals, bool accumulated) const
{
  bool max_based{ false };
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
      if (val == "mult-max" || val == "sum-max" || val == "max-max")
      {
        max_based = true;
      }
    }
  }

  // Choose desired transform
  auto iter_t = _transforms.find(transform_ID);
  if (iter_t == _transforms.end())
  {
    LOGE("Tranform not found!");
  }
  const auto& transform = *(iter_t->second);

  const DataInfo* p_data_info{nullptr};

  if (accumulated)
  {
    p_data_info =  max_based ? &transform.data_max_info() : &transform.data_sum_info();
  } 
  else {
    p_data_info = &transform.data_linear_raw_info();
  }

  const auto& kw_IDs{ p_data_info->top_classes[frame_ID] };


  std::vector<const Keyword*> res;

  std::transform(kw_IDs.begin(), kw_IDs.end(), std::back_inserter(res),
                 [this](const size_t& kw_ID) { return &_keywords[kw_ID]; });

  return res;
};

void ViretDataPack::cache_up_example_images(const std::vector<const Keyword*>& kws,
                                            const std::string& model_commands) const
{
  auto hasher{ std::hash<std::string>{} };

  size_t curr_opts_hash{ hasher(model_commands) };

  for (auto&& cp_kw : kws)
  {
   
    Keyword& kw{ _keywords[cp_kw->ID]};

    if (curr_opts_hash != kw.lastExampleFramesHash)
    {
      CnfFormula fml{ Clause{ Literal<KeywordId>{ kw.ID } } };

      std::vector<CnfFormula> v{ fml };

      // Rank frames with query "this_kw"
      auto ranked_frames{ rank_frames(v, model_commands, NUM_EXAMPLE_FRAMES) };

      kw.m_exampleImageFilenames.clear();
      for (auto&& f_ID : ranked_frames.m_frames)
      {
        std::string filename{ get_imageset_ptr()->operator[](f_ID).m_filename };
        kw.m_exampleImageFilenames.emplace_back(std::move(filename));
      }

      // Update hash
      kw.lastExampleFramesHash = curr_opts_hash;
    }
  }
}

DataPackInfo ViretDataPack::get_info() const
{
  return DataPackInfo{ get_ID(),           get_description(),          get_model_options(), target_imageset_ID(),
                       _keywords.get_ID(), _keywords.get_description() };
}

CnfFormula ViretDataPack::keyword_IDs_to_vector_indices(CnfFormula ID_query) const
{
  for (auto&& ID_clause : ID_query)
  {
    assert(ID_clause.size() == 1);

    KeywordId kw_ID{ ID_clause[0].atom };

    const Keyword& kw{ _keywords[kw_ID] };

    std::unordered_set<size_t> vecIds;
    _keywords.GetVectorKeywordsIndicesSetShallow(vecIds, kw.m_wordnetId);

    ID_clause.clear();
    for (auto&& id : vecIds)
    {
      ID_clause.emplace_back(Literal<KeywordId>{ id, false });
    }
  }
  return ID_query;
}

CnfFormula ViretDataPack::wordnetIDs_to_vector_indices(CnfFormula ID_query) const
{
  for (auto&& ID_clause : ID_query)
  {
    assert(ID_clause.size() == 1);

    KeywordId kw_ID{ ID_clause[0].atom };

    std::unordered_set<size_t> vecIds;
    _keywords.GetVectorKeywordsIndicesSetShallow(vecIds, kw_ID);

    ID_clause.clear();
    for (auto&& id : vecIds)
    {
      ID_clause.emplace_back(Literal<KeywordId>{ id, false });
    }
  }
  return ID_query;
}
