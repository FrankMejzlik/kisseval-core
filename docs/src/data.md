
\page data Data

What data will be loaded is specified inside JSON file `config/data_info.json` that should be placed at the location pointed to by variable `dataInfoFpth` in `config/config.json`. In the `data_info.json` there is flag `active` for each data pack that determines whether this data pack will be loaded or not.

Three data pack types are supported:
- `VIRET_based` - classification data from neural networks
- `Google_based` - classification done by Google Vision AI 
- `W2VV_based` - data for W2VV++ model

To run KISSEval on your data you need to do 2 things:
- Prepare imageset (\ref cust_imgset) - that is selected frames and their mapping.
- Prepare data files in correct formats (\ref cust_datapack) and add this data pack in the `config/data_info.json` file.





\section cust_imgset Adding custom imageset
Example imageset configuration in `config/data_info.json` looks like this:
@code{.json}
"imagesets": [
{
    "active": true,
    "ID": "V3C1_20k",
    "name": "V3C1_20k",
    "description": "Subset of V31C dataset, every 50th selected frame.",
    "ID_to_frame_fpth": "./imageset_V3C1_20k/files.txt",
    "frames_dir": "./imageset_V3C1_20k/frames/",
    "num_frames": 20000,
    "filename_vID_from": 1,
    "filename_vID_len": 5,
    "filename_sID_from": 8,
    "filename_sID_len": 5,
    "filename_fn_from": 40,
    "filename_fn_len": 6
}]
@endcode

### Data needed
You need frames (in a format that is displayable inside the HTML `img` tag) and mapping of their IDs to filenames (field `ID_to_frame_fpth`).

File pointed at by `ID_to_frame_fpth` value is just list of filenames where line number determines the ID (e.g. line 1 => ID 0). It can look like this:
@code{.json}
00000/v00000_s00000(f00000000-f00001321)_f00000031.jpg
00000/v00000_s00000(f00000000-f00001321)_f00000131.jpg
00000/v00000_s00000(f00000000-f00001321)_f00000294.jpg
...
@endcode

### Adding frames and their thumbnails to public KISSEval directory
In order to allow KISSEval system access the frames, you need to put them inside `public/images/<imageset_ID>` directory and their thumbnails (we reccomend size ~200px) into the `public/images/thumbs/<imageset_ID>`.

\note Please note that everything inside the `public` folder is accesible publically through HTTP requests! 





\section cust_datapack Adding custom data pack
Just prepare the files in formats specified below and add that record to the `config/data_config.json` file.

\section data_datapacks Supported data pack formats

\subsection data_viret_based Data of type 'VIRET_based'
Adding VIRET dataset in config looks like this:
@code{.json}
{
    "active": true,
    "ID": "NasNet2019",
    "name": "NasNet2019",
    "description": "VIRET_based",
    "model_options": "model=mult-sum-max;transform=linear_01;model_operations=mult-sum;model_ignore_treshold=0.0",
    "vocabulary": {
        "ID": "VIRET_WordNet2019",
        "description": "With ~1300 phrases.",
        "keyword_synsets_fpth": "./imageset_V3C1_20k/VIRET/keyword_classes.txt"
    },
    "data": {
        "target_dataset": "V3C1_20k",
        "presoftmax_scorings_fpth": "./imageset_V3C1_20k/VIRET/NasNet2019/Studenti_NasNetLarge.pre-softmax",
        "softmax_scorings_fpth": "",
        "deep_features_fpth": ""
    }
}
\endcode

In total we need only 2 data files, since the other two are now used as of now. We need
- **Vocabulary file** located at the `keyword_synsets_fpth` filepath.
- **Classification data file** located at the `presoftmax_scorings_fpth` filepath.

## File format of `keyword_synsets_fpth`
Text file with `~` separator. Each line is one keyword and columns are:
\code{.json}
[H or vector index]~[wordnet ID]~['#'-separated word representations]~['#'-separated hypernyms]~['#'-separated hyponyms]~[description]

\endcode

\code{.json}
.
.
.
H~07800091~feed#provender~7800740~~food for domestic livestock
H~07800740~fodder~7802026~~coarse food (especially for livestock) composed of entire plants or the leaves and stalks of a cereal crop
1009~07802026~hay~~7800740~780040#7820740~grass mowed and cured for use as fodder
.
.
.
\endcode

## File format of `presoftmax_scorings_fpth`
Binary file in format:
\code{.json}
|-----------------------------------------------------------|
|                36B header              |  int32 - vec_dim |
|-----------------------------------------------------------|
|  int32: frame_ID  | ... vec_dim x float32: frame_vec ...  |
|-----------------------------------------------------------|
                ... x num_frames
|-----------------------------------------------------------|
|  int32: frame_ID  | ... vec_dim x float32: frame_vec ...  |
|-----------------------------------------------------------|
\endcode










\subsection data_google_based Data of type 'Google_based'

### Example configuration in `data_info.json`
Adding Google Vision dataset in config looks like this:
\code{.json}
{
    "active": false,
    "ID": "GoogleVisionAi_Sep2019",
    "name": "GoogleVisionAi_Sep2019",
    "description": "Google_based",
    "model_options": "model=mult-sum-max;transform=linear_01;model_operations=mult-sum;model_ignore_treshold=0.0",
    "vocabulary": {
        "ID": "Google_AI_2019",
        "description": "With ~5000 phrases.",
        "keyword_synsets_fpth": "./imageset_V3C1_20k/Google/keyword_classes.google_vision_ai.txt"
    },
    "data": {
        "target_dataset": "V3C1_20k",
        "presoftmax_scorings_fpth": "./imageset_V3C1_20k/Google/extracted_sep_2019/scorings.google_vision_ai.bin"
    }
}
\endcode

We need only 2 data files, since the other two are now used as of now. We need
- **Vocabulary file** located at the `keyword_synsets_fpth` filepath.
- **Classification data file** located at the `presoftmax_scorings_fpth` filepath.



## File format of `keyword_synsets_fpth`
Text file with `~` separator. Each line is one keyword and columns are:
\code{.json}
[H or vector index]~[wordnet ID]~['#'-separated word representations]~['#'-separated hypernyms]~['#'-separated hyponyms]~[description]
\endcode

\code{.json}
.
.
.
H~07800091~feed#provender~7800740~~food for domestic livestock
H~07800740~fodder~7802026~~coarse food (especially for livestock) composed of entire plants or the leaves and stalks of a cereal crop
1009~07802026~hay~~7800740~780040#7820740~grass mowed and cured for use as fodder
.
.
.
\endcode


## File format of `presoftmax_scorings_fpth`
Binary file in format:
\code{.json}
|---------------------------------------------------------------------------|
|       int32: num_of_frames       |        int32: num_unique_keywords      |
|---------------------------------------------------------------------------|
| int32: frame_ID | int32: num_labels | ... num_labels x (kw_ID, score) ... |
|---------------------------------------------------------------------------|
                ... x num_frames
|---------------------------------------------------------------------------|
| int32: frame_ID | int32: num_labels | ... num_labels x (kw_ID, score) ... |
|---------------------------------------------------------------------------|
\endcode




\subsection data_w2vv_based Data of type 'W2VV_based'

Adding W2VV dataset in config looks like this:
\code{.json}
{
    "active": false,
    "ID": "W2VV_BoW_Dec2019",
    "name": "W2VV_BoW_Dec2019",
    "description": "W2VV_based",
    "model_options": "model=w2vv_bow_plain;model_sub_PCA_mean=true;",
    "vocabulary": {
        "ID": "native_language",
        "description": "Supported ~12k words.",
        "keyword_synsets_fpth": "./imageset_V3C1_20k/W2VV/W2VV_BoW_Dec2019/keyword_classes.bow2019.txt",
        "keyword_features_fpth": "./imageset_V3C1_20k/W2VV/W2VV_BoW_Dec2019/txt_weight-11147x2048floats.bin",
        "keyword_features_dim": 2048,
        "keyword_features_data_offset": 0,
        "keyword_bias_vec_fpth": "./imageset_V3C1_20k/W2VV/W2VV_BoW_Dec2019/txt_bias-2048floats.bin",
        "keyword_bias_vec_dim": 2048,
        "keyword_bias_vec_data_offset": 0,
        "keyword_PCA_mat_fpth": "./imageset_V3C1_20k/W2VV/W2VV_BoW_Dec2019/keyword_PCA_matrix.resnext152_resnext101.w2vv_bow.bin",
        "keyword_PCA_mat_dim": 2048,
        "keyword_PCA_mat_data_offset": 0,
        "keyword_PCA_mean_vec_fpth": "./imageset_V3C1_20k/W2VV/W2VV_BoW_Dec2019/keyword_PCA_mean.resnext152_resnext101.w2vv_bow.bin",
        "keyword_PCA_mean_vec_dim": 2048,
        "keyword_PCA_mean_vec_offset": 0
    },
    "data": {
        "target_dataset": "V3C1_20k",
        "deep_features_fpth": "./imageset_V3C1_20k/W2VV/W2VV_BoW_Dec2019/frame_features.resnext152_resnext101.w2vv_bow.bin",
        "deep_features_dim": 2048,
        "deep_features_data_offset": 0
    }
}
\endcode

We need only 6 data files, since the other two are now used as of now. We need
- **Vocabulary file** located at the `keyword_synsets_fpth` filepath.
- **Keyword vectors** located at the `keyword_features_fpth` filepath.
- **Keyword bias vector** located at the `keyword_bias_vec_fpth` filepath.
- **Keyword PCA matrix** located at the `keyword_PCA_mat_fpth` filepath.
- **Keyword PCA mean vector** located at the `keyword_PCA_mean_vec_fpth` filepath.
- **Frames' deep features** located at the `deep_features_fpth` filepath.


## File format of `keyword_synsets_fpth`
Text file with `~` separator. Each line is one keyword and columns are:
\code{.json}
[H or vector index]~[wordnet ID]~['#'-separated word representations]~['#'-separated hypernyms]~['#'-separated hyponyms]~[description]
\endcode

\code{.json}
.
.
.
H~07800091~feed#provender~7800740~~food for domestic livestock
H~07800740~fodder~7802026~~coarse food (especially for livestock) composed of entire plants or the leaves and stalks of a cereal crop
1009~07802026~hay~~7800740~780040#7820740~grass mowed and cured for use as fodder
.
.
.
\endcode


## File format of `keyword_features_fpth`
Binary row-wise matrix of float32 numbers with vector dimension specified. Data start at the provided offset.

## File format of `keyword_bias_vec_fpth`
Binary vector of float32 numbers with vector dimension specified. Data start at the provided offset.

## File format of `keyword_PCA_mat_fpth`
Binary row-wise matrix of float32 numbers with vector dimension specified. Data start at the provided offset.

## File format of `keyword_PCA_mean_vec_fpth`
Binary vector of float32 numbers with vector dimension specified. Data start at the provided offset.

## File format of `deep_features_fpth`
Binary row-wise matrix of float32 numbers with vector dimension specified. Data start at the provided offset.
