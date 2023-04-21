/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/llama/LlamaDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(const int  hidden_units,
                                                        const int  inter_size,
                                                        const int  tensor_para_size,
                                                        const int  tensor_para_rank,
                                                        const bool use_gptj_residual):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    use_gptj_residual_(use_gptj_residual)
{
    mallocWeights();
    setWeightPtr();
}

template<typename T>
LlamaDecoderLayerWeight<T>::~LlamaDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 7; i++) {
            // if (!use_gptj_residual_ && i != attention_dense_bias_weight_id) {
                cudaFree(weights_ptr[i]);
            // }
        }

        pre_layernorm_weights.beta                            = nullptr;
        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;
        post_attention_layernorm_weights.beta                 = nullptr;
        post_attention_layernorm_weights.gamma                = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias   = nullptr;
        ffn_weights.intermediate_weight2.kernel = nullptr;
        ffn_weights.intermediate_weight2.bias   = nullptr;
        ffn_weights.output_weight.kernel       = nullptr;
        ffn_weights.output_weight.bias         = nullptr;
        is_maintain_buffer                     = false;
    }
}

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(const LlamaDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    use_gptj_residual_(other.use_gptj_residual_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
    // if (!use_gptj_residual_) {
    //     cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    // }

    // kernel of intermediate1, intermediate2, output
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], inter_size_ / tensor_para_size_ * hidden_units_);

    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    setWeightPtr();
}

template<typename T>
LlamaDecoderLayerWeight<T>& LlamaDecoderLayerWeight<T>::operator=(const LlamaDecoderLayerWeight& other)
{
    hidden_units_      = other.hidden_units_;
    inter_size_        = other.inter_size_;
    tensor_para_size_  = other.tensor_para_size_;
    tensor_para_rank_  = other.tensor_para_rank_;
    use_gptj_residual_ = other.use_gptj_residual_;

    mallocWeights();

    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);

    // kernel of intermediate1, intermediate2, output
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[4], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], inter_size_ / tensor_para_size_ * hidden_units_);

    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    setWeightPtr();
    return *this;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);
    const std::string rank_spec = std::to_string(tensor_para_rank_);

    loadWeightFromBin<T>(
        weights_ptr[0], {(size_t)hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1],
                         {(size_t)hidden_units_, (size_t)(3 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin",
                         model_file_type);

    loadWeightFromBin<T>(weights_ptr[2],
                         {(size_t)(hidden_units_ / tensor_para_size_), (size_t)hidden_units_},
                         dir_path + ".self_attn.o_proj.weight." + rank_spec + ".bin",
                         model_file_type);

    loadWeightFromBin<T>(weights_ptr[3],
                         {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[4],
                         {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5],
                         {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                         dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                         model_file_type);

    loadWeightFromBin<T>(
        weights_ptr[6], {(size_t)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);
}

template<typename T>
void LlamaDecoderLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta                            = nullptr;
    pre_layernorm_weights.gamma                           = weights_ptr[0];
    self_attention_weights.query_weight.kernel            = weights_ptr[1];
    self_attention_weights.query_weight.bias              = nullptr;
    self_attention_weights.attention_output_weight.kernel = weights_ptr[2];
    self_attention_weights.attention_output_weight.bias   = nullptr;

    ffn_weights.intermediate_weight.kernel = weights_ptr[3];
    ffn_weights.intermediate_weight.bias   = nullptr;
    ffn_weights.intermediate_weight2.kernel = weights_ptr[4];
    ffn_weights.intermediate_weight2.bias   = nullptr;
    ffn_weights.output_weight.kernel       = weights_ptr[5];
    ffn_weights.output_weight.bias         = nullptr;

    post_attention_layernorm_weights.beta  = nullptr;
    post_attention_layernorm_weights.gamma = weights_ptr[6];
    is_maintain_buffer                     = true;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);

    deviceMalloc(&weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[4], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[5], inter_size_ / tensor_para_size_ * hidden_units_);

    deviceMalloc(&weights_ptr[6], hidden_units_);
}

template struct LlamaDecoderLayerWeight<float>;
template struct LlamaDecoderLayerWeight<half>;

}  // namespace fastertransformer
