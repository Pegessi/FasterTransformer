# From https://github.com/NVIDIA/FasterTransformer/pull/569 by AkiyamaYummy 

import argparse
import configparser
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def split_and_convert(args=None):
    #saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    #if os.path.exists(saved_dir) == False:
    #    os.makedirs(saved_dir)

    #factor = args.infer_gpu_num

    #dummy args
    factor = 1
    args = argparse.Namespace()
    args.in_file = "/root/autodl-tmp/model_scope/Llama-2-7b-hf"
    saved_dir = "/root/autodl-tmp/model_scope/Llama-2-7b-FT"
    args.weight_data_type = "fp16"
    args.model_name = "llama"
    model = LlamaForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)

    # Get tokenizer
    # from transformers import LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained(args.in_file)
    # get input_ids from tokenizer
    # tokenizer.encode("Hello this is a test")
    # testSeq= "510 1457 2816 28260 452 247 747 1481 25050 3110"
    # tokenizer.decode([int(x) for x in testSeq.split(" ")])


    np_weight_data_type = get_weight_data_type(args.weight_data_type)
    hidden_size = hf_config["hidden_size"]
    n_heads = hf_config["num_attention_heads"]
   

    try:
        model_name = args.model_name
        head_size = hf_config["hidden_size"] // n_heads
        # rotary_dim = int(head_size * hf_config["rotary_pct"])

        config = configparser.ConfigParser()
        config["llama"] = {}
        config["llama"]["model_name"] = model_name
        config["llama"]["head_num"] = str(n_heads)
        config["llama"]["size_per_head"] = str(head_size)
        config["llama"]["inter_size"] = str(hf_config["intermediate_size"])
        config["llama"]["num_layer"] = str(hf_config["num_hidden_layers"])
        config["llama"]["rotary_embedding"] = str(head_size)
        config["llama"]["vocab_size"] = str(hf_config["vocab_size"])
        config["llama"]["start_id"] = str(hf_config["bos_token_id"])
        config["llama"]["end_id"] = str(hf_config["eos_token_id"])
        config["llama"]["use_gptj_residual"] = str(int(False))
        config["llama"]["weight_data_type"] = args.weight_data_type


        with open((Path(saved_dir) / f"config.ini").as_posix(), "w") as configfile:
            config.write(configfile)
    except Exception as e:
        print(f"Fail to save the config in config.ini.", e)


    numHiddenLayers = hf_config["num_hidden_layers"]
    essentialParamsMap = {
        "model.embed_tokens.weight": "model.wte.weight.bin",
        "model.norm.weight": "model.final_layernorm.weight.bin",
        "lm_head.weight": "model.lm_head.weight.bin",
    }
    paramToArr = lambda param: param.detach().cpu().numpy().astype(np_weight_data_type)
    for key in essentialParamsMap:
        value = model.state_dict()[key]
        array = paramToArr(value)
        array.tofile(os.path.join(saved_dir,essentialParamsMap[key]))
    for hiddenLayerNum in range(numHiddenLayers):
      print("Processing layer", hiddenLayerNum)
      print("Processing attention")
      # Handle the qkv weights
      qkvArr = np.empty((hidden_size, 3, n_heads, head_size), dtype=np_weight_data_type)
      qArr = paramToArr(model.state_dict()[f"model.layers.{hiddenLayerNum}.self_attn.q_proj.weight"])
      # Hopefully this reshaping is correct... last two dims could also be swapped & need to be transposed
      qkvArr[:, 0, :, :] = qArr.reshape(hidden_size, n_heads, head_size)
      kArr = paramToArr(model.state_dict()[f"model.layers.{hiddenLayerNum}.self_attn.k_proj.weight"])
      qkvArr[:, 1, :, :] = kArr.reshape(hidden_size, n_heads, head_size)
      vArr = paramToArr(model.state_dict()[f"model.layers.{hiddenLayerNum}.self_attn.v_proj.weight"])
      qkvArr[:, 2, :, :] = vArr.reshape(hidden_size, n_heads, head_size)
      split_vals = np.split(qkvArr, factor, axis=-1)
      for j in range(factor):
          split_vals[j].tofile(os.path.join(saved_dir, f"model.layers.{hiddenLayerNum}.attention.query_key_value.weight.{j}.bin"))

      otherKeys = ["mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "input_layernorm.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight"]
      for key in otherKeys:
        print("Processing", key)
        value = model.state_dict()[f"model.layers.{hiddenLayerNum}.{key}"]
        array = paramToArr(value)
        if "layernorm" in key:
            array.tofile(os.path.join(saved_dir, f"model.layers.{hiddenLayerNum}.{key}.bin"))
        else:
          split_vals=None
          # Not sure if the split axies are correct here. o_proj is like attention.dense.weight. Down_proj is like mlp.dense_4h_to_h.weight
          if ("o_proj" in key) or ("down_proj" in key):
              split_vals = np.split(array, factor, axis=0)
          else:
              split_vals = np.split(array, factor, axis=-1)
          for j in range(factor):
              split_vals[j].tofile(os.path.join(saved_dir, f"model.layers.{hiddenLayerNum}.{key}.{j}.bin"))

if __name__ == "__main__":
  split_and_convert()

