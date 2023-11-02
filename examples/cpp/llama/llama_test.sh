#!/bin/bash
bs_list=(2 4 8)
len_list=(128 256 512 1024 2048 4096)

data_file=/root/autodl-tmp/llm_workspace/FasterTransformer/examples/cpp/gptneox/start_ids.csv
config_file=/root/autodl-tmp/llm_workspace/FasterTransformer/examples/cpp/llama/llama_config.ini
# config_content=$(<$config_file)
# echo $config_content > $config_file


for bs in "${bs_list[@]}"
do
    for len in "${len_list[@]}"
    do
        echo "$bs, $len, $len"
        data_str=""
        for ((i=1; i<=$len; i++))
        do
            data_str="${data_str}666,"
        done
        echo $data_str > $data_file
        # query
        bs_text="request_batch_size=$bs # determine by the request"
        # out_len=$((${len}+1))
        out_len=1
        out_text="request_output_len=$out_len # determine by the request"
        sed -i "31s/.*/${bs_text}/" $config_file
        sed -i "32s/.*/${out_text}/" $config_file
        ./bin/llama_example &
        wait $!
        # answer
        bs_text="request_batch_size=$bs # determine by the request"
        out_len=$((${len}+1))
        out_text="request_output_len=$out_len # determine by the request"
        sed -i "31s/.*/${bs_text}/" $config_file
        sed -i "32s/.*/${out_text}/" $config_file
        ./bin/llama_example &
        wait $!
        # echo "test" > $data_file
    done
done

# ./bin/llama_example