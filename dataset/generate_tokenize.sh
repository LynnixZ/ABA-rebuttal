#bash dataset/generate_tokenize.sh
python dataset/create_tokenize.py --fixed --min-len 1 --max-len 10 --target-len 20 --samples 100000 --output data/newtask/tokenize_fixed_train.txt
python dataset/create_tokenize.py --fixed --min-len 10 --max-len 20 --target-len 20 --samples 1000 --test --output data/newtask/tokenize_fixed_test_20.txt
python dataset/create_tokenize.py --fixed --min-len 1 --max-len 10 --target-len 20 --samples 1000 --test --output data/newtask/tokenize_fixed_test_10.txt
python dataset/create_tokenize.py --min-len 1 --max-len 10 --target-len 20 --samples 100000 --output data/newtask/tokenize_var_train.txt
python dataset/create_tokenize.py --min-len 10 --max-len 20 --target-len 0 --samples 1000 --test --output data/newtask/tokenize_var_test_20.txt
python dataset/create_tokenize.py --min-len 1 --max-len 10 --target-len 0 --samples 1000 --test --output data/newtask/tokenize_var_test_10.txt




    # 对于每个任务，对长度从 1 到 100 生成数据集
    for length in $(seq 1 50); do
        # 定义最终的输出文件路径，例如: data/string_evaluation/string_reverse/string_reverse_1.txt
        output_file="${output_dir}/$op/${op}_${length}.txt"
        
        echo "  生成长度为 ${length} 的数据集，保存到 ${output_file}"
        
        # 调用 Python 脚本生成数据
        # --min_string_length 和 --max_string_length 都设置为当前的循环变量 "length"
        # 以确保文件中的所有字符串都有完全相同的长度。
        python dataset/create_tokenize.py --fixed --min-len ${length} --max-len ${length} --target-len 20 --samples 500 --test --output data/newtask/eval/tokenize_fixed/tokenize_fixed_${length}.txt
        python dataset/create_tokenize.py --min-len ${length} --max-len ${length} --target-len 0 --samples 500 --test --output data/newtask/eval/tokenize_var/tokenize_var_${length}.txt

    done

 
echo
echo "========================================="
echo "所有固定长度的数据集生成完成。"
echo "文件保存在目录: $output_dir"