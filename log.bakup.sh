[32;1mRUN: /home/boren/.openai/bin/brix ssh dev-n1-hpe2 -- 'pgrep -fa /launch_'[0m
31570 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_seed_e1_simple_err_t12_n10_sc3k_ctx8_vllm_imp5.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n1-uks7 -- 'pgrep -fa /launch_'[0m
2649615 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_seed_e1_simple_err_t12_n10x2_sc3k_ctx8_imp5.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n2-hpe4 -- 'pgrep -fa /launch_'[0m
55744 python3 ./launch_train.py orng_conf/biasing/done/grpo_hcv2_fy22_m1000_seed_e1_simple_err_ga8_t12_n10x2_sc3k_ctx5.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n2-uks7 -- 'pgrep -fa /launch_'[0m
2828520 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e1_simple_err_t12_n10_sc3k_ctx8_inject_ref_entropy80_imp5.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n2-wus2 -- 'pgrep -fa /launch_'[0m
719808 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e3_simple_err_t12_n10_sc3k_ctx8_inject_ref_imp5.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n4-hpe4 -- 'pgrep -fa /launch_'[0m
55257 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e6_simple_err_t12_n10_sc3k_ctx8_inject_ref_imp5.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h1-n2-hpe4 -- 'pgrep -fa /launch_'[0m
15306 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_seed_e1_simple_char_err_t12_n10_sc3k_ctx8.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h1-n2-wus2 -- 'pgrep -fa /launch_'[0m
1619154 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_seed0_e1_simple_err_t12_n10_sc3k_ctx8_imp5.yaml
