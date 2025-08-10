[32;1mRUN: /home/boren/.openai/bin/brix ssh dev-n1-hpe2 -- 'pgrep -fa /launch_'[0m
42409 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8_inject_ref_entropy80.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh dev-n1-hpe5 -- 'pgrep -fa /launch_'[0m
46097 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8_entropy50.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh dev-n1-wus2 -- 'pgrep -fa /launch_'[0m
4154101 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n20_sc3k_ctx8.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n1-uks7 -- 'pgrep -fa /launch_'[0m
3072984 python3 ./launch_train.py orng_conf/biasing/done/grpo_hcv2_fy22_m1000_seed_egs300k_simple_err_ga8_t12_n10x2_sc3k_ctx8.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n2-hpe4 -- 'pgrep -fa /launch_'[0m
667165 python3 ./launch_train.py orng_conf/biasing/done/grpo_hcv2_fy22_m1000_seed_e1_simple_err_ga8_t12_n10x2_sc3k_ctx5.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n2-uks7 -- 'pgrep -fa /launch_'[0m
19232 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e6_simple_err_ga8_t12_n10_sc3k_ctx8.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n2-wus2 -- 'pgrep -fa /launch_'[0m
3820292 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8_inject_ref.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h-n4-hpe4 -- 'pgrep -fa /launch_'[0m
2508494 python3 ./launch_train.py orng_conf/biasing/done/grpo_hcv2_fy22_m1000_zero_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8.yaml --seed_name grpo_hcv2_fy22_m1000_zero_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8_G2x8
[32;1mRUN: /home/boren/.openai/bin/brix ssh h1-n2-hpe4 -- 'pgrep -fa /launch_'[0m
1202887 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8_entropy80.yaml
[32;1mRUN: /home/boren/.openai/bin/brix ssh h1-n2-wus2 -- 'pgrep -fa /launch_'[0m
1460064 python3 ./launch_train.py orng_conf/biasing/pending/grpo_ls_m1000_seed1_e1_simple_err_ga8_t12_n10_sc3k_ctx8.yaml
