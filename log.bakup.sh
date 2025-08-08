ssh dev-n1-hpe2 -- 'pgrep -fa /launch_'
574169 python3 ./launch_train.py orng_conf/biasing/grpo_ls_m1000_seed1k_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8.yaml

ssh dev-n1-uks7 -- 'pgrep -fa /launch_'
798170 python3 ./launch_train.py orng_conf/biasing/grpo_hcv2_fy22_m1000_seed_egs300k_simple_err_ga8_t12_n10x2_sc3k_ctx8.yaml

ssh dev-n1-wus2 -- 'pgrep -fa /launch_'
exit status 1

ssh h-n1-uks7 -- 'pgrep -fa /launch_'
281103 python3 ./launch_train.py orng_conf/biasing/grpo_ls_m1000_seed_e3_simple_err_ga8_t15_n10x2_sc3k_ctx9.yaml

ssh h-n2-hpe4 -- 'pgrep -fa /launch_'
1046705 python3 ./launch_train.py orng_conf/biasing/grpo_hcv2_fy22_m1000_seed_e1_simple_err_ga8_t12_n10x2_sc3k_ctx5.yaml


ssh h-n2-uks7 -- 'pgrep -fa /launch_'
1351182 python3 ./launch_train.py orng_conf/biasing/grpo_hcv2_fy22_m1000_seed_egs300k_simple_err_ga8_t12_n10x2_sc3k_ctx5.yaml --seed_name grpo_hcv2_fy22_m1000_seed_egs300k_simple_err_ga8_t12_n10x2_sc3k_ctx5_G1x8

ssh h-n2-wus2 -- 'pgrep -fa /launch_'
796322 python3 ./launch_train.py orng_conf/biasing/grpo_ls_m1000_zero_e2_simple_err_ga8_t12_n10_sc3k_ctx9.yaml

ssh h-n4-hpe4 -- 'pgrep -fa /launch_'
3042998 python3 ./launch_train.py orng_conf/biasing/grpo_ls_m1000_seed_e6_simple_err_ga8_t15_n10x2_sc3k_ctx8.yaml

ssh h1-n2-hpe4 -- 'pgrep -fa /launch_'
206979 python3 ./launch_train.py orng_conf/biasing/grpo_hcv2_fy22_m1000_zero_e1_simple_err_ga8_t12_n10x2_sc3k_ctx8.yaml

ssh h1-n2-wus2 -- 'pgrep -fa /launch_'
2561069 python3 ./launch_train.py orng_conf/biasing/grpo_ls_rare_zero_e3_simple_err_ga8_t12_n10x2_sc3k_ctx0.yaml

ssh l-n1-hpe2 -- 'pgrep -fa /launch_'
195086 python3 ./launch_train.py orng_conf/biasing/grpo_ls_m1000_zero_e1_simple_err10_ga8_t12_n10x2_sc3k_ctx8.yaml

