# MiniOneReason 实验状态与结果

已生成正式汇总：116 项。数值来自 reports/results_v2 原始汇总，未完成的实验不填数。

数据边界：按历史输入分组留出；不能证明用户级互斥或与预训练集无重叠。主指标在商品 SID 层面计算。
机器条件：主模型实验来自共享 RTX 4090，迁移后的成本与历史实验来自 RTX 5090；各结果保留 runtime_id。跨显卡或共享负载下的耗时不能直接解释为方法收益。

| 实验 | 样本 | Recall@32 | 无效比例 | token/用户 |
| --- | ---: | ---: | ---: | ---: |
| base_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.024814 | 0.2492 | 322.7 |
| base_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.020779 | 0.2575 | 321.1 |
| direct_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.027295 | 0.2885 | 320.0 |
| direct_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.033766 | 0.2881 | 320.0 |
| direct_tokenmatched_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.032258 | 0.2199 | 320.0 |
| direct_tokenmatched_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.031169 | 0.2169 | 320.0 |
| main_history_mixed_balanced_42_test_balanced_1024 | 403 | 0.019851 | 0.3778 | 50074.9 |
| main_history_mixed_balanced_42_test_recent_1024 | 403 | 0.014888 | 0.3727 | 49988.4 |
| main_history_mixed_balanced_42_validation_balanced_1024 | 385 | 0.012987 | 0.3736 | 49928.3 |
| main_history_mixed_balanced_42_validation_recent_1024 | 385 | 0.012987 | 0.3653 | 49873.4 |
| main_history_mixed_recent_history_42_test_balanced_1024 | 403 | 0.019851 | 0.3850 | 50058.4 |
| main_history_mixed_recent_history_42_test_recent_1024 | 403 | 0.014888 | 0.3848 | 49994.5 |
| main_history_mixed_recent_history_42_validation_balanced_1024 | 385 | 0.023377 | 0.3750 | 50027.5 |
| main_history_mixed_recent_history_42_validation_recent_1024 | 385 | 0.020779 | 0.3715 | 49947.3 |
| main_history_rl_nk_prefix_42_test_balanced_1024 | 403 | 0.037221 | 0.3083 | 49259.5 |
| main_history_rl_nk_prefix_42_test_balanced_2048 | 403 | 0.037221 | 0.3161 | 49375.5 |
| main_history_rl_nk_prefix_42_test_balanced_512 | 403 | 0.019851 | 0.2867 | 48390.9 |
| main_history_rl_nk_prefix_42_test_domain_recent_1024 | 403 | 0.044665 | 0.3132 | 49231.9 |
| main_history_rl_nk_prefix_42_test_domain_recent_2048 | 403 | 0.037221 | 0.3161 | 49375.5 |
| main_history_rl_nk_prefix_42_test_domain_recent_512 | 403 | 0.017370 | 0.2875 | 48374.5 |
| main_history_rl_nk_prefix_42_test_recent_1024 | 403 | 0.037221 | 0.3118 | 49164.5 |
| main_history_rl_nk_prefix_42_test_recent_2048 | 403 | 0.037221 | 0.3161 | 49375.5 |
| main_history_rl_nk_prefix_42_test_recent_512 | 403 | 0.009926 | 0.2829 | 48138.1 |
| main_history_rl_nk_prefix_42_validation_balanced_1024 | 385 | 0.038961 | 0.3101 | 49208.6 |
| main_history_rl_nk_prefix_42_validation_balanced_2048 | 385 | 0.038961 | 0.3063 | 49333.9 |
| main_history_rl_nk_prefix_42_validation_balanced_512 | 385 | 0.023377 | 0.2827 | 48287.9 |
| main_history_rl_nk_prefix_42_validation_domain_recent_1024 | 385 | 0.038961 | 0.3067 | 49207.2 |
| main_history_rl_nk_prefix_42_validation_domain_recent_2048 | 385 | 0.038961 | 0.3062 | 49329.9 |
| main_history_rl_nk_prefix_42_validation_domain_recent_512 | 385 | 0.023377 | 0.2825 | 48400.2 |
| main_history_rl_nk_prefix_42_validation_recent_1024 | 385 | 0.036364 | 0.2996 | 49116.2 |
| main_history_rl_nk_prefix_42_validation_recent_2048 | 385 | 0.038961 | 0.3062 | 49329.9 |
| main_history_rl_nk_prefix_42_validation_recent_512 | 385 | 0.025974 | 0.2876 | 48104.7 |
| popularity_42_test | 403 | 0.000000 | 0.0000 | 0.0 |
| popularity_42_validation | 385 | 0.000000 | 0.0000 | 0.0 |
| main_rl_nk_prefix_route_42_test_all_non_thinking | 403 | 0.049628 | 0.2964 | 160.0 |
| main_rl_nk_prefix_route_42_test_history_length_0.25 | 403 | 0.049628 | 0.2914 | 2263.5 |
| main_rl_nk_prefix_route_42_test_history_length_0.25_random | 403 | 0.042184 | 0.2981 | 2261.7 |
| main_rl_nk_prefix_route_42_test_history_length_0.5 | 403 | 0.049628 | 0.2904 | 4161.6 |
| main_rl_nk_prefix_route_42_test_history_length_0.5_random | 403 | 0.037221 | 0.2984 | 4160.3 |
| main_rl_nk_prefix_route_42_test_uncertainty_0.25 | 403 | 0.049628 | 0.3000 | 2284.1 |
| main_rl_nk_prefix_route_42_test_uncertainty_0.25_random | 403 | 0.042184 | 0.2979 | 2282.1 |
| main_rl_nk_prefix_route_42_test_uncertainty_0.5 | 403 | 0.044665 | 0.2987 | 4406.1 |
| main_rl_nk_prefix_route_42_test_uncertainty_0.5_random | 403 | 0.037221 | 0.2988 | 4405.5 |
| main_rl_nk_prefix_route_42_validation_all_non_thinking | 385 | 0.023377 | 0.2969 | 160.0 |
| main_rl_nk_prefix_route_42_validation_history_length_0.25 | 385 | 0.033766 | 0.3011 | 2212.4 |
| main_rl_nk_prefix_route_42_validation_history_length_0.25_random | 385 | 0.033766 | 0.2943 | 2212.6 |
| main_rl_nk_prefix_route_42_validation_history_length_0.5 | 385 | 0.044156 | 0.2963 | 4265.1 |
| main_rl_nk_prefix_route_42_validation_history_length_0.5_random | 385 | 0.046753 | 0.2937 | 4264.8 |
| main_rl_nk_prefix_route_42_validation_uncertainty_0.25 | 385 | 0.033766 | 0.3054 | 2212.3 |
| main_rl_nk_prefix_route_42_validation_uncertainty_0.25_random | 385 | 0.033766 | 0.2943 | 2212.6 |
| main_rl_nk_prefix_route_42_validation_uncertainty_0.5 | 385 | 0.036364 | 0.3055 | 4264.8 |
| main_rl_nk_prefix_route_42_validation_uncertainty_0.5_random | 385 | 0.046753 | 0.2937 | 4264.8 |
| main_rl_nk_prefix_thinkcost_42_test_0 | 403 | 0.029777 | 0.3020 | 160.0 |
| main_rl_nk_prefix_thinkcost_42_test_1024 | 403 | 0.037221 | 0.3072 | 24635.3 |
| main_rl_nk_prefix_thinkcost_42_test_128 | 403 | 0.034739 | 0.2981 | 4256.3 |
| main_rl_nk_prefix_thinkcost_42_test_256 | 403 | 0.034739 | 0.2956 | 8345.8 |
| main_rl_nk_prefix_thinkcost_42_test_512 | 403 | 0.037221 | 0.3030 | 16523.2 |
| main_rl_nk_prefix_thinkcost_42_test_mixed16 | 403 | 0.047146 | 0.2959 | 4252.2 |
| main_rl_nk_prefix_thinkcost_42_validation_0 | 385 | 0.038961 | 0.2903 | 160.0 |
| main_rl_nk_prefix_thinkcost_42_validation_1024 | 385 | 0.044156 | 0.2993 | 24614.2 |
| main_rl_nk_prefix_thinkcost_42_validation_128 | 385 | 0.038961 | 0.2985 | 4256.3 |
| main_rl_nk_prefix_thinkcost_42_validation_256 | 385 | 0.051948 | 0.2959 | 8351.5 |
| main_rl_nk_prefix_thinkcost_42_validation_512 | 385 | 0.049351 | 0.3010 | 16527.0 |
| main_rl_nk_prefix_thinkcost_42_validation_mixed16 | 385 | 0.046753 | 0.2950 | 4256.3 |
| main_route_42_test_all_non_thinking | 403 | 0.029777 | 0.3935 | 160.0 |
| main_route_42_test_history_length_0.25 | 403 | 0.029777 | 0.3882 | 2264.4 |
| main_route_42_test_history_length_0.25_random | 403 | 0.022333 | 0.3964 | 2263.7 |
| main_route_42_test_history_length_0.5 | 403 | 0.027295 | 0.3893 | 4163.8 |
| main_route_42_test_history_length_0.5_random | 403 | 0.019851 | 0.3945 | 4162.1 |
| main_route_42_test_uncertainty_0.25 | 403 | 0.022333 | 0.3991 | 2140.9 |
| main_route_42_test_uncertainty_0.25_random | 403 | 0.022333 | 0.3964 | 2141.2 |
| main_route_42_test_uncertainty_0.5 | 403 | 0.019851 | 0.4024 | 4568.9 |
| main_route_42_test_uncertainty_0.5_random | 403 | 0.022333 | 0.3954 | 4570.0 |
| main_route_42_validation_all_non_thinking | 385 | 0.015584 | 0.3830 | 160.0 |
| main_route_42_validation_history_length_0.25 | 385 | 0.015584 | 0.3879 | 2212.1 |
| main_route_42_validation_history_length_0.25_random | 385 | 0.015584 | 0.3822 | 2209.3 |
| main_route_42_validation_history_length_0.5 | 385 | 0.023377 | 0.3827 | 4262.8 |
| main_route_42_validation_history_length_0.5_random | 385 | 0.020779 | 0.3817 | 4260.8 |
| main_route_42_validation_uncertainty_0.25 | 385 | 0.020779 | 0.3927 | 2212.0 |
| main_route_42_validation_uncertainty_0.25_random | 385 | 0.015584 | 0.3822 | 2209.3 |
| main_route_42_validation_uncertainty_0.5 | 385 | 0.020779 | 0.3996 | 4262.8 |
| main_route_42_validation_uncertainty_0.5_random | 385 | 0.020779 | 0.3817 | 4260.8 |
| sasrec_42_test | 403 | 0.000000 | 0.0000 | 0.0 |
| sasrec_42_validation | 385 | 0.000000 | 0.0000 | 0.0 |
| main_thinkcost_42_test_0 | 403 | 0.017370 | 0.3911 | 160.0 |
| main_thinkcost_42_test_1024 | 403 | 0.024814 | 0.4213 | 24104.0 |
| main_thinkcost_42_test_128 | 403 | 0.019851 | 0.4021 | 4256.1 |
| main_thinkcost_42_test_256 | 403 | 0.014888 | 0.3917 | 8347.8 |
| main_thinkcost_42_test_512 | 403 | 0.024814 | 0.3995 | 16517.8 |
| main_thinkcost_42_test_mixed16 | 403 | 0.022333 | 0.3944 | 4254.0 |
| main_thinkcost_42_validation_0 | 385 | 0.038961 | 0.3836 | 160.0 |
| main_thinkcost_42_validation_1024 | 385 | 0.020779 | 0.4140 | 24043.0 |
| main_thinkcost_42_validation_128 | 385 | 0.023377 | 0.4028 | 4256.1 |
| main_thinkcost_42_validation_256 | 385 | 0.025974 | 0.3945 | 8344.1 |
| main_thinkcost_42_validation_512 | 385 | 0.020779 | 0.4057 | 16510.5 |
| main_thinkcost_42_validation_mixed16 | 385 | 0.025974 | 0.3866 | 4252.2 |
| mixed_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.019851 | 0.3896 | 320.0 |
| mixed_42_test_recent_in2048_thinkTrue_budgetNone_n64_vllm | 403 | 0.022333 | 0.4321 | 48153.1 |
| mixed_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.031169 | 0.3816 | 320.0 |
| mixed_42_validation_recent_in2048_thinkTrue_budgetNone_n64_vllm | 385 | 0.012987 | 0.4246 | 48026.6 |
| rl_nk_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.017370 | 0.3865 | 320.0 |
| rl_nk_42_test_recent_in2048_thinkTrue_budgetNone_n64_vllm | 403 | 0.027295 | 0.4256 | 47514.8 |
| rl_nk_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.031169 | 0.3805 | 320.0 |
| rl_nk_42_validation_recent_in2048_thinkTrue_budgetNone_n64_vllm | 385 | 0.020779 | 0.4181 | 47375.4 |
| rl_nk_prefix_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.034739 | 0.2968 | 320.0 |
| rl_nk_prefix_42_test_recent_in2048_thinkTrue_budgetNone_n64_vllm | 403 | 0.044665 | 0.3136 | 49428.7 |
| rl_nk_prefix_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.036364 | 0.2937 | 320.0 |
| rl_nk_prefix_42_validation_recent_in2048_thinkTrue_budgetNone_n64_vllm | 385 | 0.049351 | 0.3047 | 49352.9 |
| rl_std32_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.017370 | 0.3909 | 320.0 |
| rl_std32_42_test_recent_in2048_thinkTrue_budgetNone_n64_vllm | 403 | 0.027295 | 0.4282 | 48605.2 |
| rl_std32_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.031169 | 0.3835 | 320.0 |
| rl_std32_42_validation_recent_in2048_thinkTrue_budgetNone_n64_vllm | 385 | 0.025974 | 0.4170 | 48514.3 |
| rl_std_42_test_recent_in2048_thinkFalse_budgetNone_n64_vllm | 403 | 0.017370 | 0.3851 | 320.0 |
| rl_std_42_test_recent_in2048_thinkTrue_budgetNone_n64_vllm | 403 | 0.024814 | 0.4204 | 47370.1 |
| rl_std_42_validation_recent_in2048_thinkFalse_budgetNone_n64_vllm | 385 | 0.033766 | 0.3779 | 320.0 |
| rl_std_42_validation_recent_in2048_thinkTrue_budgetNone_n64_vllm | 385 | 0.018182 | 0.4096 | 47253.1 |

不能根据训练 loss 下降或脚本退出成功声明推荐效果提升。收益结论需配对比较及区间；全零奖励跳过与训练失败分别记录。

配对差值（mixed 减 direct，测试 Recall@32）：
- seed=42，thinking=False：-0.007444，95% 区间 [-0.020836060209424082, 0.004739617926174421]。
- seed=42，thinking=True：-0.004963，95% 区间 [-0.019709335097909467, 0.007854957244885271]。
