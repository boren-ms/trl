# Dataset Information

## Teams Dataset

1. **type1**  
   [teams_type1.yaml](https://dev.azure.com/speedme/Speech/_git/Cascades?path=/tools/zetta/dataset_configs/teams_type1.yaml)
2. **type2**  
   [teams_type2_200.yaml](https://dev.azure.com/speedme/Speech/_git/Cascades?path=/tools/zetta/dataset_configs/teams_type2_200.yaml)

3. **type3**  
   [teams_type2_1000.yaml](https://dev.azure.com/speedme/Speech/_git/Cascades?path=/tools/zetta/dataset_configs/teams_type2_1000.yaml)


### Download

1. Go to  `amgpu06` machine and navigate to the `Cascades` repository.
2. Switch to the branch: `boren/main_cust_whisper`.
3. Activate the `csd_main_zetta` conda environment.
4. Run the script:
   ```
   python exp/download_dataset.py
   ```