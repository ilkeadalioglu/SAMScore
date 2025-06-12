import samscore  # Make sure this is available in your environment
import os
from my_functions import calculate_samscore_by_epoch_and_patient, samscore_per_epoch, samscore_per_patient

# Set your input/output paths
source_dir = "/scratch/zhangh/ilke/SAMScore/imgs/cycle_gan_E2M4_v0/A"
generated_dir = "/scratch/zhangh/ilke/SAMScore/imgs/cycle_gan_E2M4_v0/A2B"
json_out = "./results/MEDsamscore_results_A2A2B.json"

SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_b", model_weight_path = "weights/medsam_vit_b.pth" )

# Calculate SAM scores and retrieve per-patient, per-epoch info
existing_scores = calculate_samscore_by_epoch_and_patient(
    source_folder=source_dir,
    generated_folder=generated_dir,
    output_json_file=json_out,
    evaluator=SAMScore_Evaluation
) 

samscore_per_epoch(json_out)