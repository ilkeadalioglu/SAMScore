import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

def calculate_samscore_by_epoch_and_patient(source_folder, generated_folder, output_json_file, evaluator, selected_patients=None):
    #evaluator = samscore_class(model_type="vit_b")

    # Load existing scores if available
    if os.path.exists(output_json_file):
        with open(output_json_file, "r") as f:
            existing_data = json.load(f)
        existing_scores = existing_data.get("scores", {})
    else:
        existing_data = {}
        existing_scores = {}

    for epoch in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,150,200,250,300,350,400]:
        for patient_id in range(1000):  # arbitrary upper bound for patient IDs
            key_str = f"e{epoch:03d}_p{patient_id:03d}"
            if key_str in existing_scores:
                continue

            if selected_patients is not None and patient_id not in selected_patients:
                continue

            frame_id = 1
            score_list = []

            while True:
                src_filename = f"p{patient_id:03d}_f{frame_id:03d}.png"
                gen_filename = f"ec_e{epoch:03d}_p{patient_id:03d}_f{frame_id:03d}.png"
                src_path = os.path.join(source_folder, src_filename)
                gen_path = os.path.join(generated_folder, gen_filename)

                if not os.path.exists(src_path) or not os.path.exists(gen_path):
                    if frame_id == 1:
                        break  # skip this patient if no frames found
                    else:
                        break  # end of frames for this patient

                try:
                    score = evaluator.evaluation_from_path(source_image_path=src_path, generated_image_path=gen_path)
                    if hasattr(score, 'cpu'):
                        score = score.cpu().item()
                    elif hasattr(score, 'item'):
                        score = score.item()
                    score_list.append(score)
                except Exception as e:
                    print(f"Error processing {src_filename} and {gen_filename}: {e}")

                frame_id += 1

            if score_list:
                mean_score = float(np.mean(score_list))
                existing_scores[key_str] = mean_score
                print(f"{key_str}: {mean_score:.4f}")

        # Save intermediate result after each epoch
        result_dict = {
            "scores": existing_scores
        }
        with open(output_json_file, "w") as f:
            json.dump(result_dict, f, indent=2)

    return existing_scores


def samscore_per_epoch(json_path):
    import matplotlib.pyplot as plt
    import json
    import numpy as np
    import re

    with open(json_path, "r") as f:
        data = json.load(f)
    scores = data["scores"]

    fixed_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55,60,65,70,75,80,85,90,95, 100]
    epoch_patient_scores = {epoch: [] for epoch in fixed_epochs}

    for key, val in scores.items():
        match = re.match(r"e(\d+)_p(\d+)", key)
        if match:
            epoch = int(match.group(1))
            if epoch in epoch_patient_scores:
                epoch_patient_scores[epoch].append(val)

    plt.figure(figsize=(10, 6))
    for epoch in fixed_epochs:
        vals = epoch_patient_scores[epoch]
        if vals:
            x = [epoch] * len(vals)
            y = vals
            plt.scatter(x, y, alpha=0.5, s=10, label=f"epoch {epoch}")
            plt.scatter(epoch, np.mean(y), color='black', s=80, edgecolor='white', zorder=5)

    plt.xlabel("Epoch")
    plt.ylabel("SAM Score")
    plt.title("SAM Score per Patient and Epoch")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("samscore_per_epoch.png")
    plt.show()

def samscore_per_patient(json_path, selected_epoch):
    import matplotlib.pyplot as plt
    import json
    import numpy as np
    import re

    with open(json_path, "r") as f:
        data = json.load(f)
    scores = data["scores"]

    patient_scores = {}

    for key, val in scores.items():
        match = re.match(r"e(\d+)_p(\d+)", key)
        if match:
            epoch = int(match.group(1))
            patient = int(match.group(2))
            if epoch == selected_epoch:
                patient_scores[patient] = val

    patient_ids = sorted(patient_scores.keys())
    labels = [f"{pid:03d}" for pid in patient_ids]
    values = [patient_scores[p] for p in patient_ids]

    plt.figure(figsize=(14, 6))
    x = np.arange(len(patient_ids))
    plt.scatter(x, values, s=30, color='blue')
    plt.xticks(x, labels, rotation=90)
    plt.xlabel("Patient ID")
    plt.ylabel("SAM Score")
    plt.title(f"SAM Score per Patient (Epoch {selected_epoch})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"samscore_per_patient_epoch{selected_epoch}.png")
    plt.show()




