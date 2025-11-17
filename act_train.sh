

python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /home_shared/grail_andre/code/bimaminobolonana/bc-train-data-test \
  --output_dir runs/act_results \
  --device cuda \
  --wandb_project "act-1" \
  --wandb_name "policy-act_pri3d"


sleep 10


python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /home_shared/grail_andre/code/bimaminobolonana/bc-train-data-test \
  --output_dir runs/act_results \
  --device cuda \
  --wandb_project "act-1" \
  --wandb_name "policy-act"
