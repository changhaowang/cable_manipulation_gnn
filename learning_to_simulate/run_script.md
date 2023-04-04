python -m learning_to_simulate.train \
    --data_path=learning_to_simulate/datasets/Water-3D \
    --model_path=learning_to_simulate/models/Water-3D

python -m learning_to_simulate.train \
    --mode="eval_rollout" \
    --data_path=learning_to_simulate/datasets/Water-3D \
    --model_path=learning_to_simulate/models/Water-3D \
    --output_path=learning_to_simulate/rollouts/Water-3D

python -m learning_to_simulate.render_rollout \
    --rollout_path=learning_to_simulate/rollouts/WaterRamps/rollout_test_0.pkl

python -m learning_to_simulate.render_rollout \
    --rollout_path=learning_to_simulate/rollouts/Water-3D/rollout_test_0.pkl