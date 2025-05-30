from transformers import TrainerCallback
from src.utils.visualization import TrainingVisualizer

class VisualizationCallback(TrainerCallback):
    def __init__(self, visualizer: TrainingVisualizer):
        self.visualizer = visualizer
        
    def on_step_end(self, args, state, control, **kwargs):
        """每个训练步骤结束时调用"""
        if state.global_step % args.logging_steps == 0:
            # 获取当前步骤的rewards
            if hasattr(state, 'metrics') and 'rewards' in state.metrics:
                rewards = state.metrics['rewards']
                self.visualizer.log_rewards({
                    'total_rewards': rewards.mean().item(),
                }, state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """每次评估时调用"""
        if metrics:
            self.visualizer.log_rewards({
                'eval_rewards': metrics.get('eval_rewards', 0),
            }, state.global_step)
    
    def on_prediction_step(self, args, state, control, inputs=None, outputs=None, **kwargs):
        """每次预测步骤时调用"""
        if outputs and hasattr(outputs, 'predictions'):
            predictions = outputs.predictions
            targets = inputs['labels'] if 'labels' in inputs else None
            
            if predictions is not None and targets is not None:
                # 记录预测样本
                self.visualizer.log_prediction_sample(
                    input_coords=inputs.get('input_text', ''),
                    predicted_coords=predictions[0],  # 假设batch_size=1
                    actual_coords=targets[0],        # 假设batch_size=1
                    total_reward=outputs.metrics.get('rewards', 0),
                    step=state.global_step
                )
