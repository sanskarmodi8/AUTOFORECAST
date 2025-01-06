from zenml.steps import step

STAGE_NAME = 'MODEL TRAINING STAGE'

@step
def train_step(success:bool) -> bool:
    return True