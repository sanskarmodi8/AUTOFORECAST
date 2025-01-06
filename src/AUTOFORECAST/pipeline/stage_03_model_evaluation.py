from zenml.steps import step

STAGE_NAME = "MODEL EVALUATION STAGE"


@step
def evaluate_step(success: bool) -> bool:
    return True
