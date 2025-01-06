from zenml.steps import step

STAGE_NAME = 'DATA TRANSFORMATION STAGE'

@step
def transform_step() -> bool:
    return True
