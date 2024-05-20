








from .pipeline import Pipeline
from ...src.misc.utils import env_meta_info


class PipelineHub:
    def __init__(self, *args, **kwargs):
        super(PipelineHub, self).__init__(*args, **kwargs)
        self.metadata = env_meta_info()

    @staticmethod
    def load(pipeline_name_or_path, local_only=False, **kwargs):
        return Pipeline.load(pipeline_name_or_path, local_only=local_only, **kwargs)

    def push(self, pipeline, **kwargs):
        raise NotImplementedError("This method has not implemented yet.")
