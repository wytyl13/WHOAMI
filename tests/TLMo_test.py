import pytest



from whoami.tool.transformer.model import TLMo
from whoami.tool.transformer.model_config import TransformerModelConfig
from whoami.tool.transformer.check_point_config import ActivationCheckpointStrategy
def test_TLMo_model():

    transformer_config = TransformerModelConfig()

    tlmo_model = TLMo(transformer_config=transformer_config)
    print(tlmo_model.config.embedding_size)
    print(ActivationCheckpointStrategy.whole_layer.value == "whole_layer")
    print(ActivationCheckpointStrategy.one_in_three.value == "one_in_three")
    
    def test(enum_: ActivationCheckpointStrategy = None):
        print(enum_.value == "one_in_three")
        if enum_ == "one_in_three":
            return "whoami"
        else:
            return "whoareyou"
            
    print(test(ActivationCheckpointStrategy.one_in_three))