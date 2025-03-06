import torch
import torch.nn as nn

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

class OLMoOutput(): ...


class OLMo(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformers = nn.ModuleDict(
            dict(
                wte=nn.Embedding(50304),
            )
        )
    
    @property
    def device(self):
        return "cuda:0"
    
    
    
    
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        output_hidden_states: Optional[bool] = None
    ) -> OLMoOutput:
        """_summary_

        Args:
            input_ids (torch.LongTensor): _description_
            input_embeddings (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            past_key_values (Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]], optional): _description_. Defaults to None.
                the dimensions of past_key_value is 
            output_hidden_states (Optional[bool], optional): _description_. Defaults to None.

        Returns:
            OLMoOutput: _description_
        """

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        
        if past_key_values:
            # the length of past_key_value must be equal to the number of layer
            # because must cache each key and value for each layer.
            assert len(past_key_values) == 12 # the number of layers are 12.
        
        # get input embedding
        x = self.transformer.wte(input_ids) if input_ids is None else input_embeddings
        

    


if __name__ == "__main__":
    olmo = OLMo()
    print(olmo.device)