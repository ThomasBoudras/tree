#End-to-end model with super resolution and subsequent regression

from torch import nn


class classicNetwork(nn.Module):

    def __init__(self, 
                 regression_model, 
                ):
        super().__init__()
        self.regression_model = regression_model
        

    def forward(self, inputs, meta_data):
        preds = self.regression_model(inputs, meta_data)
        return preds
