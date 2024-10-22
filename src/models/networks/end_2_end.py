#End-to-end model with super resolution and subsequent regression

from torch import nn


class end2EndNetwork(nn.Module):

    def __init__(self, 
                 super_resolution_model, 
                 regression_model, 
                ):
        super().__init__()
        self.super_resolution_model = super_resolution_model
        self.regression_model = regression_model
        

    def forward(self, inputs):
        SR_inputs = self.super_resolution_model(inputs)
        preds = self.regression_model(SR_inputs)
        return preds
