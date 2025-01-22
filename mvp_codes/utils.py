
def estimate_params(model):
    '''
    Estimate the number of parameters in a model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)