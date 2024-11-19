import torch 
import torch.nn as nn
import torch.distributions as dist


class MMDGM(nn.Module):
    def __init__(self, encoders_classes, decoders_classes, params):
        super().__init__()
        self.aggregation_method = params.aggregation_method
        self.latent_dim = params.latent_dim
        self.num_modalities = params.num_modalities
        self.decoder_latent_dim = params.latent_dim*params.num_modalities if params.aggregation_method=='concatenation' else params.latent_dim
        self.encoders = nn.ModuleList([enc(params.latent_dim).to(params.device) for enc in encoders_classes]) 
        self.decoders = nn.ModuleList([dec(self.decoder_latent_dim).to(params.device) for dec in decoders_classes]) 
        self.posteriors_not_instanciated = [dist.Normal if params.is_gaussian else dist.Laplace for _ in range(5)]
        
    def aggregate(self, zs_list):
        #Must be defined in subclasses default is concatenation
        if self.aggregation_method=='concatenation':
            zs = torch.cat(zs_list,1)
        return zs
    
    def encode(self, modalities):
        q_parameters =  [enc(modality) for enc, modality in zip(self.encoders, modalities)] # posteriors parameters
        qs = [q(*q_param) for q,q_param in zip(self.posteriors_not_instanciated,q_parameters)]
        zs_list = [q.rsample() for q in qs]
        zs = self.aggregate(zs_list)
        return zs
    
    def forward(self, modalities):
        zs = self.encode(modalities)
        if self.aggregation_method=='concatenation':
            outputs = [dec(zs) for dec in self.decoders]
        return outputs
        