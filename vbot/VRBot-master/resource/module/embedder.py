# -*- coding: utf-8 -*-


import torch
import torch.nn as nn


class Embedder(nn.Embedding):
    def load_embeddings(self, embeds, scale=0.05):
        assert len(embeds) == self.num_embeddings

        embeds = torch.tensor(embeds)
        num_known = 0
        for i in range(len(embeds)):
            if len(embeds[i].nonzero()) == 0:
                nn.init.uniform_(embeds[i], -scale, scale)
            else:
                num_known += 1
        self.weight.data.copy_(embeds)
        print("{} words have pretrained embeddings".format(num_known),
              "(coverage: {:.3f})".format(num_known / self.num_embeddings))


class PositionEmbedder(nn.Embedding):
    def __init__(self, max_image_num, position_dim):
        super(PositionEmbedder, self).__init__(max_image_num, position_dim)
