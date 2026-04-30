import torch
import torch.nn as nn
from data.dataloader import BabiDataset

class MemN2N_MultiWord(nn.Module):
    def __init__(self, vocab_size, embed_size=20, max_story_len=50, hops=3, max_ans_len=5):
        super(MemN2N_MultiWord, self).__init__()
        self.hops = hops
        self.embed_size = embed_size
        self.max_ans_len = max_ans_len
        self.vocab_size = vocab_size

        # Partie Encodeur (MemN2N)[cite: 8, 15]
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])
        self.temporal_embeddings = nn.ModuleList([
            nn.Embedding(max_story_len + 1, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])

        # Partie Décodeur (GRU)[cite: 2, 6]
        self.decoder_rnn = nn.GRU(embed_size, embed_size, batch_first=True)
        self.out = nn.Linear(embed_size, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.embeddings:
            nn.init.normal_(m.weight, std=0.1)[cite: 15]
        for m in self.temporal_embeddings:
            nn.init.normal_(m.weight, std=0.1)[cite: 15]

    def forward(self, story, query):
        batch_size, num_sentences, _ = story.size()
        time_idx = torch.arange(num_sentences, 0, -1, device=story.device).unsqueeze(0).expand(batch_size, -1)[cite: 15]
        
        # Encodage du contexte 'u'
        u = self.embeddings[0](query).sum(dim=1)[cite: 15]
        for k in range(self.hops):
            m = self.embeddings[k](story).sum(dim=2) + self.temporal_embeddings[k](time_idx)[cite: 15]
            c = self.embeddings[k+1](story).sum(dim=2) + self.temporal_embeddings[k+1](time_idx)[cite: 15]
            p = torch.softmax(torch.bmm(m, u.unsqueeze(2)).squeeze(2), dim=-1)[cite: 15]
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1)[cite: 15]
            u = u + o[cite: 15]

        # Décodage séquentiel
        hidden = u.unsqueeze(0) 
        # On commence avec un vecteur nul pour le premier mot
        decoder_input = torch.zeros(batch_size, 1, self.embed_size, device=story.device)
        
        all_logits = []
        for t in range(self.max_ans_len):
            output, hidden = self.decoder_rnn(decoder_input, hidden)[cite: 6]
            logits = self.out(output.squeeze(1))
            all_logits.append(logits)
            
            # Le mot prédit devient l'entrée suivante (Greedy decoding)
            top_i = logits.argmax(1)
            decoder_input = self.embeddings[-1](top_i).unsqueeze(1)
            
        return torch.stack(all_logits, dim=1) # (batch, max_ans_len, vocab_size)