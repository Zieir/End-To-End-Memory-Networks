import torch
import torch.nn as nn
from data.dataloader import BabiDataset

class MemN2N(nn.Module):
    def __init__(self, vocab_size, embed_size=20, max_story_len=50, hops=3,
                 use_pe=False):
        super(MemN2N, self).__init__()
        self.hops = hops
        self.embed_size = embed_size
        self.max_story_len = max_story_len
        self.use_softmax = True
        self.use_pe = use_pe

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])

        self.temporal_embeddings = nn.ModuleList([
            nn.Embedding(max_story_len + 1, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])
        self._init_weights()
        self._pe_cache = {}

    def _position_encoding(self, sentence_len, device):
        key = (sentence_len, str(device))
        if key in self._pe_cache:
            return self._pe_cache[key]

        J = sentence_len
        d = self.embed_size
        j = torch.arange(1, J + 1, device=device, dtype=torch.float32)
        k = torch.arange(1, d + 1, device=device, dtype=torch.float32)
        l = (1.0 - j.unsqueeze(0) / J) - (k.unsqueeze(1) / d) * (1.0 - 2.0 * j.unsqueeze(0) / J)
        l = l.t().contiguous()
        self._pe_cache[key] = l
        return l

    def _encode_sentences(self, story, embed_layer):
        e = embed_layer(story)
        if self.use_pe:
            J = story.size(2)
            l = self._position_encoding(J, story.device)
            e = e * l.unsqueeze(0).unsqueeze(0)
        return e.sum(dim=2)

    def _encode_query(self, query, embed_layer):
        e = embed_layer(query)
        if self.use_pe:
            J = query.size(1)
            l = self._position_encoding(J, query.device)
            e = e * l.unsqueeze(0)
        return e.sum(dim=1)

    def _init_weights(self):
        """Initializes weights with mean=0, std=0.1 as per the paper."""
        for m in self.embeddings:
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.weight[0], 0) 
            
        for m in self.temporal_embeddings:
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.weight[0], 0) 

    def forward(self, story, query):
        batch_size, num_sentences, _ = story.size()
        time_idx = torch.arange(num_sentences, 0, -1, device=story.device).unsqueeze(0).expand(batch_size, -1)
        
        u = self._encode_query(query, self.embeddings[0])

        for k in range(self.hops):
            A = self.embeddings[k]
            C = self.embeddings[k+1]
            T_A = self.temporal_embeddings[k]
            T_C = self.temporal_embeddings[k+1]

            m = self._encode_sentences(story, A) + T_A(time_idx)
            c = self._encode_sentences(story, C) + T_C(time_idx)
            
            scores = torch.bmm(m, u.unsqueeze(2)).squeeze(2)
            p = torch.softmax(scores, dim=-1) if self.use_softmax else scores
            
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1) 
            u = u + o
            
        W = self.embeddings[-1].weight
        logits = torch.matmul(u, W.T) 
        return logits

def prepare_data(questions, stories, word2idx, max_story_len=50):
    """Converts the raw text into padded PyTorch tensors."""
    X_story, X_query, Y_answer = [], [], []
    
    max_sen_len = max([len(BabiDataset.tokenize(s)) for story in stories for s in story["sentences"]])
    max_q_len = max([len(BabiDataset.tokenize(q["question"])) for q in questions])
    
    for q in questions:
        story_idx = q["story_id"]
        story_text = stories[story_idx]["sentences"][-max_story_len:] 
        
        story_tensor = torch.zeros((max_story_len, max_sen_len), dtype=torch.long)
        for i, sentence in enumerate(story_text):
            tokens = BabiDataset.tokenize(sentence)
            for j, word in enumerate(tokens):
                story_tensor[i, j] = word2idx.get(word, 1) 
                
        query_tensor = torch.zeros(max_q_len, dtype=torch.long)
        for j, word in enumerate(BabiDataset.tokenize(q["question"])):
            query_tensor[j] = word2idx.get(word, 1)
            
        answer_idx = word2idx.get(q["answer"].lower(), 1)
        
        X_story.append(story_tensor)
        X_query.append(query_tensor)
        Y_answer.append(answer_idx)
        
    return torch.stack(X_story), torch.stack(X_query), torch.tensor(Y_answer)