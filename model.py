import torch
import torch.nn as nn
from data.dataloader import BabiDataset

class MemN2N(nn.Module):
    def __init__(self, vocab_size, embed_size=20, max_story_len=50, hops=3):
        super(MemN2N, self).__init__()
        self.hops = hops
        self.embed_size = embed_size
        self.max_story_len = max_story_len
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_size, padding_idx=0) 
            for _ in range(hops + 1)
        ])
        
        self.temporal_embeddings = nn.ModuleList([
            nn.Embedding(max_story_len + 1, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])
        self._init_weights()

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
        
        u = self.embeddings[0](query).sum(dim=1) 
        
        for k in range(self.hops):
            A = self.embeddings[k]
            C = self.embeddings[k+1]
            T_A = self.temporal_embeddings[k]
            T_C = self.temporal_embeddings[k+1]
            
            m = A(story).sum(dim=2) + T_A(time_idx) 
            c = C(story).sum(dim=2) + T_C(time_idx) 
            
            scores = torch.bmm(m, u.unsqueeze(2)).squeeze(2) 
            p = torch.softmax(scores, dim=-1)
            
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