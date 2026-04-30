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

        # Partie Encodeur (MemN2N)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])
        self.temporal_embeddings = nn.ModuleList([
            nn.Embedding(max_story_len + 1, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])

        # Partie Décodeur (GRU)
        self.decoder_rnn = nn.GRU(embed_size, embed_size, batch_first=True)
        self.out = nn.Linear(embed_size, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.embeddings:
            nn.init.normal_(m.weight, std=0.1)
        for m in self.temporal_embeddings:
            nn.init.normal_(m.weight, std=0.1)

    def forward(self, story, query):
        batch_size, num_sentences, _ = story.size()
        time_idx = torch.arange(num_sentences, 0, -1, device=story.device).unsqueeze(0).expand(batch_size, -1)
        
        # Encodage du contexte 'u'
        u = self.embeddings[0](query).sum(dim=1)
        for k in range(self.hops):
            m = self.embeddings[k](story).sum(dim=2) + self.temporal_embeddings[k](time_idx)
            c = self.embeddings[k+1](story).sum(dim=2) + self.temporal_embeddings[k+1](time_idx)
            p = torch.softmax(torch.bmm(m, u.unsqueeze(2)).squeeze(2), dim=-1)
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1)
            u = u + o

        # Décodage séquentiel
        hidden = u.unsqueeze(0) 
        decoder_input = torch.zeros(batch_size, 1, self.embed_size, device=story.device)
        
        all_logits = []
        for t in range(self.max_ans_len):
            output, hidden = self.decoder_rnn(decoder_input, hidden)
            logits = self.out(output.squeeze(1))
            all_logits.append(logits)
            
            top_i = logits.argmax(1)
            decoder_input = self.embeddings[-1](top_i).unsqueeze(1)
            
        return torch.stack(all_logits, dim=1)


def prepare_data_multi(questions, stories, word2idx, max_story_len=50, max_ans_len=5):
    """Prépare les tenseurs pour les réponses à mots multiples."""
    X_story, X_query, Y_answer = [], [], []
    
    for q in questions:
        # Traitement Story
        story_text = stories[q["story_id"]]["sentences"][-max_story_len:]
        story_tensor = torch.zeros((max_story_len, 15), dtype=torch.long)
        for i, sentence in enumerate(story_text):
            for j, word in enumerate(BabiDataset.tokenize(sentence)[:15]):
                story_tensor[i, j] = word2idx.get(word, 1)
        
        # Traitement Query
        query_tensor = torch.zeros(10, dtype=torch.long)
        for j, word in enumerate(BabiDataset.tokenize(q["question"])[:10]):
            query_tensor[j] = word2idx.get(word, 1)

        # Multi-word Answer: Tokenisation de la séquence entière
        ans_tokens = BabiDataset.tokenize(q["answer"].lower())
        ans_tensor = torch.zeros(max_ans_len, dtype=torch.long)
        for j, word in enumerate(ans_tokens[:max_ans_len]):
            ans_tensor[j] = word2idx.get(word, 1)

        X_story.append(story_tensor)
        X_query.append(query_tensor)
        Y_answer.append(ans_tensor)
        
    return torch.stack(X_story), torch.stack(X_query), torch.stack(Y_answer)