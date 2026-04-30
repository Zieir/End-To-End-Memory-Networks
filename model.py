import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataloader import BabiDataset

class MemN2N(nn.Module):
    def __init__(self, vocab_size, embed_size=20, max_story_len=50, hops=3,
                 use_pe=False, use_rn=False, rn_p=0.1, rn_max_extra=10,
                 tying='adjacent', use_relu=False):
        super(MemN2N, self).__init__()
        if tying not in ('adjacent', 'layerwise'):
            raise ValueError(f"Unknown tying scheme: {tying!r}")
        self.hops = hops
        self.embed_size = embed_size
        self.max_story_len = max_story_len
        self.use_softmax = True
        self.use_pe = use_pe
        self.use_rn = use_rn
        self.rn_p = rn_p
        self.rn_max_extra = rn_max_extra
        self.temporal_capacity = max_story_len + rn_max_extra + 1
        self.tying = tying
        self.use_relu = use_relu

        if tying == 'adjacent':
            # Adjacent tying: A_{k+1} = C_k, B = A_1, W^T = C_K
            # Stored as K+1 distinct matrices; forward indexes them as A_k = embeddings[k], C_k = embeddings[k+1].
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, embed_size, padding_idx=0)
                for _ in range(hops + 1)
            ])
            self.temporal_embeddings = nn.ModuleList([
                nn.Embedding(self.temporal_capacity, embed_size, padding_idx=0)
                for _ in range(hops + 1)
            ])
        else:
            # Layer-wise (RNN-like) tying: one A, one C, separate B and W, plus learned H.
            self.A = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.C = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.B_query = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.T_A = nn.Embedding(self.temporal_capacity, embed_size, padding_idx=0)
            self.T_C = nn.Embedding(self.temporal_capacity, embed_size, padding_idx=0)
            self.H = nn.Linear(embed_size, embed_size, bias=False)
            self.W_out = nn.Linear(embed_size, vocab_size, bias=False)

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

    def _jittered_time_idx(self, story, num_sentences, device):
        batch_size = story.size(0)
        real_mask = (story != 0).any(dim=2)
        rand = torch.rand(batch_size, num_sentences, device=device)
        dummy = ((rand < self.rn_p) & real_mask).long()
        base = torch.arange(num_sentences, 0, -1, device=device)
        shift = torch.cumsum(dummy.flip(1), dim=1).flip(1)
        idx = base.unsqueeze(0) + shift
        return torch.clamp(idx, max=self.temporal_capacity - 1)

    def _init_weights(self):
        """Initializes weights with mean=0, std=0.1 as per the paper."""
        if self.tying == 'adjacent':
            for m in self.embeddings:
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.weight[0], 0)
            for m in self.temporal_embeddings:
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.constant_(m.weight[0], 0)
        else:
            for emb in (self.A, self.C, self.B_query):
                nn.init.normal_(emb.weight, mean=0.0, std=0.1)
                nn.init.constant_(emb.weight[0], 0)
            for emb in (self.T_A, self.T_C):
                nn.init.normal_(emb.weight, mean=0.0, std=0.1)
                nn.init.constant_(emb.weight[0], 0)
            nn.init.normal_(self.H.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.W_out.weight, mean=0.0, std=0.1)

    def _hop_embeddings(self, k):
        """Return (A_k, C_k, T_A^k, T_C^k) for hop k under whichever tying scheme is active."""
        if self.tying == 'adjacent':
            return (self.embeddings[k], self.embeddings[k+1],
                    self.temporal_embeddings[k], self.temporal_embeddings[k+1])
        return (self.A, self.C, self.T_A, self.T_C)

    def forward(self, story, query):
        batch_size, num_sentences, _ = story.size()
        if self.use_rn and self.training:
            time_idx = self._jittered_time_idx(story, num_sentences, story.device)
        else:
            time_idx = torch.arange(num_sentences, 0, -1, device=story.device).unsqueeze(0).expand(batch_size, -1)

        # Mask padded memory slots: a slot is real iff it has any non-zero word index
        mem_mask = (story != 0).any(dim=2)  # (B, S)

        if self.tying == 'adjacent':
            u = self._encode_query(query, self.embeddings[0])
        else:
            u = self._encode_query(query, self.B_query)

        for k in range(self.hops):
            A, C, T_A, T_C = self._hop_embeddings(k)

            m = self._encode_sentences(story, A) + T_A(time_idx)
            c = self._encode_sentences(story, C) + T_C(time_idx)

            scores = torch.bmm(m, u.unsqueeze(2)).squeeze(2)
            if self.use_softmax:
                scores = scores.masked_fill(~mem_mask, -1e9)
                p = torch.softmax(scores, dim=-1)
            else:
                # Linear-start phase: zero out padded weights so they contribute nothing
                p = scores * mem_mask.float()

            o = torch.bmm(p.unsqueeze(1), c).squeeze(1)

            if self.tying == 'adjacent':
                u = u + o
            else:
                u = self.H(u) + o

            if self.use_relu:
                # Paper §5 / Appendix A †: ReLU on half the units of the internal state after each hop
                half = self.embed_size // 2
                u = torch.cat([F.relu(u[:, :half]), u[:, half:]], dim=-1)

        if self.tying == 'adjacent':
            W = self.embeddings[-1].weight
            logits = torch.matmul(u, W.T)
        else:
            logits = self.W_out(u)
        return logits

def prepare_data(questions, stories, word2idx, max_story_len=50,
                 max_sen_len=None, max_q_len=None):
    """Converts the raw text into padded PyTorch tensors."""
    X_story, X_query, Y_answer = [], [], []

    if max_sen_len is None:
        max_sen_len = max([len(BabiDataset.tokenize(s)) for story in stories for s in story["sentences"]])
    if max_q_len is None:
        max_q_len = max([len(BabiDataset.tokenize(q["question"])) for q in questions])
    
    for q in questions:
        story_idx = q["story_id"]
        story_text = stories[story_idx]["sentences"][-max_story_len:]

        story_tensor = torch.zeros((max_story_len, max_sen_len), dtype=torch.long)
        # Right-align: most recent sentence ends up at row max_story_len - 1
        # so time_idx [N, N-1, ..., 1] correctly maps row 49 -> time_idx 1 (newest).
        offset = max_story_len - len(story_text)
        for i, sentence in enumerate(story_text):
            tokens = BabiDataset.tokenize(sentence)
            for j, word in enumerate(tokens):
                story_tensor[offset + i, j] = word2idx.get(word, 1)
                
        query_tensor = torch.zeros(max_q_len, dtype=torch.long)
        for j, word in enumerate(BabiDataset.tokenize(q["question"])):
            query_tensor[j] = word2idx.get(word, 1)
            
        answer_idx = word2idx.get(q["answer"].lower(), 1)
        
        X_story.append(story_tensor)
        X_query.append(query_tensor)
        Y_answer.append(answer_idx)
        
    return torch.stack(X_story), torch.stack(X_query), torch.tensor(Y_answer)