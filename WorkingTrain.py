import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import BabiDataset

class MemN2N(nn.Module):
    def __init__(self, vocab_size, embed_size=20, max_story_len=50, hops=3):
        super(MemN2N, self).__init__()
        self.hops = hops
        self.embed_size = embed_size
        self.max_story_len = max_story_len
        
        # Adjacent weight tying: We need (hops + 1) embedding matrices.
        # A^1 = E[0], C^1 = E[1]
        # A^2 = E[1], C^2 = E[2]
        # B (query) = A^1 = E[0]
        # W^T (prediction) = C^K = E[-1]
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_size, padding_idx=0) 
            for _ in range(hops + 1)
        ])
        
        # Temporal encoding matrices T_A and T_C. 
        # Sentences are indexed in reverse order (1 is the most recent sentence).
        self.temporal_embeddings = nn.ModuleList([
            nn.Embedding(max_story_len + 1, embed_size, padding_idx=0)
            for _ in range(hops + 1)
        ])
        self._init_weights()

    def _init_weights(self):
        """Initializes weights with mean=0, std=0.1 as per the paper."""
        for m in self.embeddings:
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.weight[0], 0) # Constraint null symbol to zero
            
        for m in self.temporal_embeddings:
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.weight[0], 0) # Constraint null symbol to zero

    def forward(self, story, query):
        # story shape: (batch_size, num_sentences, max_words_per_sen)
        # query shape: (batch_size, max_words_per_query)
        
        batch_size, num_sentences, _ = story.size()
        
        # Time indices: [num_sentences, num_sentences-1, ..., 1]
        # Pad with 0 if needed, but we assume inputs are truncated/padded to num_sentences
        time_idx = torch.arange(num_sentences, 0, -1, device=story.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed query (Bag of Words) -> B = E[0]
        u = self.embeddings[0](query).sum(dim=1) # (batch_size, embed_size)
        
        for k in range(self.hops):
            # A^k = E[k], C^k = E[k+1]
            A = self.embeddings[k]
            C = self.embeddings[k+1]
            T_A = self.temporal_embeddings[k]
            T_C = self.temporal_embeddings[k+1]
            
            # Embed story memories: Bag of Words + Temporal Encoding
            # m_i = sum(A * x_ij) + T_A(i)
            m = A(story).sum(dim=2) + T_A(time_idx) # (batch_size, num_sentences, embed_size)
            
            # c_i = sum(C * x_ij) + T_C(i)
            c = C(story).sum(dim=2) + T_C(time_idx) # (batch_size, num_sentences, embed_size)
            
            # Attention weights: p_i = Softmax(u^T * m_i)
            # We use bmm (batch matrix multiplication) for the dot product
            scores = torch.bmm(m, u.unsqueeze(2)).squeeze(2) # (batch_size, num_sentences)
            p = torch.softmax(scores, dim=-1)
            
            # Output vector: o = sum(p_i * c_i)
            o = torch.bmm(p.unsqueeze(1), c).squeeze(1) # (batch_size, embed_size)
            
            # Update state: u^{k+1} = u^k + o^k
            u = u + o
            
        # Final prediction matrix W is tied to the last embedding matrix C^K (E[-1])
        # a_hat = Softmax(W * u) --> we just output logits for CrossEntropyLoss
        W = self.embeddings[-1].weight
        logits = torch.matmul(u, W.T) # (batch_size, vocab_size)
        
        return logits

def prepare_data(questions, stories, word2idx, max_story_len=50):
    """Converts the raw text into padded PyTorch tensors."""
    X_story, X_query, Y_answer = [], [], []
    
    # Calculate max lengths for padding
    max_sen_len = max([len(BabiDataset.tokenize(s)) for story in stories for s in story["sentences"]])
    max_q_len = max([len(BabiDataset.tokenize(q["question"])) for q in questions])
    
    for q in questions:
        story_idx = q["story_id"]
        story_text = stories[story_idx]["sentences"][-max_story_len:] # Keep only recent memory
        
        # Tokenize and pad story sentences
        story_tensor = torch.zeros((max_story_len, max_sen_len), dtype=torch.long)
        for i, sentence in enumerate(story_text):
            tokens = BabiDataset.tokenize(sentence)
            for j, word in enumerate(tokens):
                story_tensor[i, j] = word2idx.get(word, 1) # 1 is <unk>
                
        # Tokenize and pad query
        query_tensor = torch.zeros(max_q_len, dtype=torch.long)
        for j, word in enumerate(BabiDataset.tokenize(q["question"])):
            query_tensor[j] = word2idx.get(word, 1)
            
        # Tokenize answer
        answer_idx = word2idx.get(q["answer"].lower(), 1)
        
        X_story.append(story_tensor)
        X_query.append(query_tensor)
        Y_answer.append(answer_idx)
        
    return torch.stack(X_story), torch.stack(X_query), torch.tensor(Y_answer)

def main():
    # 1. Load Data
    dataset = BabiDataset(download=True)
    task_id = 1 # Let's train on Task 1 for the project demonstration
    stories, questions, vocab = dataset.load_task(task_id, train=True)
    
    # Convert to tensors
    X_story, X_query, Y = prepare_data(questions, stories, vocab)
    
    # 2. Initialize Model
    vocab_size = len(vocab)
    model = MemN2N(vocab_size=vocab_size, embed_size=20, max_story_len=50, hops=3)
    
    # 3. Setup Training
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    epochs = 100 # Paper trains for 100 epochs
    batch_size = 32
    num_samples = len(Y)
    
    print(f"\nTraining MemN2N on Task {task_id} with vocabulary size {vocab_size}...")
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for i in range(0, num_samples, batch_size):
            batch_story = X_story[i:i+batch_size]
            batch_query = X_query[i:i+batch_size]
            batch_y = Y[i:i+batch_size]
            
            optimizer.zero_grad()
            
            logits = model(batch_story, batch_query)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0) 
            optimizer.step()

            # Since reduction='sum', loss.item() is already the total for this batch
            total_loss += loss.item() 
            preds = torch.argmax(logits, dim=1)

            correct += (preds == batch_y).sum().item()
            
        scheduler.step() # Step the learning rate down if needed
        
        acc = correct / num_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/num_samples:.4f} | Accuracy: {acc:.4f}")

            
    # --- EVALUATION ON TEST SET ---
    print("\nLoading Test Data...")
    test_stories, test_questions, _ = dataset.load_task(task_id, train=False)
    
    X_story_test, X_query_test, Y_test = prepare_data(test_questions, test_stories, vocab)
    
    print("Evaluating...")
    model.eval() 
    
    with torch.no_grad():
        test_logits = model(X_story_test, X_query_test)
        test_preds = torch.argmax(test_logits, dim=1)
        
        correct_test = (test_preds == Y_test).sum().item()
        test_acc = correct_test / len(Y_test)
        
    print(f"Final Test Accuracy on Task {task_id}: {test_acc:.4f} (Error Rate: {(1 - test_acc) * 100:.1f}%)")

if __name__ == "__main__":
    main()