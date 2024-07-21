import torch
import torch.nn.functional as F
import random
import re

class MakeMoreNN(torch.nn.Module):
    def __init__(self, words):
        super().__init__()

        EMBEDDING_SPACE_DIM = 20
        HIDDEN_LAYER_1_OUTPUT = 200
        BLOCK_SIZE = 3
        NUM_ITR = 200_000
        MINI_BATCH_SIZE = 64

        self.block_size = BLOCK_SIZE
        
        # Create vocabulary
        cleaned_names = self.preprocess_names(words)
        chars = sorted(list(set(''.join(cleaned_names))))
        self.stoi = {s:i+1 for i,s in enumerate(chars)}
        self.stoi['.'] = 0  # to represent start/end of the word
        self.itos = {i:s for s, i in self.stoi.items()}
        
        self.vocab_size = len(self.stoi)
        print(f"lenth of vocab: {self.vocab_size}")
        self.embedding_dim = EMBEDDING_SPACE_DIM
        self.hidden_size = HIDDEN_LAYER_1_OUTPUT

        # Initialize layers
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.fc1 = torch.nn.Linear(self.block_size * self.embedding_dim, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.vocab_size)

        # Prepare data
        self.prepare_data(cleaned_names)

        # train_model 
        print("Now training the model")
        self.train_model(NUM_ITR, MINI_BATCH_SIZE)
    
    def preprocess_names(self, names):
        def clean_name(name):
            # Convert to lowercase
            name = name.lower()
            # Remove special characters, keeping only letters and spaces
            name = re.sub(r'[^a-z\s]', '', name)
            # Remove extra spaces
            name = ' '.join(name.split())
            return name

        # Apply the cleaning function to each name in the list
        cleaned_names = [clean_name(name) for name in names]
        
        # Remove any empty strings that might result from names that were only special characters
        cleaned_names = [name for name in cleaned_names if name]
        
        return cleaned_names


    def forward(self, x):
        emb = self.embedding(x)
        h = torch.tanh(self.fc1(emb.view(-1, self.block_size * self.embedding_dim)))
        logits = self.fc2(h)
        return logits

    def construct_model_data(self, words):
        X, Y = [], []
        for w in words:
            context = [0] * self.block_size
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)

    def prepare_data(self, words):
        random.seed(42)
        random.shuffle(words)
        n1 = int(0.8 * len(words))
        n2 = int(0.9 * len(words))

        self.Xtr, self.Ytr = self.construct_model_data(words[:n1])
        self.Xdev, self.Ydev = self.construct_model_data(words[n1:n2])
        self.Xtest, self.Ytest = self.construct_model_data(words[n2:])

    def train_model(self, num_iterations, batch_size):
        optimizer = torch.optim.Adam(self.parameters())
        
        for i in range(num_iterations):
            # Mini batch construction
            ix = torch.randint(0, self.Xtr.shape[0], (batch_size,))
            
            # Forward pass
            logits = self(self.Xtr[ix])
            
            # Loss calculation
            loss = F.cross_entropy(logits, self.Ytr[ix])
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Learning rate scheduling
            lr = lr = 0.1 if i < num_iterations/2 else 0.01
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.step()
            
            if i % 10_000 == 0:
                print(f"-{i}--lr:{lr}--loss:{loss.item()}--log_loss:{loss.log10().item()}")
        
        return loss.item()

    def calculate_loss(self, X, Y):
        logits = self(X)
        loss = F.cross_entropy(logits, Y)
        return loss.item()

    def predict(self, prefix=None, max_length=20):
        if prefix is None:
            context = [0] * self.block_size
        else:
            context = [0] * (self.block_size - len(prefix)) + [self.stoi[c] for c in prefix]
            context = context[-self.block_size:]
        
        result = []
        for _ in range(max_length):
            logits = self(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            
            if ix == 0:  # End of word
                break
            
            result.append(self.itos[ix])
            context = context[1:] + [ix]
        
        return ''.join(result)