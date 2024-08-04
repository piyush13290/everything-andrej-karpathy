import torch
import math

class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        assert d_model % 2 == 0, "d_model must be even in this implementation. See comments for explanation."

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()

        pe.require_grad = False

        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # wow, x has no role here but i guess nn.module neds forward func as this 
        return self.pe
    
    # Explanation of the assert statement:
    """
    The assert statement checks if d_model is even. This is necessary because:

    1. The positional encoding is computed pairwise for sine and cosine functions.
    2. The implementation assumes d_model is even by iterating with a step of 2.
    3. If d_model were odd, the last dimension would be left uninitialized.
    4. This could lead to inconsistent behavior or reduced effectiveness of the encoding.

    To support odd d_model values, the implementation would need to be modified
    to handle the last dimension separately when d_model is odd.
    """

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)
        # why only 3 --> segment has only 3 possible vales, either part 
        # of the first segment, or second or paddings 
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
