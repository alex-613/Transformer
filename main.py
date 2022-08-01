import torch
import torch.nn as nn

# Build single headed self-attention
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # Save the embedding size, the number of attention heads and the dimension for the heads
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads # This is the size of the head (query)

        assert (self.head_dim*heads == embed_size), "Embed size needs to be divisible by heads"

        # Lets define the values, keys, queries and fully connected output after concatenation after attention

        # Please note that the keys, values and queries all have the same dimension
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # We also combine all of the heads and their dimensions together to get the final fc_output
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        # This is the number of queries we are sending in each time, not the length of the query!
        N = query.shape[0]

        # Depending on where you use the attention, these lengths may differ, this corresponds to source and target length

        # Note that these are the lengths of the heads
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # print(f"Value: {value_len}, Key: {key_len}, Query:{query_len}")
        # Observation: the query and the value does not need to match up. Although value and key has to!

        # Split embedding/query into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        print(f"Value: {values.shape}, Key: {keys.shape}, Query:{keys.shape}")

        # Pass the values, keys and queries into the fully connected layer and get the output out.
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Multiply the keys with the queries, call it energy
        # Einsum is used for multidimensional matrix multiplication, we multiply across the head_dimension for all the keys and queries
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # Queries shape: (N, query_len, heads, heads_dim)
        # Keys shape: (N, key_len, heads, heads_dim)
        # Energy shape: (N, heads, query_len, key_len)

        # Option to mask out everything beyond an index in a sequence, needed for decoder
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Do the softmax and scalling
        attention = torch.softmax(energy /(self.embed_size ** (1/2)), dim=3)

        # Do another dot product between the energy and the values
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum: (N, query_len, heads, head_dim) then flatten last two dimensions
        # As said before, this fully connected layer is to combine all of the heads together, not for the actual fully connected part
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        # Define the attention block
        self.attention = SelfAttention(embed_size, heads)

        # Layer norm is similar to batch norm.
        # Batch takes average across batch, layer takes average across every example.
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # We got a norm after the attention and after the FC layer

        # Define your FC layer
        self.feed_forward = nn.Sequential(
            # Output from multi-headed attention goes into here
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
            # Input to the next layer of multi-headed attention
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Multi-headed attention
        attention = self.attention(value, key, query, mask)
        # Residual connection + regularization
        x =  self.dropout(self.norm1(attention + query))
        # FF part
        forward = self.feed_forward(x)
        # Residual connection + regularization
        out = self.dropout(self.norm2(forward + x))

        return out

# Inorder to stack multi attention layers together, we define the Encoder
class Encoder(nn.Module):
    def __init__(self,src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length):

        super(Encoder, self).__init__()
        # What is the embedding size you want to achieve at the end of the encoding
        self.embed_size = embed_size
        self.device = device
        # Convert words into matrix of weights, note that this is learnable
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # Convert word positions into matrix of weights, note that this is also learnable
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Define a list of TransformerBlocks
        self.layers = nn.ModuleList(
        [
            TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion
                ) for _ in range(num_layers)])

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # N examples sent in of certain sequence length
        N, seq_length = x.shape
        # Create a range list from 0 to sequence length, for every example. Expanding singleton dimensions in the proecess
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # The line above creates a positional encoding matrix consisting of many rows of 1,2,3,4,... seq_length
        # So if seq_length = 10, N = 2
        # Positions =
        """ [[1,2,3,4,5,6,7,8,9,10],
             [1,2,3,4,5,6,7,8,9,10]]
        """

        # Put the positional and word embedding together and adding regularization
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # The only thing we are sending in is this positional embedding

        # Run the encoder network with Q, K, and V all from the input
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
            )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, value, key, src_mask, trg_mask):
        # Once again have attention, but note that you must put in the mask ask well in order to mask out the subsequent terms
        attention = self.attention(x,x,x,trg_mask)
        # Add a res connection
        query = self.dropout(self.norm(attention + x))
        # Finally pass the query into the attention block
        out = self.transformer_block(value, key, query, src_mask)
        return out
        
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
    
        super(Decoder, self).__init__()
        self.device = device
        # Word embedding and positional embedding layers
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # Have multiple decoder blocks
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)] 
            )
        # Fully connected output
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # Perform the positional encoding thing again
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # Add layers according to users wish
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cuda", max_length=100):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
        )

        self.decoder = Decoder(
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
        )


        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src  != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)

        # This is useful if you want to ignore parts of the source sentence for any reason
        
        return src_mask.to(self.device)
        
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape 
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            N, 1, trg_len, trg_len
            )

        # Create a triangular mask. Like shown below.
        """
        111111
        011111
        001111
        000111
        000011
        000001
        000000
        """
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x= torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [ 1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    out = model(x,trg[:, :-1])
    print(out.shape)
