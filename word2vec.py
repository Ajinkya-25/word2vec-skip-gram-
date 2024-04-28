import numpy as np

np.random.seed(42)
def tokenize(text):
    return text.lower().split()


def create_vocab(tokens):
    vocab = set(tokens)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    return vocab, word_to_index, index_to_word

def generate_training_samples(tokens, context_window_size=1):
    samples = []
    for i, word in enumerate(tokens):
        for j in range(i - context_window_size, i + context_window_size + 1):
            if j != i:
                if 0 < j < len(tokens):
                    context_word = tokens[j]
                    samples.append((context_word, word))
    return samples


class word2vec:
    def __init__(self, vocab_size, embedding_dim):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W_in = np.random.randn(vocab_size, embedding_dim)
        self.W_out = np.random.randn(embedding_dim, vocab_size)

    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x))
        print("Shape of exp_scores array:", exp_scores.shape)
        return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    def forward_pass(self, context_word_index):

        context_embedding = self.W_in[context_word_index]
        hidden = np.dot(context_embedding, self.W_out.T)
        final = np.dot(hidden, self.W_out.T)
        op = self.softmax(final)
        return op

    def backpass(self, context_word_index, target_word_index, learning_rate):
        np.random.seed(42)
        probe=self.forward_pass(context_word_index)
        for i,_ in enumerate(probe):
            if i!=target_word_index:
                probe[i]-=1
        grad_out = np.dot(self.W_in[context_word_index].T, probe)
        self.W_out -= learning_rate * grad_out
        grad_in=np.dot(probe,self.W_out.T)
        self.W_in -= learning_rate*grad_in

    def similarity(self, word1, word2):
        if word1 not in word_to_index or word2 not in word_to_index:
            return "One or both of the words are not in the vocabulary."
        else:
            index1 = word_to_index[word1]
            index2 = word_to_index[word2]
            vector1 = self.W_in[index1]
            vector2 = self.W_in[index2]
            cos_sim = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            return cos_sim
    def train(self,epoch,samples,learning_rate=0.01):
        for epochs in range(epoch):
            np.random.seed(42)
            total_loss=0
            np.random.shuffle(samples)
            for context_word_index,target_word_index in samples:
                self.backpass(context_word_index, target_word_index, learning_rate)
                probs = self.forward_pass(context_word_index)
                if epochs==epoch-1:
                    print(probs)
                loss = -np.log(probs[:])
                total_loss += np.sum(loss)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {total_loss / len(samples)}")


def initialise_embeddings(vocab_size,embeddings_dim):
    return np.random.randn(vocab_size,embeddings_dim)

text = "The quick brown fox jumps over the lazy dog. The dog barks loudly as the fox runs away. Birds chirp in the trees, and squirrels scurry on the ground. The sun shines brightly in the clear blue sky. People laugh and chat in the nearby park. Children play games and chase butterflies. Cars zoom by on the busy streets, honking their horns. The city buzzes with energy and life. At night, the stars twinkle overhead, and the moon casts a soft glow. Crickets chirp, and owls hoot in the darkness. The world is full of wonders and mysteries waiting to be explored."

token = tokenize(text)
vocab, word_to_index, index_to_word = create_vocab(token)
vocab_size = len(vocab)
embeddings = initialise_embeddings(vocab_size, 2)

samples = generate_training_samples(token)
indexed_samples = [(word_to_index[context_word], word_to_index[target_word]) for context_word, target_word in samples]

word2vec = word2vec(vocab_size, 2)
word2vec.train(100, indexed_samples)

