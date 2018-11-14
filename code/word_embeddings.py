import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print(type(model))
if 'hello' in model:
    print("thank fuck")

print(model['hello'].shape)
# print(model.most_similar(positive=['woman', 'king'], negative=['man']))
