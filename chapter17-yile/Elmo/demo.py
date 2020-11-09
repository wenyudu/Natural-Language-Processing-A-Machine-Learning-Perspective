from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder(cuda_device=0)

context_tokens = [['I', 'love', 'you', '.'], ['Sorry', ',', 'I', 'don', "'t", 'love', 'you', '.']]

elmo_embedding, elmo_mask = elmo.batch_to_embeddings(context_tokens)

print(elmo_embedding)
print(elmo_mask)