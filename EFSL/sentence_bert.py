import torch

from sentence_transformers import models, losses, evaluation
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer, util, InputExample

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = "usc-isi/sbert-roberta-large-anli-mnli-snli"
word_embedding_model = models.Transformer(model_name).to(device)

# add special tokens
tokens = ['chinavirus', 'cherry picker', 'china virus','coronaviruschina', 'ccpvirus',
          'kungflu','chinese virus','wuhanvirus', 'wuhan virus', 'maskless', 'womensuch', 'walkaway',
          'antimask','antivaccine', 'novaccine', 'maskoff', 'boomer', 'maskfree', 'babyboomer',
          'boomerremover', 'boomer remover', 'wuflu']

word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 
                                  pooling_mode_mean_tokens=True, 
                                  pooling_mode_cls_token=False, 
                                  pooling_mode_max_tokens=False)
# update the model
sent_model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)