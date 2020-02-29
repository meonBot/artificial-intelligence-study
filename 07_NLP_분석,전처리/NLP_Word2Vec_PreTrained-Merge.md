# Word2Vec 모델과 Pre-Trained 모델의 Feature벡터 merge방법

### 1. Merge모델 생성
```
### 1. 모델생성
num_features   = 300  # 문자 벡터 차원 수 (300:wiki, data가 적으므로 100차원만)
min_word_count = 1   # 최소 문자 수 : 해당 출현빈도수 이하의 단어는 무시(*학습하는데 부담이 되지 않는다면 일반적으로 작게 설정)
num_workers    = 4    # 병렬 처리 스레드 수
context        = 3    # 문자열 창 크기 (평균 문자 길이를 고려?: HS품명 평균3)
downsampling   = 1e-3 # 문자 빈도수 Downsample : 빈도수가 큰값에 대해 평준화적용
#negative_sampling = 64 # 5~64?
sg_type        = 1    # 0:CBOW(skip-gram보다 빠름), 1:skip-gram(느리지만 성능이 더좋음?)

model_name = '300features_1minwords_3context_sg1'
# 모델 생성
model = word2vec.Word2Vec(new_sentence, 
						  workers=num_workers, 
						  size=num_features, 
						  min_count=min_word_count,
						  window=context,
						  sample=downsampling,
						  #negative = negative_sampling,
						  sg = sg_type) 
print(model)

# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)
model.save(model_name)

if IS_UPLOAD:
	# predict 모델 저장/업로드
	# 3가지 생성파일 모두 필요
	upload_weight('', model_name, drive)
	upload_weight('', model_name + '.wv.vectors.npy', drive)
	upload_weight('', model_name + '.trainables.syn1neg.npy', drive)
	print('IS_W2V_CREATE END!')


### 2. 1차생성한 모델과 Pre-Trained 모델(Wiki, 6G)등 Merge
# 3가지 생성파일 모두 필요
gcp_download(model_name, key = '1aTFkJtAlb_FGvf2Q7B7q-IWFseOA4jz8') 
model_name_v = model_name + '.wv.vectors.npy'
gcp_download(model_name_v, key = '1M1DFuoOQeWCZ7au-idDF_XCl6ZVu-NQ_') 
model_name_t = model_name + '.trainables.syn1neg.npy'
gcp_download(model_name_t, key = '14PAkIBiVPI9GHvGIo6MOyU6XfBdtJKit')

#모델 로딩
model = Word2Vec.load('/tmp/' + model_name)
print(model)

#merge
model.intersect_word2vec_format(fname=pre_trained_feature, binary=True, encoding='utf-8', unicode_errors='ignore')
print(model)

# predict 모델 저장/업로드
model_name = model_name + '_merge'
model.save(model_name)
upload_weight('', model_name, drive)
upload_weight('', model_name + '.wv.vectors.npy', drive)
upload_weight('', model_name + '.trainables.syn1neg.npy', drive)
```

### 2. 훈련모델에 적용(추출한 어휘의 가중치 벡터로 업데이트)
```
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

model_name = '300features_1minwords_3context_sg1_merge' 
gcp_download(model_name, key = '1aTFkJtAlb_FGvf2Q7B7q-IWFseOA4jz8') 
model_name_v = model_name + '.wv.vectors.npy'
gcp_download(model_name_v, key = '1M1DFuoOQeWCZ7au-idDF_XCl6ZVu-NQ_') 
model_name_t = model_name + '.trainables.syn1neg.npy'
gcp_download(model_name_t, key = '14PAkIBiVPI9GHvGIo6MOyU6XfBdtJKit')

#모델 로딩
model = Word2Vec.load('/tmp/' + model_name)
print(model)

idx2word = model.wv.index2word
for idx in range(len(idx2word)):
	word = idx2word[idx]
	embeddings_index[word] = model.wv[word]# word: 어휘의 feature vector
	if(idx==0): print('word:', ' embeddings_index[word]:', embeddings_index[word])

# Embedding Layer의 가중치 행렬값으로 업데이트 
embedding_matrix = np.zeros((len(input_vocab.tokenizer.word_index) + 1, EMBEDDING_DIM))
for word, i in input_vocab.tokenizer.word_index.items():
embedding_vector = embeddings_index.get(word)
if embedding_vector is not None:
	# words not found in embedding index will be all-zeros.
	embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)
```
