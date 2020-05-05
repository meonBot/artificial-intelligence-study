import numpy as np

# 1. dictionary(map) 생성
dict_key_val = {'key1': 1, 'key2': 2, 'key3':3}
print (type(dict_key_val))
print ('dict_key_val:', dict_key_val)

# 2. key 정렬
sort_key_result = sorted(dict_key_val.items(), key=lambda x:x[0], reverse = True) # lambda를 안쓸려면, def f1(x) return x[0] 펑션 생성해서 쓸수 있음.
print('sort_key_result:', sort_key_result)

# 3. val 정렬
sort_val_result = sorted(dict_key_val.items(), key=lambda x:x[1], reverse = False) # lambda를 안쓸려면, def f1(x) return x[0] 펑션 생성해서 쓸수 있음.
print('sort_val_result:', sort_val_result)

# 4. list를 정렬하고, 정렬된 list의 정렬전 index번호를 추출하는 방법
list_temp = [100, 200, 300, 400, 500]
dict_temp = {}
for idx, data in enumerate(list_temp):
    dict_temp[data]=idx #key:data, val:index로 구성
sort_dict_list = sorted(dict_temp.items(), key=lambda x:x[0], reverse = True) #key로 desc로 정렬
print('sort_dict_list:', sort_dict_list)
#정렬된 top3 항목의 index 번호 출력
for num, dict_data in enumerate(sort_dict_list):
    if num >= 3 : break
    print('dict_data:', dict_data, ' ,idx:', dict_data[1])
