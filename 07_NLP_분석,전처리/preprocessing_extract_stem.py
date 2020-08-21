
### 설명
#1) socks6264100p16whitel iphone9.1  1.25kg -> socks6264100 p16 whitel  iphone9.1  1.25 kg
#: 1차로, 문자+숫자(실수)단위로 먼저 분리
#: [a-zA-Z]*[\d*\.\d+]*
#
#2)  socks6264100 iphone9.1 ->  socks 6264100 iphone 9.1
#: 2차로, 문자 숫자(실수)를 분리 
#: ([a-zA-Z]+)([\d*\.\d+]*)

#*python 정규표현식: http://pythonstudy.xyz/python/article/401-%EC%A0%95%EA%B7%9C-%ED%91%9C%ED%98%84%EC%8B%9D-Regex


def extract_extra_word_digit(word_list, compile_pattern):
    new_w_list = []
    for word in word_list:
        
        r = re.compile(compile_pattern)   
        match = r.match(word)
        if match == None: 
            new_w_list.append(word)
            continue
        mat_gr = match.groups()
        #print('match_items:', mat_gr)

        mat_list = []
        mat_list.append(mat_gr[0])
        mat_list.append(mat_gr[1])
        
        if word != ' '.join(mat_list):
            #print('w:', word, ' ,N:', ' '.join(mat_list))
            new_w_list.append(' '.join(mat_list))
        else: 
            new_w_list.append(word)
    return (' '.join(new_w_list)).split()

import re
#특수문자 제거(단, 오류품명은 제외), 정상품명은 알파벳으로 구성된 단어만 사용
def get_word_cleaning_sentence(input_list, label, skip_word_cnt_vec, MIN_WORD_LEN, is_rmv_2word=False):
    new_word_list = []
    
    p = re.compile('[a-zA-Z]*[\d*\.\d+]*') # socks6264100p16whitel 1.25kg -> socks6264100 p16 whitel 1.25 kg
    word_list = p.findall(input_list)
    word_list = (' '.join(word_list)).split() #연속된공백을 하나로 줄여줌        

    word_list = extract_extra_word_digit(word_list, "([a-zA-Z]+)([\d*\.\d+]*)") #영문+단어를 분리하고 기존 단어뒤에 추가, socks6264100 iphone9.1 -> socks 6264100 iphone 9.1
    print('2word_list:', word_list)

    for word in word_list: 
        #단어가 모두 제거된 문장은 정상품명만 skip 시
        if(word.strip()==''): continue  #strip: trim
        #소문자 변환, 좌우공백제거
        word = word.lower().strip() 
        
        #참고: https://programmers.co.kr/learn/courses/21/lessons/1694 => 임의제거/원형추출이 약간의 성능저하를 일으킴?
        #불용어 제거(ex: i, you ,this...)
        #if not word in STOP_WORD_SET: 
        #     word = STEMMER.stem(word)  # 어간추출
        #     new_word_list.append(word)
        #else: skip_word_cnt_vec[0] = skip_word_cnt_vec[0] + 1 #print('>>> stops word:', word)

        #알파벳 단어만 사용-> ex) ab-cd도 사용
        # if word.isalpha() == True: new_word_list.append(word.strip())
        # else: skip_word_cnt_vec[0] = skip_word_cnt_vec[0] + 1 # call by reference

        #불용어 제거
        #if word in STOP_WORD_SET:
        #    skip_word_cnt_vec[0] = skip_word_cnt_vec[0] + 1
        #    continue
        
        #숫자로만 구성된 단어는 사용안함, 깨진문자(ex: ?놁쓬)도 사용안함
        if word.startswith("?")==False: new_word_list.append(word)  # word.isdigit()==False and -> (08/20) 숫자도 사용함
        #if(word.startswith("?")==False): new_word_list.append(word)
        else: skip_word_cnt_vec[0] = skip_word_cnt_vec[0] + 1 # call by reference  
        
    # 1단어이하이면, 사용하지 않음
    if (is_rmv_2word==True and (self.rule=="" or self.rule=="HS_GD") and len(new_word_list)<=MIN_WORD_LEN):
        if len(new_word_list) > 0 : 
            self.word_cnt2_under.append([label, ' '.join(new_word_list)])
        new_word_list = []
    
    return ' '.join(new_word_list), ' '

remove_word_cnt=0
MIN_WORD_LEN=2
origin_str = 'nike m spark running socks6264100p16whitel iphone9.1 123abc 0.25kg'
print('origin_str:', origin_str)
data, _ = get_word_cleaning_sentence(origin_str, '11', remove_word_cnt, MIN_WORD_LEN, is_rmv_2word=False)
print(data)

### 결과
#origin_str: nike m spark running socks6264100p16whitel iphone9.1 123abc 0.25kg
#1word_list: ['nike', 'm', 'spark', 'running', 'socks6264100', 'p16', 'whitel', 'iphone9.1', '123', 'abc', '0.25', 'kg']
#2word_list: ['nike', 'm', 'spark', 'running', 'socks', '6264100', 'p', '16', 'whitel', 'iphone', '9.1', '123', 'abc', '0.25', 'kg']
#nike m spark running socks 6264100 p 16 whitel iphone 9.1 123 abc 0.25 kg
