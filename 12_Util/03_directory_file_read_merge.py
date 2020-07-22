print('-start excel pre-processing-')


#http://pythonstudy.xyz/python/article/207-CSV-%ED%8C%8C%EC%9D%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
import os

data_path1 = ""
file_list1 = os.listdir(data_path1)

tot_line = []
tot_cnt = 0

for idx, file_name in enumerate(file_list1):
    f = open(data_path1+file_name, 'r', encoding='utf-8')
    cnt = 0
    while True:
        line = f.readline()
        if not line: break
        tot_line.append(line)
        cnt += 1
    f.close()
    tot_cnt += cnt
    print('file idx:', (idx+1), ' ,cnt:', cnt, ' ,tot_cnt:', tot_cnt)
print('1. len:', tot_cnt)


data_path2 = ""
file_list2 = os.listdir(data_path2)

for idx, file_name in enumerate(file_list2):
    f = open(data_path2+file_name, 'r', encoding='utf-8')
    cnt = 0
    while True:
        line = f.readline()
        if not line: break
        tot_line.append(line)
        cnt += 1
    f.close()
    tot_cnt += cnt
    print('file idx:', (idx+1), ' ,cnt:', cnt, ' ,tot_cnt:', tot_cnt)
print('2. len:', tot_cnt)

uniq_gd = list(set(tot_line))
print('3. len(uniq_gd):', len(uniq_gd))

save_file_name = "DBPidea_Wikipidea_Uniq.csv"
data_path3 = ""
f_w = open(data_path3 + save_file_name , 'w', -1, 'utf-8')
for data in uniq_gd:
    f_w.write(data + "\n")
f_w.close()
