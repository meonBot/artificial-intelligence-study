print('-start excel pre-processing-')

#http://pythonstudy.xyz/python/article/207-CSV-%ED%8C%8C%EC%9D%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
import csv

data_path = ""
file_name = "" 

tot_line = []
tot_cnt = 0
for num in range(33):
    f = open(data_path+file_name +"_"+ str(num+1)+".csv", 'r', encoding='utf-8')
    rdr = csv.reader(f)
    cnt = 0
    for line in rdr:
        tot_line.append(''.join(line))
        cnt += 1
    f.close()
    tot_cnt += cnt
    print('num:', (num+1), ' ,cnt:', cnt)
print('len:', tot_cnt)

uniq_gd = list(set(tot_line))
print('len(uniq_gd):', len(uniq_gd))


save_file_name = ""
f_w = open(data_path + save_file_name , 'w')
for data in uniq_gd:
    f_w.write(data + "\n")
f_w.close()
