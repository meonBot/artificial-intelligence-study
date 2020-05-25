# python Excel Read/Write

- 첫번째 열의 cell을 읽어서 숫자가 아니면 skip

```
print('-start excel pre-processing-')

# 참고: https://myjamong.tistory.com/51
from openpyxl import load_workbook

data_path = ""
file_name = ""

# data_only=Ture로 해줘야 수식이 아닌 값으로 받아온다.
load_wb = load_workbook(data_path + file_name, data_only=True)
# 시트 이름으로 불러오기
load_ws = load_wb['Sheet1']

def check_is_digit(val):
    if type(row[0].value) != int:
        if row[0].value == None:
            #print('None(', row_idx+1, ') ', row[0].value, ',', row[4].value)
            return False
        elif type(row[0].value) == str :
            try:
                int(row[0].value) #0101은 정상통과, 영문자는 오류처리되어 continue
            except Exception as ex:
                #print('Str(', row_idx+1, ') ', row[0].value, ',', row[4].value + ',ex:', ex)
                return False
        else :
            #print(type(row[0].value), ' (', row_idx+1, ') ', row[0].value, ',', row[4].value)
            return False
    #else: print('INT(', row_idx+1, ') ', row[0].value, ',', row[4].value)
    return True


print('\n-----모든 행과 열 출력-----')
cnt = 0
all_rows = []
for row_idx, row in enumerate(load_ws.rows):
    # 1)첫번째 열이 숫자가 아니면 제외
    if check_is_digit(row[0].value) == False : continue

    cnt += 1
    row_value = []
    row_value.append('cnt:' + str(cnt) + ':')
    for col_idx, cell in enumerate(row):
        row_value.append(cell.value)

    all_rows.append(row_value)
    #print(row_value)
    #if cnt >= 100 :  break
print('cnt:', cnt)
print('len:', len(all_rows))
print(all_rows[0:10])
print(all_rows[-1])

print('-end excel pre-processing-')

# from openpyxl import Workbook
# write_wb = Workbook()
# # 이름이 있는 시트를 생성
# # write_ws = write_wb.create_sheet('생성시트')
# # Sheet1에다 입력
# write_ws = write_wb.active
#
# # 행 단위로 추가
# write_ws.append([1, 2, 3])
# save_file_name = "2017품목분류표(HSK)_DATA추출.xlsx"
# write_wb.save(data_path + save_file_name)
#
# print('-end excel save')
```
