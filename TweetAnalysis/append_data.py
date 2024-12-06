import json
import os

with open('result.json', 'r') as f:
    obj = json.load(f)
    for stock in obj:
        file_path = f'data/{stock}.csv'
        if os.path.isfile(file_path):
            with open(file_path, 'r') as fs:
                with open('output/' + file_path, 'w') as fd:
                    for index, line in enumerate(fs.readlines()):
                        line = line.strip()
                        if index == 0:
                            fd.write(f'{line},Tweet Score,Tweet Volume\n')
                        else:
                            fields = line.split(',')
                            if fields[0] in obj[stock]:
                                fd.write(f'{line},{obj[stock][fields[0]][0]},{obj[stock][fields[0]][1]}\n')
                            else:
                                fd.write(line + ',0.5,0\n')
                    print('Completed ' + stock)