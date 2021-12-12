import sys,os,csv
from sys import argv as inputs
import pickle
#reading list from string import ast; ast.literal_eval(x)
if(len(inputs)<4):
    print("help:\n python data_extractor.py <window_size> <target_time_frame>(future) <skip_time> <taget_percentage>\n")
    sys.exit(1)
open_index        = 1
high_index        = 2
low_index         = 3
close_index       = 4

sucess            = 1
failure           = 2
middle            = 3

window_size             = int(inputs[1])
target_time_frame       = int(inputs[2])
skip_time               = int(inputs[3])
target_percentage       = int(inputs[4])
#crash_price_percent     = int(inputs[5])
#explode_percentage      = int(inputs[6])

file= open('final_file.csv')

csvreader = csv.reader(file)

rows = []
for row in csvreader:
    if(row!=[]):
        rows.append(list(map(float,[row[open_index],row[high_index],row[high_index],row[low_index],row[close_index]])))

successes = 0
failures  = 0
neutrals_loss  = 0 
neutrals_profit  = 0 
data = {'data':[],'label':[]}

for pointer in range(0,len(rows)-skip_time-window_size-target_time_frame,skip_time):
    columns_future          = list(zip(*rows[pointer+window_size:pointer+window_size+target_time_frame+1]))
    high_column             = list(columns_future[high_index])
    max_value               = max(high_column)
    current_buying_value    = rows[pointer+window_size-1][close_index]
    next_closing_value      = columns_future[close_index][-1]
    #print(current_buying_value,max_value,next_closing_value)
    profit = ((max_value-current_buying_value)/current_buying_value)*100
    loss   = ((current_buying_value-next_closing_value)/current_buying_value*100)
    data['data'].append(rows[pointer:pointer+window_size])

    if(profit>target_percentage):
        successes+=1
        data['label'].append(3)
        #print("sucess",profit)
    elif( loss> target_percentage):
        failures+=1
        data['label'].append(2)
        #print("failure",loss)
    else:
        if(loss>0):
            data['label'].append(1)
            neutrals_loss+=1
            #print("loss neutral",loss)
        else:
            data['label'].append(0)
            neutrals_profit+=1
            #print("profit neutral",-loss)

with open('data.pkl', 'wb') as handle:
    pickle.dump(data, handle)
print(successes,failures,neutrals_profit,neutrals_loss)
