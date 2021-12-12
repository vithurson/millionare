import os,sys
window= sys.argv[1]
first_time=1
month=1
with open("final_file.csv","w") as writer:
    for year in range(2017,2022):
        month=1
        while(month <=12):
           if(year==2021 and month==12):
                break
           if(first_time==1):
                month=8
                first_time=0
           file_name="BTCUSDT-"+window+"-"+str(year)+"-"+str(f"{month:02}")
           zip_file=file_name+".zip"
           url="https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/"+window+"/"+zip_file
           os.system("wget "+url)
           os.system("unzip "+zip_file)
           csv_file=file_name+".csv"
           with open(csv_file,"r") as cur_file:
                content = cur_file.read()
                writer.write(content+'\n')
           os.system("rm " + csv_file)
           month+=1
os.system("rm *.zip*")
