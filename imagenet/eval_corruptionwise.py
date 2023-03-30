from glob import glob
import numpy as np
import re
def read_file(filename):
    lines = open(filename, "r").readlines()
    acc = []
    bri = []
    nll = []
    corr_name_all = []
    for line in lines:
        if "error : " in line:
            corr_name = line.split(" ")[6][1:-1].split("_")[0]
            corr_name = re.sub(r'\d+', '', corr_name)
            acc.append(float(line.split(":")[-3].split(",")[0].strip()[:-1]))
            bri.append(float(line.split(":")[-2].split(",")[0].strip()))
            nll.append(float(line.split(":")[-1].strip()))
            corr_name_all.append(corr_name)
    assert len(acc)==12
    print(corr_name_all)
    return np.mean(np.array(acc)), np.mean(np.array(bri)), np.mean(np.array(nll))
def read_file_corr(filename, acc, bri, nll):
    lines = open(filename, "r").readlines()
    for line in lines:
        if "error : " in line:
            corr_name = line.split(" ")[6][1:-1].split("_")[0]
            corr_name = re.sub(r'\d+', '', corr_name)
            if corr_name in acc:
                acc[corr_name].append(float(line.split(":")[-3].split(",")[0].strip()[:-1]))
                bri[corr_name].append(float(line.split(":")[-2].split(",")[0].strip()))
                nll[corr_name].append(float(line.split(":")[-1].strip()))
            else:
                acc[corr_name] = [float(line.split(":")[-3].split(",")[0].strip()[:-1])]
                bri[corr_name] = [float(line.split(":")[-2].split(",")[0].strip())]
                nll[corr_name] = [float(line.split(":")[-1].strip())]
                
    assert len(acc)==12
    return acc, bri, nll

def read_files(files):
    acc = {}
    bri = {}
    nll = {}
    if len(files) == 1:
        for f in files:
            acc, bri, nll = read_file_corr(f, acc, bri, nll)
        print("read", len(files), "files.")

        for key in acc:
            acc[key] = acc[key][0]
        for key in bri:
            bri[key] = bri[key][0]
        for key in nll:
            nll[key] = nll[key][0]
        return acc, bri, nll
    else:
        for f in files:
            acc, bri, nll = read_file_corr(f, acc, bri, nll)
        print("read", len(files), "files.")

        for key in acc:
            acc[key] = np.mean(np.array(acc[key]))
        for key in bri:
            bri[key] = np.mean(np.array(bri[key]))
        for key in nll:
            nll[key] = np.mean(np.array(nll[key]))
        return acc, bri, nll
        
print("\n\nread petalfim files:")
acc, bri, nll = read_files(glob("output/petalfim/petalfim[0-9]_*.txt"))
ordered_corr = ['gaussian', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic', 'pixelate', 'jpeg']
print("\nError")
avg_err=0.0
cnt=0
for c in ordered_corr:
    avg_err+=acc[c]
    cnt+=1
    print(c + "," + str(round(acc[c],2)))
avg_err=avg_err/cnt

print("\nBrier")
avg_bri=0.0
cnt=0
for c in ordered_corr:
    avg_bri+=bri[c]
    cnt+=1
    print(c + "," + str(round(bri[c],4)))
avg_bri=avg_bri/cnt

print("\nNLL")
avg_nll=0.0
cnt=0
for c in ordered_corr:
    avg_nll+=nll[c]
    cnt+=1
    print(c + "," + str(round(nll[c],4)))

print("Average Error over all the 10 orders:",avg_err)
print("Average Brier over all the 10 orders:",avg_bri)
print("Average NLL over all the 10 orders:",avg_nll)


print("\n\nread petalfim files:")
acc, bri, nll = read_files(glob("output/petalfim/petalfim[0-9]_*.txt"))
ordered_corr = ['gaussian', 'shot', 'impulse', 'defocus', 'glass', 'motion', 'zoom', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic', 'pixelate', 'jpeg']
print("\nError")
avg_err=0.0
cnt=0
for c in ordered_corr:
    avg_err+=acc[c]
    cnt+=1
    print(c + "," + str(round(acc[c],2)))
avg_err=avg_err/cnt

print("\nBrier")
avg_bri=0.0
cnt=0
for c in ordered_corr:
    avg_bri+=bri[c]
    cnt+=1
    print(c + "," + str(round(bri[c],4)))
avg_bri=avg_bri/cnt

print("\nNLL")
avg_nll=0.0
cnt=0
for c in ordered_corr:
    avg_nll+=nll[c]
    cnt+=1
    print(c + "," + str(round(nll[c],4)))

print("Average Error over all the 10 orders:",avg_err)
print("Average Brier over all the 10 orders:",avg_bri)
print("Average NLL over all the 10 orders:",avg_nll)

